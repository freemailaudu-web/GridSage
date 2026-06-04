from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _edge_key(bus1: str, bus2: str) -> Tuple[str, str]:
    return tuple(sorted((str(bus1), str(bus2))))


def _default_constraints(constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {
        "max_switch_operations": 1,
        "allow_open_main_feeder": False,
        "allow_multi_nop": False,
    }
    if isinstance(constraints, dict):
        merged.update(constraints)
    return merged


def _normalize_bus_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower().startswith("b"):
        try:
            return "b%s" % int(float(text[1:]))
        except (TypeError, ValueError):
            return text
    try:
        return "b%s" % int(float(text))
    except (TypeError, ValueError):
        return text


def _find_path_edges(
    base_edges: List[Dict[str, Any]],
    source: str,
    target: str,
) -> Optional[List[Dict[str, Any]]]:
    adjacency: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for edge in base_edges:
        a = edge["source"]
        b = edge["target"]
        adjacency.setdefault(a, []).append((b, edge))
        adjacency.setdefault(b, []).append((a, edge))

    queue = deque([(source, [])])
    seen = set()
    while queue:
        bus, path = queue.popleft()
        if bus == target:
            return path
        if bus in seen:
            continue
        seen.add(bus)
        for nxt, edge in adjacency.get(bus, []):
            if nxt not in seen:
                queue.append((nxt, path + [edge]))
    return None


def _topology_status(
    bus_ids: Iterable[str],
    edges: List[Dict[str, Any]],
    root_bus: str,
) -> Dict[str, Any]:
    buses = list(dict.fromkeys(bus_ids))
    adjacency: Dict[str, List[str]] = {bus: [] for bus in buses}
    for edge in edges:
        a = edge["source"]
        b = edge["target"]
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    visited = set()
    queue = deque([root_bus])
    while queue:
        bus = queue.popleft()
        if bus in visited:
            continue
        visited.add(bus)
        for nxt in adjacency.get(bus, []):
            if nxt not in visited:
                queue.append(nxt)

    node_count = len(buses)
    edge_count = len(edges)
    is_connected = node_count > 0 and len(visited) == node_count
    is_radial = is_connected and edge_count == node_count - 1
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "is_connected": is_connected,
        "is_radial": is_radial,
        "has_cycle": is_connected and edge_count >= node_count,
        "islands": 0 if is_connected else max(1, node_count - len(visited)),
    }


def _active_edges_for_plan(
    base_edges: List[Dict[str, Any]],
    open_line_id: Optional[str],
    close_nop: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    active = [dict(edge) for edge in base_edges if edge["id"] != open_line_id]
    if close_nop:
        active.append({
            "id": close_nop["line_id"],
            "source": close_nop["source"],
            "target": close_nop["target"],
            "edge_type": "nop_line",
            "nop_id": close_nop["id"],
        })
    return active


def generate_reconfiguration_plans_from_records(
    bus_ids: Iterable[str],
    base_edges: List[Dict[str, Any]],
    nop_edges: List[Dict[str, Any]],
    root_bus: str = "b1",
    constraints: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    constraints = _default_constraints(constraints)
    root_bus = _normalize_bus_id(root_bus) or "b1"
    buses = [_normalize_bus_id(bus) for bus in bus_ids]
    normalized_base_edges = []
    for edge in base_edges:
        normalized_base_edges.append({
            "id": str(edge["id"]),
            "source": _normalize_bus_id(edge["source"]),
            "target": _normalize_bus_id(edge["target"]),
            "edge_type": edge.get("edge_type", "base_line"),
        })

    status_r0 = _topology_status(buses, normalized_base_edges, root_bus)
    plans = [{
        "plan_id": "R0",
        "plan_name": "R0 - keep original radial topology",
        "close_nop_id": None,
        "close_nop_bus1": None,
        "close_nop_bus2": None,
        "open_line_id": None,
        "open_line_from_bus": None,
        "open_line_to_bus": None,
        "loop_edges": [],
        "active_edges": normalized_base_edges,
        "is_radial": status_r0["is_radial"],
        "is_connected": status_r0["is_connected"],
        "node_count": status_r0["node_count"],
        "edge_count": status_r0["edge_count"],
        "has_cycle": status_r0["has_cycle"],
        "islands": status_r0["islands"],
        "closed_nop_count": 0,
        "opened_line_count": 0,
        "switch_operation_count": 0,
        "description_cn": "R0: Do not close any NOP and maintain the original radial topology.",
    }]

    if constraints.get("max_switch_operations", 1) < 1:
        return plans

    plan_index = 1
    for raw_nop in sorted(nop_edges, key=lambda item: str(item.get("id", ""))):
        nop = {
            "id": str(raw_nop["id"]),
            "source": _normalize_bus_id(raw_nop["source"]),
            "target": _normalize_bus_id(raw_nop["target"]),
            "line_id": str(raw_nop.get("line_id") or "line_for_%s" % raw_nop["id"]),
        }
        if not nop["source"] or not nop["target"]:
            continue
        path_edges = _find_path_edges(normalized_base_edges, nop["source"], nop["target"])
        if not path_edges:
            continue

        for open_edge in path_edges:
            if not constraints.get("allow_open_main_feeder", False):
                if root_bus in {open_edge["source"], open_edge["target"]}:
                    continue

            active_edges = _active_edges_for_plan(normalized_base_edges, open_edge["id"], nop)
            status = _topology_status(buses, active_edges, root_bus)
            if not (status["is_connected"] and status["is_radial"]):
                continue

            loop_edges = [
                {
                    "id": edge["id"],
                    "source": edge["source"],
                    "target": edge["target"],
                    "edge_type": "base_line",
                }
                for edge in path_edges
            ]
            loop_edges.append({
                "id": nop["line_id"],
                "source": nop["source"],
                "target": nop["target"],
                "edge_type": "nop_line",
                "nop_id": nop["id"],
            })

            plan_id = "R%s" % plan_index
            plans.append({
                "plan_id": plan_id,
                "plan_name": "%s - close %s, open %s" % (plan_id, nop["id"], open_edge["id"]),
                "close_nop_id": nop["id"],
                "close_nop_bus1": nop["source"],
                "close_nop_bus2": nop["target"],
                "open_line_id": open_edge["id"],
                "open_line_from_bus": open_edge["source"],
                "open_line_to_bus": open_edge["target"],
                "loop_edges": loop_edges,
                "active_edges": active_edges,
                "is_radial": True,
                "is_connected": True,
                "node_count": status["node_count"],
                "edge_count": status["edge_count"],
                "has_cycle": False,
                "islands": 0,
                "closed_nop_count": 1,
                "opened_line_count": 1,
                "switch_operation_count": 2,
                "description_cn": (
                    "%s: Close %s (%s-%s), open the original branch %s (%s-%s),"
                    "The reconstructed topology is connected, acyclic, and radial."
                ) % (
                    plan_id,
                    nop["id"],
                    nop["source"],
                    nop["target"],
                    open_edge["id"],
                    open_edge["source"],
                    open_edge["target"],
                ),
            })
            plan_index += 1

    return plans


def _grid_base_edge_records(grid) -> List[Dict[str, Any]]:
    records = []
    for line in list(grid.Lines):
        if getattr(line, "is_nop_tie_line", False):
            continue
        if str(line.ID).startswith("line_for_"):
            continue
        records.append({
            "id": line.ID,
            "source": line.fBus,
            "target": line.tBus,
            "edge_type": "base_line",
        })
    return records


def _grid_nop_edge_records(grid) -> List[Dict[str, Any]]:
    records = []
    for nop in sorted(list(getattr(grid, "NOPs", {}).values()), key=lambda item: item.ID):
        records.append({
            "id": nop.ID,
            "source": nop.Bus1,
            "target": nop.Bus2,
            "line_id": "line_for_%s" % nop.ID,
        })
    return records


def generate_reconfiguration_plans(grid, gui_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    gui_params = gui_params or {}
    constraints = gui_params.get("reconfiguration_constraints") or {}
    root_bus = gui_params.get("slack_bus", "b1")
    return generate_reconfiguration_plans_from_records(
        bus_ids=list(grid.BusNames),
        base_edges=_grid_base_edge_records(grid),
        nop_edges=_grid_nop_edge_records(grid),
        root_bus=root_bus,
        constraints=constraints,
    )


def _rebuild_line_adjacency(grid) -> None:
    if not hasattr(grid, "_ladjfb") or not hasattr(grid, "_ladjtb"):
        return
    grid._ladjfb = {bus_id: [] for bus_id in grid.BusNames}
    grid._ladjtb = {bus_id: [] for bus_id in grid.BusNames}
    for line in grid.Lines:
        grid._ladjfb[line.fBus].append(line)
        grid._ladjtb[line.tBus].append(line)


def orient_active_lines_radially(grid, root_bus: str = "b1") -> None:
    root_bus = _normalize_bus_id(root_bus) or "b1"
    active_lines = list(grid.ActiveLines)
    adjacency: Dict[str, List[Tuple[str, Any]]] = {bus: [] for bus in grid.BusNames}
    for line in active_lines:
        adjacency.setdefault(line.fBus, []).append((line.tBus, line))
        adjacency.setdefault(line.tBus, []).append((line.fBus, line))

    visited = set()
    queue = deque([root_bus])
    while queue:
        bus = queue.popleft()
        if bus in visited:
            continue
        visited.add(bus)
        for nxt, line in adjacency.get(bus, []):
            if nxt in visited:
                continue
            line._fBus = bus
            line._tBus = nxt
            queue.append(nxt)

    if len(visited) != len(list(grid.BusNames)):
        missing = sorted(set(grid.BusNames) - visited)
        raise ValueError("Reconfiguration produced disconnected topology. Missing buses: %s" % missing[:10])
    _rebuild_line_adjacency(grid)


def _ensure_nop_lines(grid, selected_plan: Dict[str, Any]) -> None:
    from fpowerkit import Line

    selected_nop_id = selected_plan.get("close_nop_id")
    for nop in list(getattr(grid, "NOPs", {}).values()):
        line_id = "line_for_%s" % nop.ID
        if line_id in getattr(grid, "_lines", {}):
            line = grid.Line(line_id)
        else:
            line = Line(
                id=line_id,
                fbus=nop.Bus1,
                tbus=nop.Bus2,
                r_pu=nop.R,
                x_pu=nop.X,
                max_I_kA=nop.MaxI,
                active=False,
            )
            grid.AddLine(line)
        line.is_nop_tie_line = True
        line.nop_id = nop.ID
        line.active = nop.ID == selected_nop_id
        nop.active = line.active


def apply_reconfiguration_plan(grid, gui_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    gui_params = gui_params or {}
    use_nop = bool(gui_params.get("nop_nodes_active", gui_params.get("use_nop", True)))
    mode = str(gui_params.get("reconfiguration_mode", "radial_reconfiguration") or "none")
    selected_id = str(gui_params.get("selected_reconfiguration_plan_id", "R0") or "R0").strip()

    plans = generate_reconfiguration_plans(grid, gui_params)
    plan_by_id = {plan["plan_id"]: plan for plan in plans}

    if (not use_nop) or mode == "none" or selected_id.lower() in {"", "none"}:
        selected_id = "R0"
    if selected_id.lower() in {"auto", "best", "enumerate"}:
        selected_id = "R0"

    if selected_id not in plan_by_id:
        available = ", ".join(plan_by_id.keys())
        raise ValueError(
            "Unknown reconfiguration plan '%s'. Available plans: %s" % (selected_id, available)
        )

    selected_plan = plan_by_id[selected_id]
    active_line_ids = {edge["id"] for edge in selected_plan["active_edges"]}
    open_line_id = selected_plan.get("open_line_id")

    for line in list(grid.Lines):
        if getattr(line, "is_nop_tie_line", False) or str(line.ID).startswith("line_for_"):
            line.active = False
            continue
        line.active = line.ID in active_line_ids
        line.opened_by_reconfiguration = line.ID == open_line_id

    _ensure_nop_lines(grid, selected_plan)
    orient_active_lines_radially(grid, gui_params.get("slack_bus", "b1"))

    refreshed_plans = generate_reconfiguration_plans(grid, gui_params)
    refreshed_plan = next((plan for plan in refreshed_plans if plan["plan_id"] == selected_plan["plan_id"]), selected_plan)
    grid.available_reconfiguration_plans = refreshed_plans
    grid.reconfiguration_plan = refreshed_plan
    return refreshed_plan


def fixed_nop_status(grid, time_steps: int) -> Dict[str, List[int]]:
    plan = getattr(grid, "reconfiguration_plan", {}) or {}
    closed = plan.get("close_nop_id")
    result = {}
    for nop in list(getattr(grid, "NOPs", {}).values()):
        result[nop.ID] = [1 if nop.ID == closed else 0 for _ in range(time_steps)]
    return result

