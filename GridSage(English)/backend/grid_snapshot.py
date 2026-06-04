"""
Grid Snapshot Module
====================
Builds a read-only, presentation-oriented snapshot of the current grid
topology, per-node base devices, and user modifications for the frontend
right-side panel.

This module does NOT run any simulation or power-flow — it only reads
the Excel parameter file and the current ScenarioConfig to assemble
display data.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
#  Pydantic models for the snapshot response
# ---------------------------------------------------------------------------

class DeviceInfo(BaseModel):
    device_type: str
    device_name: str
    description: str = ""


class ModificationInfo(BaseModel):
    device_name: str
    change_type: str
    change_value: str


class NodeSnapshot(BaseModel):
    id: str
    label: str
    base_devices: List[DeviceInfo] = Field(default_factory=list)
    modifications: List[ModificationInfo] = Field(default_factory=list)


class EdgeSnapshot(BaseModel):
    id: str
    source: str  # "from" node
    target: str  # "to" node
    edge_type: str = "line"
    name: str = ""
    is_active: bool = True
    is_opened: bool = False


class NodeModificationRow(BaseModel):
    node: str
    device_name: str
    change_type: str
    change_value: str
    source: str = "User natural-language modification"


class DeviceSummary(BaseModel):
    bus_count: int = 0
    line_count: int = 0
    load_bus_count: int = 0
    pv_count: int = 0
    wind_count: int = 0
    ess_count: int = 0
    ev_station_count: int = 0
    sop_count: int = 0
    nop_count: int = 0
    generator_count: int = 0


class GlobalControls(BaseModel):
    pv_multiplier: float = 1.0
    load_multiplier: float = 1.0
    ev_multiplier: float = 1.0
    use_pv: bool = True
    use_wind: bool = True
    use_ess: bool = True
    use_sop: bool = True
    use_nop: bool = True
    reconfiguration_mode: str = "radial_reconfiguration"
    selected_reconfiguration_plan_id: str = "R0"
    reconfiguration_constraints: Dict[str, Any] = Field(default_factory=dict)
    start_hour: int = 0
    end_hour: int = 24
    step_minutes: int = 60
    time_profiles: Dict[str, Any] = Field(default_factory=dict)


class GridSnapshot(BaseModel):
    grid_model: str = "ieee33"
    global_controls: GlobalControls = Field(default_factory=GlobalControls)
    nodes: List[NodeSnapshot] = Field(default_factory=list)
    edges: List[EdgeSnapshot] = Field(default_factory=list)
    device_summary: DeviceSummary = Field(default_factory=DeviceSummary)
    node_modification_rows: List[NodeModificationRow] = Field(default_factory=list)
    available_reconfiguration_plans: List[Dict[str, Any]] = Field(default_factory=list)
    selected_reconfiguration_plan: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
#  IEEE topology definitions (edges as (from, to) bus-number pairs)
# ---------------------------------------------------------------------------

_IEEE33_EDGES = [
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
    (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
    (16, 17), (17, 18),
    (2, 19), (19, 20), (20, 21), (21, 22),
    (3, 23), (23, 24), (24, 25),
    (6, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32),
    (32, 33),
]

_IEEE69_EDGES = [
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
    (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
    (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23),
    (23, 24), (24, 25), (25, 26), (26, 27),
    (3, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34),
    (34, 35),
    (4, 36), (36, 37), (37, 38), (38, 39), (39, 40),
    (8, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47),
    (47, 48),
    (9, 49), (49, 50),
    (11, 51), (51, 52), (52, 53),
    (12, 54), (54, 55), (55, 56),
    (27, 57), (57, 58), (58, 59), (59, 60), (60, 61), (61, 62), (62, 63),
    (63, 64), (64, 65), (65, 66), (66, 67), (67, 68), (68, 69),
]

def _get_edges(model: str):
    model = model.lower()
    if model == "ieee33":
        return _IEEE33_EDGES, 33
    elif model == "ieee69":
        return _IEEE69_EDGES, 69
    elif model == "ieee123":
        # For IEEE123, generate a simple linear chain as a fallback
        return [(i, i + 1) for i in range(1, 123)], 123
    else:
        return _IEEE33_EDGES, 33


# ---------------------------------------------------------------------------
#  Excel reading helpers
# ---------------------------------------------------------------------------

def _resolve_excel_path() -> str:
    """Resolve the grid_parameters.xlsx path."""
    # Compatibility note: vgridsim_core remains the runtime directory name to avoid breaking paths.
    core_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vgridsim_core")
    path = os.path.join(core_dir, "data", "grid_parameters.xlsx")
    # Also try env variable
    path = os.environ.get("GRID_PARAMS_XLSX", path)
    return path


def _read_sheet_safe(excel_path: str, sheet_name: str) -> Optional[pd.DataFrame]:
    """Read an Excel sheet; return None if it doesn't exist."""
    try:
        return pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    except Exception:
        return None


def _bus_has_load(df_loads: Optional[pd.DataFrame], bus_id: str) -> bool:
    """Check whether a bus has load data defined in the Excel."""
    if df_loads is None:
        return False
    return bus_id in {_normalize_bus_id(value) for value in df_loads["BusID"].values}


def _get_load_stats(df_loads: Optional[pd.DataFrame], bus_id: str) -> Optional[Dict]:
    """Get peak and average Pd for a bus, if available."""
    if df_loads is None:
        return None
    row = df_loads[df_loads["BusID"].apply(_normalize_bus_id) == bus_id]
    if row.empty:
        return None
    row = row.iloc[0]
    pd_cols = [f"Pd_t{t}" for t in range(24)]
    existing = [col for col in pd_cols if col in row.index and pd.notna(row[col])]
    if not existing:
        return None
    values = [float(row[col]) for col in existing]
    return {"peak_kw": round(max(values), 2), "avg_kw": round(sum(values) / len(values), 2)}


def _normalize_bus_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    if text.lower().startswith("b"):
        suffix = text[1:]
        try:
            return f"b{int(float(suffix))}"
        except (TypeError, ValueError):
            return "b" + suffix
    try:
        return f"b{int(float(text))}"
    except (TypeError, ValueError):
        return text


def _disabled_map(config) -> Dict[str, Dict[str, List[str]]]:
    result: Dict[str, Dict[str, List[str]]] = {}
    for node, devices in (getattr(config, "disabled_devices", {}) or {}).items():
        node_id = _normalize_bus_id(node)
        if not node_id or not isinstance(devices, dict):
            continue
        result.setdefault(node_id, {})
        for device_type, ids in devices.items():
            if isinstance(ids, list):
                result[node_id][str(device_type)] = [str(item) for item in ids]
            elif ids:
                result[node_id][str(device_type)] = [str(ids)]
    return result


def _is_disabled(disabled: Dict[str, Dict[str, List[str]]], bus_id: str, device_type: str, device_id: str) -> bool:
    ids = (disabled.get(bus_id, {}) or {}).get(device_type, [])
    return "*" in ids or str(device_id) in ids


# ---------------------------------------------------------------------------
#  Main snapshot builder
# ---------------------------------------------------------------------------

def build_grid_snapshot(scenario_config) -> GridSnapshot:
    """
    Build a GridSnapshot from the current ScenarioConfig.

    Parameters
    ----------
    scenario_config : ScenarioConfig
        The current session state (from backend.schema).

    Returns
    -------
    GridSnapshot
    """
    model = scenario_config.grid_model or "ieee33"
    excel_path = _resolve_excel_path()

    # ---- Global Controls ----
    global_controls = GlobalControls(
        pv_multiplier=scenario_config.global_pv_multiplier,
        load_multiplier=scenario_config.global_load_multiplier,
        ev_multiplier=scenario_config.global_ev_multiplier,
        use_pv=scenario_config.use_pv,
        use_wind=scenario_config.use_wind,
        use_ess=scenario_config.use_ess,
        use_sop=scenario_config.use_sop,
        use_nop=scenario_config.use_nop,
        reconfiguration_mode=scenario_config.reconfiguration_mode,
        selected_reconfiguration_plan_id=scenario_config.selected_reconfiguration_plan_id,
        reconfiguration_constraints=scenario_config.reconfiguration_constraints or {},
        start_hour=scenario_config.start_hour,
        end_hour=scenario_config.end_hour,
        step_minutes=scenario_config.step_minutes,
        time_profiles=scenario_config.time_profiles or {},
    )

    # ---- Topology edges ----
    edge_list, num_buses = _get_edges(model)
    edges = []
    for idx, (f, t) in enumerate(edge_list, 1):
        edges.append(EdgeSnapshot(
            id=f"line_{idx}",
            source=f"b{f}",
            target=f"b{t}",
            edge_type="line",
            name=f"Line {f}-{t}",
        ))

    # ---- Read Excel sheets ----
    bus_loads_sheet = f"BusLoads_{model}"
    df_loads = _read_sheet_safe(excel_path, bus_loads_sheet)
    df_pv_wind = _read_sheet_safe(excel_path, "PVWind")
    df_ess = _read_sheet_safe(excel_path, "ESS")
    df_ev = _read_sheet_safe(excel_path, "EVStation")
    df_sop = _read_sheet_safe(excel_path, "SOP")
    df_nop = _read_sheet_safe(excel_path, "NOP")
    df_gen = _read_sheet_safe(excel_path, "Generators")
    disabled = _disabled_map(scenario_config)

    reconfiguration_plans: List[Dict[str, Any]] = []
    selected_reconfiguration_plan: Dict[str, Any] = {}
    try:
        import sys

        # Compatibility note: vgridsim_core remains the runtime directory name to avoid breaking imports.
        core_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vgridsim_core")
        if core_dir not in sys.path:
            sys.path.insert(0, core_dir)
        from reconfiguration import generate_reconfiguration_plans_from_records

        base_edge_records = [
            {
                "id": f"line_{idx}",
                "source": f"b{f}",
                "target": f"b{t}",
                "edge_type": "base_line",
            }
            for idx, (f, t) in enumerate(edge_list, 1)
        ]
        nop_edge_records = []
        if df_nop is not None:
            for _, row in df_nop.iterrows():
                nop_id = str(row["ID"])
                bus1 = _normalize_bus_id(row["Bus1"])
                bus2 = _normalize_bus_id(row["Bus2"])
                if _is_disabled(disabled, bus1, "nop", nop_id) or _is_disabled(disabled, bus2, "nop", nop_id):
                    continue
                nop_edge_records.append({
                    "id": nop_id,
                    "source": bus1,
                    "target": bus2,
                    "line_id": f"line_for_{nop_id}",
                })
        reconfiguration_plans = generate_reconfiguration_plans_from_records(
            bus_ids=[f"b{i}" for i in range(1, num_buses + 1)],
            base_edges=base_edge_records,
            nop_edges=nop_edge_records,
            root_bus="b1",
            constraints=scenario_config.reconfiguration_constraints or {},
        )
        selected_id = str(scenario_config.selected_reconfiguration_plan_id or "R0")
        if (not scenario_config.use_nop) or scenario_config.reconfiguration_mode == "none":
            selected_id = "R0"
        if selected_id.lower() in {"auto", "best", "enumerate"}:
            selected_id = "R0"
        selected_reconfiguration_plan = next(
            (plan for plan in reconfiguration_plans if plan["plan_id"] == selected_id),
            reconfiguration_plans[0] if reconfiguration_plans else {},
        )
    except Exception:
        reconfiguration_plans = []
        selected_reconfiguration_plan = {}

    active_edge_ids = {
        edge["id"]
        for edge in selected_reconfiguration_plan.get("active_edges", [])
    }
    opened_line_id = selected_reconfiguration_plan.get("open_line_id")
    if active_edge_ids:
        for edge in edges:
            edge.is_active = edge.id in active_edge_ids
            edge.is_opened = edge.id == opened_line_id
    if selected_reconfiguration_plan.get("close_nop_id"):
        edges.append(EdgeSnapshot(
            id=f"line_for_{selected_reconfiguration_plan['close_nop_id']}",
            source=selected_reconfiguration_plan["close_nop_bus1"],
            target=selected_reconfiguration_plan["close_nop_bus2"],
            edge_type="nop_line",
            name=f"Closed NOP {selected_reconfiguration_plan['close_nop_id']}",
            is_active=True,
            is_opened=False,
        ))

    # Build per-bus device index
    bus_pvs: Dict[str, List[Dict]] = {}
    bus_winds: Dict[str, List[Dict]] = {}
    bus_ess: Dict[str, List[Dict]] = {}
    bus_ev: Dict[str, List[Dict]] = {}
    bus_sop: Dict[str, List[Dict]] = {}
    bus_nop: Dict[str, List[Dict]] = {}
    bus_gen: Dict[str, List[Dict]] = {}

    pv_count = 0
    wind_count = 0
    ess_count = 0
    ev_count = 0
    sop_count = 0
    nop_count = 0
    gen_count = 0

    if df_pv_wind is not None:
        for _, row in df_pv_wind.iterrows():
            bus_id = _normalize_bus_id(row["BusID"])
            pvw_id = str(row["ID"])
            pvw_type = str(row["Type"]).lower()
            if _is_disabled(disabled, bus_id, pvw_type, pvw_id):
                continue
            if pvw_type == "pv":
                bus_pvs.setdefault(bus_id, []).append({"id": pvw_id})
                pv_count += 1
            elif pvw_type == "wind":
                bus_winds.setdefault(bus_id, []).append({"id": pvw_id})
                wind_count += 1

    if df_ess is not None:
        for _, row in df_ess.iterrows():
            bus_id = _normalize_bus_id(row["BusID"])
            ess_id = str(row["ID"])
            if _is_disabled(disabled, bus_id, "ess", ess_id):
                continue
            cap = row.get("Cap_puh", "")
            bus_ess.setdefault(bus_id, []).append({"id": ess_id, "cap": cap})
            ess_count += 1

    if df_ev is not None:
        for _, row in df_ev.iterrows():
            bus_id = _normalize_bus_id(row.get("Bus_ID", row.get("BusID", "")))
            st_id = str(row.get("Station_ID", row.get("StationID", row.get("ID", ""))))
            if _is_disabled(disabled, bus_id, "ev_station", st_id):
                continue
            spots = row.get("Num_Spots", row.get("NumSpots", row.get("num_chargers", "")))
            if bus_id:
                bus_ev.setdefault(bus_id, []).append({"id": st_id, "spots": spots})
                ev_count += 1

    if df_sop is not None:
        for _, row in df_sop.iterrows():
            sop_id = str(row["ID"])
            bus1 = _normalize_bus_id(row["Bus1"])
            bus2 = _normalize_bus_id(row["Bus2"])
            if _is_disabled(disabled, bus1, "sop", sop_id) or _is_disabled(disabled, bus2, "sop", sop_id):
                continue
            info = {"id": sop_id, "bus1": bus1, "bus2": bus2}
            bus_sop.setdefault(bus1, []).append(info)
            bus_sop.setdefault(bus2, []).append(info)
            sop_count += 1

    if df_nop is not None:
        for _, row in df_nop.iterrows():
            nop_id = str(row["ID"])
            bus1 = _normalize_bus_id(row["Bus1"])
            bus2 = _normalize_bus_id(row["Bus2"])
            if _is_disabled(disabled, bus1, "nop", nop_id) or _is_disabled(disabled, bus2, "nop", nop_id):
                continue
            info = {"id": nop_id, "bus1": bus1, "bus2": bus2}
            bus_nop.setdefault(bus1, []).append(info)
            bus_nop.setdefault(bus2, []).append(info)
            nop_count += 1

    if df_gen is not None:
        for _, row in df_gen.iterrows():
            bus_id = _normalize_bus_id(row["BusID"])
            gen_id = str(row["ID"])
            if _is_disabled(disabled, bus_id, "generator", gen_id):
                continue
            bus_gen.setdefault(bus_id, []).append({"id": gen_id})
            gen_count += 1

    # Count buses with loads
    load_bus_count = 0
    if df_loads is not None:
        load_bus_count = len(df_loads)

    # ---- Build node list ----
    node_overrides = {
        _normalize_bus_id(node): params
        for node, params in (scenario_config.node_overrides or {}).items()
    }
    nodes: List[NodeSnapshot] = []

    for i in range(1, num_buses + 1):
        bus_id = f"b{i}"
        base_devices: List[DeviceInfo] = []

        # Bus itself
        base_devices.append(DeviceInfo(device_type="bus", device_name=f"Node {bus_id}", description=""))

        # Load
        if _bus_has_load(df_loads, bus_id):
            stats = _get_load_stats(df_loads, bus_id)
            desc = f"From {bus_loads_sheet}"
            if stats:
                desc += f", peak {stats['peak_kw']} kW, average {stats['avg_kw']} kW"
            base_devices.append(DeviceInfo(device_type="load", device_name="Base Load", description=desc))

        # Generator
        for g in bus_gen.get(bus_id, []):
            base_devices.append(DeviceInfo(device_type="generator", device_name=g["id"], description="Generator"))

        # PV
        for pv in bus_pvs.get(bus_id, []):
            base_devices.append(DeviceInfo(device_type="pv", device_name=pv["id"], description="PV Device"))

        # Wind
        for w in bus_winds.get(bus_id, []):
            base_devices.append(DeviceInfo(device_type="wind", device_name=w["id"], description="Wind Device"))

        # ESS
        for e in bus_ess.get(bus_id, []):
            cap_str = f", capacity {e['cap']} p.u.h" if e.get("cap") else ""
            base_devices.append(DeviceInfo(device_type="ess", device_name=e["id"], description=f"Energy Storage Device{cap_str}"))

        # EV station
        for ev in bus_ev.get(bus_id, []):
            spots_str = f", {ev['spots']} charging spots" if ev.get("spots") else ""
            base_devices.append(DeviceInfo(device_type="ev_station", device_name=ev["id"], description=f"EV Charging Station{spots_str}"))

        # SOP
        seen_sop = set()
        for s in bus_sop.get(bus_id, []):
            if s["id"] not in seen_sop:
                seen_sop.add(s["id"])
                other = s["bus2"] if s["bus1"] == bus_id else s["bus1"]
                base_devices.append(DeviceInfo(
                    device_type="sop", device_name=s["id"],
                    description=f"SOP endpoint, connects {s['bus1']} - {s['bus2']}",
                ))

        # NOP
        seen_nop = set()
        for n in bus_nop.get(bus_id, []):
            if n["id"] not in seen_nop:
                seen_nop.add(n["id"])
                base_devices.append(DeviceInfo(
                    device_type="nop", device_name=n["id"],
                    description=f"NOP endpoint, connects {n['bus1']} - {n['bus2']}",
                ))

        # User modifications
        modifications: List[ModificationInfo] = []
        overrides = node_overrides.get(bus_id, {})
        if overrides.get("add_load_kw"):
            modifications.append(ModificationInfo(
                device_name="Base Load", change_type="Add Load",
                change_value=f"+{overrides['add_load_kw']} kW",
            ))
        if overrides.get("add_pv_kw"):
            modifications.append(ModificationInfo(
                device_name="PV", change_type="Add PV",
                change_value=f"+{overrides['add_pv_kw']} kW",
            ))
        if overrides.get("add_ess_kwh"):
            ess_change_value = f"+{overrides['add_ess_kwh']} kWh"
            if overrides.get("add_ess_power_kw"):
                ess_change_value += f", {overrides['add_ess_power_kw']} kW"
            elif overrides.get("add_ess_c_rate"):
                ess_change_value += f", {overrides['add_ess_c_rate']}C"
            modifications.append(ModificationInfo(
                device_name="Energy Storage", change_type="Expand Storage",
                change_value=ess_change_value,
            ))
        if overrides.get("add_ev_spots"):
            modifications.append(ModificationInfo(
                device_name="EV Charging Station", change_type="Add Charging Spots",
                change_value=f"+{int(overrides['add_ev_spots'])}",
            ))
        if overrides.get("add_wind_kw"):
            modifications.append(ModificationInfo(
                device_name="Wind", change_type="Add Wind",
                change_value=f"+{overrides['add_wind_kw']} kW",
            ))
        for device_type, ids in (disabled.get(bus_id, {}) or {}).items():
            for device_id in ids:
                name = "Default device of this type" if device_id == "*" else device_id
                modifications.append(ModificationInfo(
                    device_name=device_type,
                    change_type="Disable Default Device",
                    change_value=name,
                ))

        nodes.append(NodeSnapshot(
            id=bus_id,
            label=bus_id,
            base_devices=base_devices,
            modifications=modifications,
        ))

    # ---- Node modification rows (flat table for the panel) ----
    mod_rows: List[NodeModificationRow] = []
    for node in nodes:
        for mod in node.modifications:
            mod_rows.append(NodeModificationRow(
                node=node.id,
                device_name=mod.device_name,
                change_type=mod.change_type,
                change_value=mod.change_value,
            ))

    # ---- Device summary ----
    summary = DeviceSummary(
        bus_count=num_buses,
        line_count=int(selected_reconfiguration_plan.get("edge_count", len(edge_list)) or len(edge_list)),
        load_bus_count=load_bus_count,
        pv_count=pv_count,
        wind_count=wind_count,
        ess_count=ess_count,
        ev_station_count=ev_count,
        sop_count=sop_count,
        nop_count=nop_count,
        generator_count=gen_count,
    )

    return GridSnapshot(
        grid_model=model,
        global_controls=global_controls,
        nodes=nodes,
        edges=edges,
        device_summary=summary,
        node_modification_rows=mod_rows,
        available_reconfiguration_plans=reconfiguration_plans,
        selected_reconfiguration_plan=selected_reconfiguration_plan,
    )
