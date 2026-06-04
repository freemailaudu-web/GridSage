from typing import Any, Dict, Iterable, List, Optional, Tuple

from .skill_registry import skill_registry


def _number(metrics: Dict[str, Any], candidates: Iterable[str]) -> Optional[float]:
    for key in candidates:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    for metric_key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        if any(candidate in metric_key for candidate in candidates):
            return float(value)
    return None


def _best_label(comparison: Dict[str, Dict[str, Any]], keys: Iterable[str], prefer_high: bool = False) -> Optional[Tuple[str, float]]:
    values = []
    for label, metrics in comparison.items():
        value = _number(metrics, keys)
        if value is not None:
            values.append((label, value))
    if not values:
        return None
    return max(values, key=lambda item: item[1]) if prefer_high else min(values, key=lambda item: item[1])


def _risk_line(metric: Optional[float], threshold: float, low_is_bad: bool, bad: str, good: str) -> str:
    if metric is None:
        return ""
    is_bad = metric < threshold if low_is_bad else metric > threshold
    return bad if is_bad else good


def _single_model_lines(skill_id: str, metrics: Dict[str, Any], state: Any = None) -> List[str]:
    voltage_pass = _number(metrics, ["Voltage Compliance Rate(%)", "Voltage Compliance Rate"])
    grid_loss = _number(metrics, ["Exact Total Grid Loss (kW)", "Total Grid Loss", "Grid Loss"])
    cost = _number(metrics, ["Total Cost", "Total Objective Value"])
    ev_rate = _number(metrics, ["EV Charging Satisfaction Rate (%)", "EV SOC", "Charging Satisfaction Rate"])
    renewable_curtailment = _number(metrics, ["Renewable Energy Curtailment", "abandon light and abandon wind", "curtail"])
    renewable_absorption = _number(metrics, ["Renewable Energy Absorption Rate", "absorption rate"])
    max_voltage_deviation = _number(metrics, ["Maximum Voltage Deviation", "Voltage Deviation"])
    min_voltage = _number(metrics, ["Minimum Node Voltage", "Minimum Voltage"])
    line_flow = _number(metrics, ["Maximum Line Power Flow", "Line Power Flow", "line flow"])

    lines: List[str] = []
    if skill_id == "S01_HIGH_PV_LOW_LOAD":
        lines.append("### High-PV Low-Load Scenario")
        if voltage_pass is not None:
            lines.append(f"- Voltage Compliance Rate is {voltage_pass:.2f}%." + (
                "The voltage risk in the current high photovoltaic and low load scenario indicates that the photovoltaic output may exceed the local consumption capacity. It is recommended to increase the ESS and EV adjustable load or reduce the photovoltaic rate."
                if voltage_pass < 100
                else "The current strategy has not yet exposed any obvious voltage overruns, indicating that this round of regulation has a certain buffering effect on high photovoltaic impacts."
            ))
        if max_voltage_deviation is not None:
            lines.append(f"- Maximum Voltage Deviation is {max_voltage_deviation:.4f}, which can be used to determine the impact of high photovoltaic reverse power flow on node voltage.")
        if renewable_absorption is not None:
            lines.append(f"- Renewable Energy Absorption Rate is {renewable_absorption:.2f}%. The higher the value, the more fully the local consumption of renewable energy.")
        if renewable_curtailment is not None:
            lines.append(f"- Renewable Energy Curtailment is {renewable_curtailment:.4f} kWh." + (
                "The current strategy fails to fully absorb renewable energy. It is recommended to increase energy storage capacity, extend the noon control window, or use SAC/TD3 to continue training."
                if renewable_curtailment > 0
                else " No obvious abandonment of light or wind has been observed yet, indicating that the pressure of renewable energy consumption is controllable."
            ))
        if grid_loss is not None:
            lines.append(f"- Exact Total Grid Loss is {grid_loss:.4f} kW; if it is subsequently higher than the baseline, it usually means additional losses caused by reverse power flow or long-distance power transfer.")
        if cost is not None:
            lines.append(f"- Total Cost is {cost:.4f}, which can be compared horizontally with Baseline or other RL strategies to avoid sacrificing economy at the cost of pursuing voltage only.")
        if ev_rate is not None:
            lines.append(f"- EV Charging Satisfaction Rate is {ev_rate:.2f}%." + (
                "The strategy may over-serve the grid-side goals, and the reward weight needs to be adjusted to avoid sacrificing user charging needs."
                if ev_rate < 95
                else "User-side charging needs are basically maintained."
            ))
    elif skill_id == "S02_HEAVY_LOAD_END_NODES":
        lines.append("### End node heavy load scenario")
        if min_voltage is not None:
            lines.append(f"- Minimum Node Voltage is {min_voltage:.4f} pu, which is a key indicator to judge whether the terminal voltage support is sufficient.")
        if voltage_pass is not None:
            lines.append(f"- Voltage Compliance Rate is {voltage_pass:.2f}%." + (
                "The voltage support is insufficient under the current terminal heavy load scenario. It is recommended to introduce ESS, SOP/NOP or reduce local new loads."
                if voltage_pass < 100
                else "The terminal voltage constraint is overall controllable."
            ))
        if line_flow is not None:
            lines.append(f"- Maximum Line Power Flow is {line_flow:.4f} pu. The current platform does not use line current/thermal capacity constraints, so power flow pressure is reported here instead of current load rate.")
        if grid_loss is not None:
            lines.append(f"- Exact Total Grid Loss is {grid_loss:.4f} kW. When heavy load causes the line power flow to increase, Grid Loss usually increases accordingly.")
        if cost is not None:
            lines.append(f"- Total Cost is {cost:.4f}, which can be used to judge the compromise between voltage support and economic dispatch.")
        if ev_rate is not None:
            lines.append(f"- EV Charging Satisfaction Rate is {ev_rate:.2f}%; if the value decreases, it means that EV centralized charging further aggravates the end load pressure.")
    return lines


def _comparison_lines(comparison: Dict[str, Dict[str, Any]], active_skill_ids: Iterable[str]) -> List[str]:
    lines = ["### Multi-model comparison explanation"]
    best_cost = _best_label(comparison, ["Total Cost", "Total Objective Value"])
    best_voltage = _best_label(comparison, ["Voltage Compliance Rate(%)", "Voltage Compliance Rate"], prefer_high=True)
    best_loss = _best_label(comparison, ["Exact Total Grid Loss (kW)", "Total Grid Loss", "Grid Loss"])
    if best_cost:
        lines.append(f"- The lowest Total Cost is **{best_cost[0]}**, and the cost is {best_cost[1]:.4f}.")
    if best_voltage:
        lines.append(f"- The highest Voltage Compliance Rate is **{best_voltage[0]}**, and the pass rate is {best_voltage[1]:.2f}%.")
    if best_loss:
        lines.append(f"- The lowest Grid Loss is **{best_loss[0]}**, and the Grid Loss is {best_loss[1]:.4f} kW.")

    if best_cost and best_voltage and best_cost[0] != best_voltage[0]:
        lines.append("- Currently there are models that are not completely consistent between economic efficiency and voltage quality. In the paper's experiments, it is recommended to report cost, Voltage Compliance Rate and Grid Loss at the same time instead of just sorting by a single indicator.")
    elif best_cost:
        lines.append(f"- **{best_cost[0]}** is more worthy of being used as the starting point for the next round of training or fine-tuning on the main indicators.")

    ids = set(active_skill_ids or [])
    if "S02_HEAVY_LOAD_END_NODES" in ids and best_cost and best_cost[0].upper() != "BASELINE":
        lines.append("- If the RL strategy is better than Baseline, it means that the training strategy is better able to coordinate adjustable resources under local heavy load pressure.")
    return lines


def interpret_results_with_skills(
    active_skill_ids: Iterable[str],
    metrics: Dict[str, Any],
    state: Any = None,
    comparison_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
    task_status: str = "",
) -> str:
    if not active_skill_ids:
        return ""

    metrics = metrics or {}
    comparison_metrics = comparison_metrics or metrics.get("comparison_metrics") or {}
    lines = ["\n\n**Scene related explanation**"]

    if comparison_metrics:
        lines.extend(_comparison_lines(comparison_metrics, active_skill_ids))

    for skill_id in active_skill_ids or []:
        skill = skill_registry.get(skill_id)
        if not skill:
            continue
        skill_lines = _single_model_lines(skill_id, metrics, state)
        if skill_lines:
            lines.extend(skill_lines)
        elif skill.result_interpretation_focus:
            lines.append(f"### {skill.name_cn}")
            lines.extend(f"- {item}" for item in skill.result_interpretation_focus[:3])

    lines.append("### Next step suggestions")
    if comparison_metrics:
        lines.append("- Keep the same ScenarioConfig and random seed, continue to expand the Training Steps or add more scenario stress tests.")
    else:
        lines.append("- It is recommended to run Baseline again to compare with at least one RL model to confirm whether the cost, voltage and Grid Loss improvements of the current strategy are stable.")

    return "\n\n" + "\n".join(lines)
