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
    voltage_pass = _number(metrics, ["电压合格率(%)", "电压合格率"])
    grid_loss = _number(metrics, ["精确总网损(kW)", "总网损", "网损"])
    cost = _number(metrics, ["总成本", "总目标值"])
    ev_rate = _number(metrics, ["EV充电满足率(%)", "EV SOC", "充电满足率"])
    renewable_curtailment = _number(metrics, ["弃光弃风量", "弃光弃风", "curtail"])
    renewable_absorption = _number(metrics, ["光伏风电消纳率", "消纳率"])
    max_voltage_deviation = _number(metrics, ["最大电压偏差", "电压偏差"])
    min_voltage = _number(metrics, ["最低节点电压", "最低电压"])
    line_flow = _number(metrics, ["最大线路功率流", "线路功率流", "line flow"])

    lines: List[str] = []
    if skill_id == "S01_HIGH_PV_LOW_LOAD":
        lines.append("### 光伏大发低负荷场景")
        if voltage_pass is not None:
            lines.append(f"- 电压合格率为 {voltage_pass:.2f}%。" + (
                " 当前高光伏低负荷场景下出现电压风险，说明光伏出力可能超过本地消纳能力，建议增加 ESS、EV 可调负荷或降低光伏倍率。"
                if voltage_pass < 100
                else " 当前策略暂未暴露明显电压越限，说明本轮调控对高光伏冲击有一定缓冲效果。"
            ))
        if max_voltage_deviation is not None:
            lines.append(f"- 最大电压偏差为 {max_voltage_deviation:.4f}，可用于判断高光伏反向潮流对节点电压的冲击程度。")
        if renewable_absorption is not None:
            lines.append(f"- 光伏风电消纳率为 {renewable_absorption:.2f}%，数值越高表示可再生能源本地消纳越充分。")
        if renewable_curtailment is not None:
            lines.append(f"- 弃光弃风量为 {renewable_curtailment:.4f} kWh。" + (
                " 当前策略未能充分吸收可再生能源，建议增加储能容量、延长中午调控窗口，或使用 SAC/TD3 继续训练。"
                if renewable_curtailment > 0
                else " 暂未观察到明显弃光弃风，说明可再生能源消纳压力可控。"
            ))
        if grid_loss is not None:
            lines.append(f"- 精确总网损为 {grid_loss:.4f} kW；若后续高于基线，通常意味着反向潮流或远距离输送带来了额外损耗。")
        if cost is not None:
            lines.append(f"- 总成本为 {cost:.4f}，可与 Baseline 或其他 RL 策略做横向比较，避免只追求电压而牺牲经济性。")
        if ev_rate is not None:
            lines.append(f"- EV 充电满足率为 {ev_rate:.2f}%。" + (
                " 策略可能过度服务电网侧目标，需要调整 reward 权重，避免牺牲用户充电需求。"
                if ev_rate < 95
                else " 用户侧充电需求基本得到保持。"
            ))
    elif skill_id == "S02_HEAVY_LOAD_END_NODES":
        lines.append("### 末端节点重负荷场景")
        if min_voltage is not None:
            lines.append(f"- 最低节点电压为 {min_voltage:.4f} pu，是判断末端电压支撑是否充足的关键指标。")
        if voltage_pass is not None:
            lines.append(f"- 电压合格率为 {voltage_pass:.2f}%。" + (
                " 当前末端重负荷场景下电压支撑不足，建议引入 ESS、SOP/NOP 或降低局部新增负荷。"
                if voltage_pass < 100
                else " 末端电压约束整体可控。"
            ))
        if line_flow is not None:
            lines.append(f"- 最大线路功率流为 {line_flow:.4f} pu。当前平台未使用线路电流/热容量约束，因此这里报告功率流压力而非电流负载率。")
        if grid_loss is not None:
            lines.append(f"- 精确总网损为 {grid_loss:.4f} kW。重负荷导致线路潮流增大时，网损通常会随之增加。")
        if cost is not None:
            lines.append(f"- 总成本为 {cost:.4f}，可用于判断电压支撑与经济调度之间的折中。")
        if ev_rate is not None:
            lines.append(f"- EV 充电满足率为 {ev_rate:.2f}%；若该值下降，说明 EV 集中充电进一步加剧了末端负荷压力。")
    return lines


def _comparison_lines(comparison: Dict[str, Dict[str, Any]], active_skill_ids: Iterable[str]) -> List[str]:
    lines = ["### 多模型对比解释"]
    best_cost = _best_label(comparison, ["总成本", "总目标值"])
    best_voltage = _best_label(comparison, ["电压合格率(%)", "电压合格率"], prefer_high=True)
    best_loss = _best_label(comparison, ["精确总网损(kW)", "总网损", "网损"])
    if best_cost:
        lines.append(f"- 总成本最低的是 **{best_cost[0]}**，成本为 {best_cost[1]:.4f}。")
    if best_voltage:
        lines.append(f"- 电压合格率最高的是 **{best_voltage[0]}**，合格率为 {best_voltage[1]:.2f}%。")
    if best_loss:
        lines.append(f"- 网损最低的是 **{best_loss[0]}**，网损为 {best_loss[1]:.4f} kW。")

    if best_cost and best_voltage and best_cost[0] != best_voltage[0]:
        lines.append("- 当前存在经济性与电压质量不完全一致的模型，论文实验中建议同时报告成本、电压合格率和网损，而不要只按单一指标排序。")
    elif best_cost:
        lines.append(f"- **{best_cost[0]}** 在主要指标上更值得作为下一轮训练或精调的起点。")

    ids = set(active_skill_ids or [])
    if "S02_HEAVY_LOAD_END_NODES" in ids and best_cost and best_cost[0].upper() != "BASELINE":
        lines.append("- 若 RL 策略优于 Baseline，说明训练策略在局部重负荷压力下更能协调可调资源。")
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
    lines = ["\n\n**场景相关解释**"]

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

    lines.append("### 下一步建议")
    if comparison_metrics:
        lines.append("- 保留同一 ScenarioConfig 和随机种子，继续扩大训练步数或补充更多场景压力测试。")
    else:
        lines.append("- 建议再运行 Baseline 与至少一个 RL 模型对比，确认当前策略的成本、电压和网损改进是否稳定。")

    return "\n\n" + "\n".join(lines)
