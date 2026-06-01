import ast
from typing import Any, Dict, Iterable, List, Tuple

from .schema import ScenarioConfig, ValidationMessage
from .skill_schema import MatchedSkill


ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
)


def _numeric_summary(scenario: ScenarioConfig) -> Dict[str, Any]:
    add_load_values = []
    add_pv_values = []
    add_wind_values = []
    add_ev_values = []
    add_ess_values = []

    end_nodes = {
        "ieee33": {"b18", "b25", "b33"},
        "ieee69": set(),
        "ieee123": set(),
    }.get(scenario.grid_model, set())
    end_node_add_load_kw = 0.0
    non_end_node_add_load_kw = 0.0
    end_node_add_ev_spots = 0.0

    for node_id, params in scenario.node_overrides.items():
        add_load_kw = float(params.get("add_load_kw", 0) or 0)
        add_ev_spots = float(params.get("add_ev_spots", 0) or 0)
        add_load_values.append(float(params.get("add_load_kw", 0) or 0))
        add_pv_values.append(float(params.get("add_pv_kw", 0) or 0))
        add_wind_values.append(float(params.get("add_wind_kw", 0) or 0))
        add_ev_values.append(float(params.get("add_ev_spots", 0) or 0))
        add_ess_values.append(float(params.get("add_ess_kwh", 0) or 0))
        if node_id in end_nodes:
            end_node_add_load_kw += add_load_kw
            end_node_add_ev_spots += add_ev_spots
        else:
            non_end_node_add_load_kw += add_load_kw

    return {
        "total_add_load_kw": sum(add_load_values),
        "max_add_load_kw": max(add_load_values or [0]),
        "total_add_pv_kw": sum(add_pv_values),
        "max_add_pv_kw": max(add_pv_values or [0]),
        "total_add_wind_kw": sum(add_wind_values),
        "max_add_wind_kw": max(add_wind_values or [0]),
        "total_add_ev_spots": sum(add_ev_values),
        "max_add_ev_spots": max(add_ev_values or [0]),
        "total_add_ess_kwh": sum(add_ess_values),
        "max_add_ess_kwh": max(add_ess_values or [0]),
        "has_known_end_nodes": bool(end_nodes),
        "end_node_add_load_kw": end_node_add_load_kw,
        "non_end_node_add_load_kw": non_end_node_add_load_kw,
        "end_node_add_ev_spots": end_node_add_ev_spots,
    }


def _profile_pairs(raw_values: Any) -> List[Tuple[int, float]]:
    if raw_values is None:
        return []
    if isinstance(raw_values, list):
        pairs = enumerate(raw_values)
    elif isinstance(raw_values, dict):
        pairs = raw_values.items()
    else:
        return []

    parsed = []
    for raw_hour, raw_value in pairs:
        try:
            hour = int(raw_hour)
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        parsed.append((hour, value))
    return parsed


def _max_profile_value(pairs: List[Tuple[int, float]], start_hour: int, end_hour: int) -> float:
    in_window = [value for hour, value in pairs if start_hour <= hour < end_hour]
    values = in_window or [value for _, value in pairs]
    return max(values or [0.0])


def _time_profile_summary(scenario: ScenarioConfig) -> Dict[str, Any]:
    start_hour = int(scenario.start_hour)
    end_hour = int(scenario.end_hour)
    profiles = scenario.time_profiles or {}

    pv_pairs = _profile_pairs(profiles.get("pv_multiplier_by_hour"))
    load_pairs = _profile_pairs(profiles.get("load_multiplier_by_hour"))
    ev_pairs = _profile_pairs(profiles.get("ev_multiplier_by_hour"))

    max_pv_profile = _max_profile_value(pv_pairs, start_hour, end_hour)
    max_load_profile = _max_profile_value(load_pairs, start_hour, end_hour)
    max_ev_profile = _max_profile_value(ev_pairs, start_hour, end_hour)
    midday_pv_profile = max([value for hour, value in pv_pairs if 10 <= hour <= 14] or [0.0])

    return {
        "max_pv_multiplier_by_hour": max_pv_profile,
        "max_load_multiplier_by_hour": max_load_profile,
        "max_ev_multiplier_by_hour": max_ev_profile,
        "max_midday_pv_multiplier_by_hour": midday_pv_profile,
        "effective_pv_multiplier": max(float(scenario.global_pv_multiplier), max_pv_profile),
        "effective_load_multiplier": max(float(scenario.global_load_multiplier), max_load_profile),
        "effective_ev_multiplier": max(float(scenario.global_ev_multiplier), max_ev_profile),
        "midday_window_overlap": start_hour < 15 and end_hour > 10,
    }


def _safe_eval(condition: str, context: Dict[str, Any]) -> bool:
    normalized = condition.replace("true", "True").replace("false", "False")
    tree = ast.parse(normalized, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            raise ValueError(f"Unsupported condition syntax: {condition}")
        if isinstance(node, ast.Name) and node.id not in context:
            raise ValueError(f"Unknown validation variable: {node.id}")
    return bool(eval(compile(tree, "<skill-rule>", "eval"), {"__builtins__": {}}, context))


def validate_with_skills(
    scenario: ScenarioConfig,
    matched_skills: Iterable[MatchedSkill],
) -> List[ValidationMessage]:
    context = scenario.model_dump() if hasattr(scenario, "model_dump") else scenario.dict()
    context.update(_numeric_summary(scenario))
    context.update(_time_profile_summary(scenario))

    messages: List[ValidationMessage] = []
    for match in matched_skills:
        skill = match.skill
        if not skill:
            continue
        for rule in skill.validation_rules:
            try:
                triggered = _safe_eval(rule.condition, context)
            except Exception as exc:
                messages.append(
                    ValidationMessage(
                        level="warning",
                        message=f"Skill rule {rule.rule_id} could not be evaluated: {exc}",
                        skill_id=skill.skill_id,
                        rule_id=rule.rule_id,
                    )
                )
                continue

            if triggered:
                messages.append(
                    ValidationMessage(
                        level=rule.level,
                        message=rule.message_cn,
                        skill_id=skill.skill_id,
                        rule_id=rule.rule_id,
                    )
                )

    return messages


def validation_status(messages: Iterable[ValidationMessage]) -> str:
    levels = {message.level for message in messages}
    if "reject" in levels:
        return "reject"
    if "warning" in levels:
        return "warning"
    return "pass"
