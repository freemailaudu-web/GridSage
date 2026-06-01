from typing import Iterable, List, Optional

from .schema import ScenarioConfig
from .skill_schema import MatchedSkill, ScenarioSkill


def _format_parameters(skill: ScenarioSkill) -> List[str]:
    lines = []
    for name, rule in skill.recommended_parameters.items():
        parts = [f"- {name}"]
        if rule.default is not None:
            parts.append(f"default={rule.default}")
        if rule.min is not None or rule.max is not None:
            parts.append(f"range={rule.min}-{rule.max}")
        if rule.description:
            parts.append(rule.description)
        lines.append(": ".join([parts[0], ", ".join(parts[1:])]))
    return lines


def build_skill_context(
    matches: Iterable[MatchedSkill],
    current_state: Optional[ScenarioConfig] = None,
    max_chars: int = 6000,
) -> str:
    sections = []
    grid_model = getattr(current_state, "grid_model", "ieee33") or "ieee33"
    for match in matches:
        skill = match.skill
        if not skill:
            continue

        node_rules = skill.recommended_node_rules
        preferred_nodes = node_rules.preferred_nodes_by_grid.get(grid_model, [])
        node_actions = []
        for action, rule in node_rules.allowed_node_actions.items():
            item = f"- {action}: default={rule.default}, range={rule.min}-{rule.max}"
            node_actions.append(item)

        validation = [
            f"- [{rule.level}] {rule.rule_id}: {rule.condition} => {rule.message_cn}"
            for rule in skill.validation_rules
        ]

        section = "\n".join(
            [
                f"Matched Skill: {skill.skill_id} / {skill.name_cn} ({match.role}, score={match.score:.1f})",
                f"Scenario meaning: {skill.description_cn}",
                "Recommended parameters:",
                *(_format_parameters(skill) or ["- none"]),
                "Recommended nodes/actions:",
                f"- preferred {grid_model} nodes: {', '.join(preferred_nodes) if preferred_nodes else 'not configured'}",
                *(node_actions or ["- none"]),
                "Recommended RL:",
                f"- default algorithm: {skill.recommended_rl.default_algorithm}",
                f"- candidate algorithms: {', '.join(skill.recommended_rl.algorithms)}",
                f"- reward focus: {', '.join(skill.recommended_rl.reward_terms)}",
                f"Recommended metrics: {', '.join(skill.recommended_metrics)}",
                "Skill validation summary:",
                *(validation or ["- none"]),
            ]
        )
        sections.append(section)

    if not sections:
        return ""

    header = (
        "Scenario Skills Context:\n"
        "Use only the matched skills below. If the user does not give a value, use the skill default. "
        "Respect explicit user values unless they violate safety boundaries. "
        "If a value is outside the recommended range but inside the hard boundary, mention a warning. "
        "Do not invent ScenarioConfig fields.\n"
    )
    text = header + "\n\n".join(sections)
    return text[:max_chars]
