import re
from typing import Iterable, List, Tuple

from .schema import ScenarioConfig
from .skill_registry import skill_registry
from .skill_schema import MatchedSkill, ScenarioSkill


DEVICE_HINTS = {
    "S01_HIGH_PV_LOW_LOAD": ["光伏", "pv", "solar", "弃光", "过电压", "中午", "低负荷"],
    "S02_HEAVY_LOAD_END_NODES": ["负荷", "末端", "低电压", "过载", "heavy load", "end node"],
}

CONFLICTS = {
    ("S01_HIGH_PV_LOW_LOAD", "S02_HEAVY_LOAD_END_NODES"): "同时命中光伏大发低负荷与重负荷场景，请确认是低负荷高光伏，还是末端重负荷压力测试。"
}


def _norm(text: str) -> str:
    return (text or "").strip().lower()


def _contains(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    phrase_norm = _norm(phrase)
    if re.search(r"[a-zA-Z0-9]", phrase_norm):
        return phrase_norm in text
    return phrase in text


def _score_skill(user_input: str, state: ScenarioConfig, skill: ScenarioSkill) -> float:
    text = _norm(user_input)
    score = 0.0

    for trigger in skill.intent_triggers:
        if _contains(text, trigger):
            score += 3.0

    for sample in skill.typical_user_requests:
        if _contains(text, sample):
            score += 2.0

    for hint in DEVICE_HINTS.get(skill.skill_id, []):
        if _contains(text, hint):
            score += 1.0

    if skill.skill_id == "S01_HIGH_PV_LOW_LOAD":
        if state.global_pv_multiplier >= 1.3:
            score += 1.0
        if state.global_load_multiplier <= 0.9:
            score += 1.0
    elif skill.skill_id == "S02_HEAVY_LOAD_END_NODES":
        if state.global_load_multiplier >= 1.2:
            score += 1.0
        if any(params.get("add_load_kw", 0) > 0 for params in state.node_overrides.values()):
            score += 1.0

    return score


def detect_conflicts(matches: Iterable[MatchedSkill]) -> List[str]:
    ids = {match.skill_id for match in matches}
    warnings = []
    for pair, message in CONFLICTS.items():
        if pair[0] in ids and pair[1] in ids:
            warnings.append(message)
    return warnings


def _advisory_warnings(matches: Iterable[MatchedSkill], user_input: str) -> List[str]:
    text = user_input or ""
    warnings: List[str] = []
    for match in matches:
        skill = match.skill
        if not skill:
            continue
        defaults = []
        for field, rule in skill.recommended_parameters.items():
            if rule.default is None:
                continue
            if field == "global_pv_multiplier" and any(item in text for item in ["光伏", "PV", "pv", "solar"]):
                continue
            if field == "global_load_multiplier" and any(item in text for item in ["负荷", "load"]):
                continue
            if field == "global_ev_multiplier" and any(item in text for item in ["EV", "充电", "电动车"]):
                continue
            defaults.append(f"{field}={rule.default}")
        if defaults:
            warnings.append(f"{skill.skill_id} 将对未明确给出的参数采用推荐默认值：" + "，".join(defaults))
        if skill.recommended_metrics:
            warnings.append(f"{skill.skill_id} 建议重点关注指标：" + "，".join(skill.recommended_metrics[:4]))
    return warnings


def retrieve_skills(
    user_input: str,
    current_state: ScenarioConfig,
    min_score: float = 3.0,
    max_matches: int = 3,
) -> Tuple[List[MatchedSkill], List[str]]:
    scored = []
    for skill in skill_registry.all():
        score = _score_skill(user_input, current_state, skill)
        if score >= min_score:
            scored.append((score, skill))

    scored.sort(key=lambda item: item[0], reverse=True)
    matches: List[MatchedSkill] = []
    for index, (score, skill) in enumerate(scored[:max_matches]):
        role = "primary" if index == 0 else "secondary"
        matches.append(
            MatchedSkill(
                skill_id=skill.skill_id,
                name_cn=skill.name_cn,
                name_en=skill.name_en,
                category=skill.category,
                description_cn=skill.description_cn,
                score=score,
                role=role,
                skill=skill,
            )
        )

    warnings = detect_conflicts(matches)
    warnings.extend(_advisory_warnings(matches, user_input))
    return matches, warnings
