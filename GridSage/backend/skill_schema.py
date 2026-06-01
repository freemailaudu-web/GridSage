from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SkillParameterRule(BaseModel):
    default: Optional[Any] = None
    min: Optional[float] = None
    max: Optional[float] = None
    description: str = ""


class SkillNodeRule(BaseModel):
    preferred_node_types: List[str] = Field(default_factory=list)
    preferred_nodes_by_grid: Dict[str, List[str]] = Field(default_factory=dict)
    allowed_node_actions: Dict[str, SkillParameterRule] = Field(default_factory=dict)


class SkillRewardRecommendation(BaseModel):
    algorithms: List[str] = Field(default_factory=list)
    default_algorithm: str = "SAC"
    suggested_total_timesteps: Dict[str, int] = Field(default_factory=dict)
    reward_terms: List[str] = Field(default_factory=list)


class SkillValidationRule(BaseModel):
    rule_id: str
    level: Literal["warning", "reject"] = "warning"
    condition: str
    message_cn: str
    message_en: str = ""


class SkillExample(BaseModel):
    user_request: str
    expected_delta_commands: List[Dict[str, Any]] = Field(default_factory=list)


class ScenarioSkill(BaseModel):
    skill_id: str
    version: str = "1.0.0"
    name_cn: str
    name_en: str = ""
    category: str
    description_cn: str
    description_en: str = ""
    intent_triggers: List[str] = Field(default_factory=list)
    typical_user_requests: List[str] = Field(default_factory=list)
    recommended_parameters: Dict[str, SkillParameterRule] = Field(default_factory=dict)
    recommended_node_rules: SkillNodeRule = Field(default_factory=SkillNodeRule)
    recommended_rl: SkillRewardRecommendation = Field(default_factory=SkillRewardRecommendation)
    recommended_metrics: List[str] = Field(default_factory=list)
    validation_rules: List[SkillValidationRule] = Field(default_factory=list)
    result_interpretation_focus: List[str] = Field(default_factory=list)
    examples: List[SkillExample] = Field(default_factory=list)


class MatchedSkill(BaseModel):
    skill_id: str
    name_cn: str
    name_en: str = ""
    category: str = ""
    description_cn: str = ""
    score: float
    role: Literal["primary", "secondary"] = "secondary"
    skill: Optional[ScenarioSkill] = None
    warnings: List[str] = Field(default_factory=list)

    def public_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name_cn": self.name_cn,
            "name_en": self.name_en,
            "category": self.category,
            "description_cn": self.description_cn,
            "score": round(self.score, 3),
            "role": self.role,
        }
