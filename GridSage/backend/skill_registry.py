import json
from pathlib import Path
from typing import Dict, List, Optional

from .skill_schema import ScenarioSkill


class SkillRegistry:
    """Loads and caches Scenario Skills from JSON files."""

    def __init__(self, skills_dir: Optional[Path] = None):
        self.skills_dir = skills_dir or Path(__file__).parent / "skills" / "scenario_skills"
        self._skills: Dict[str, ScenarioSkill] = {}
        self.refresh()

    def refresh(self) -> None:
        skills: Dict[str, ScenarioSkill] = {}
        if not self.skills_dir.exists():
            self._skills = {}
            return

        for path in sorted(self.skills_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            skill = ScenarioSkill(**raw)
            if skill.skill_id in skills:
                raise ValueError(f"Duplicate skill_id: {skill.skill_id}")
            skills[skill.skill_id] = skill

        self._skills = skills

    def get(self, skill_id: str) -> Optional[ScenarioSkill]:
        return self._skills.get(skill_id)

    def all(self) -> List[ScenarioSkill]:
        return list(self._skills.values())

    def summaries(self) -> List[dict]:
        return [
            {
                "skill_id": skill.skill_id,
                "version": skill.version,
                "name_cn": skill.name_cn,
                "name_en": skill.name_en,
                "category": skill.category,
                "description_cn": skill.description_cn,
                "recommended_metrics": skill.recommended_metrics,
                "default_algorithm": skill.recommended_rl.default_algorithm,
            }
            for skill in self.all()
        ]


skill_registry = SkillRegistry()
