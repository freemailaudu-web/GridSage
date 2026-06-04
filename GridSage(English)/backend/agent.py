import ast
import json
import re
from typing import List, Optional

from openai import AsyncOpenAI

from .schema import ChatResponse, DeltaCommand, ScenarioConfig
from .skill_schema import MatchedSkill


SUPPORTED_ALGOS = ("SAC", "PPO", "TD3", "DDPG")

# Compatibility note: generate_lvgs_actions is a legacy tool contract and remains
# unchanged so existing function-calling integrations continue to work.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_lvgs_actions",
            "description": "Generate GridSage Delta Commands for scenario configuration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thoughts": {
                        "type": "string",
                        "description": "User-facing brief explanation. Do not output hidden reasoning.",
                    },
                    "delta_commands": {
                        "type": "array",
                        "description": "Only the incremental changes required by the request.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": [
                                        "update_config",
                                        "modify_node",
                                        "configure_disturbance",
                                        "set_rl_algo",
                                        "set_execution_mode",
                                        "set_specific_model",
                                    ],
                                },
                                "target": {"type": "string"},
                                "value": {},
                            },
                            "required": ["action", "target", "value"],
                        },
                    },
                },
                "required": ["thoughts", "delta_commands"],
            },
        },
    }
]

SYSTEM_PROMPT = """
You are the GridSage scenario configuration agent.
Your job is to convert user natural language into valid Delta Commands.

Hard rules:
- Output commands through the generate_lvgs_actions tool.
- Do not invent ScenarioConfig fields.
- Use update_config for global fields such as grid_model, algo_name, execution_mode,
  target_model_steps, global_pv_multiplier, global_load_multiplier, global_ev_multiplier,
  start_hour, end_hour, step_minutes, use_pv, use_wind, use_ess, use_sop, use_nop,
  reconfiguration_mode, selected_reconfiguration_plan_id, reconfiguration_constraints,
  rl_hyperparams, active_skill_ids, scenario_name, scenario_description, time_profiles,
  disabled_devices.
- Use modify_node for node overrides. The target must be a node id such as b18, and value
  may contain add_pv_kw, add_wind_kw, add_load_kw, add_ev_spots, add_ess_kwh,
  add_ess_power_kw, add_ess_c_rate.
- Default devices shown in the topology are not node_overrides. If the user asks to remove,
  delete, close, or disable an existing/default device, use update_config target
  disabled_devices. Shape: {"b3": {"generator": ["*"]}} to disable all generators on b3,
  or {"b11": {"pv": ["pv2"]}} to disable a named device. Supported device types:
  generator, pv, wind, ess, ev_station, sop, nop.
- NOP is a radial topology reconfiguration action, not a power controller. If the user asks
  to close a NOP, set selected_reconfiguration_plan_id to a legal R-plan when known, or explain
  that a same-loop base line must also be opened. Do not create separate NOP P/Q commands.
- If the user asks to train, retrain, or learn, set execution_mode='train'. If they
  specify an algorithm, set algo_name. If they specify steps, set
  rl_hyperparams.total_timesteps.
- If the user asks to evaluate/test/use a trained model, set execution_mode='evaluate'.
- Phrases like "trained model" or "pretrained model" refer to an existing model for
  evaluation; do not treat them as a new training request.
- If matched Scenario Skills are provided, use their defaults when the user is vague.
- Respect user-specified values unless they violate safety boundaries.
- Dangerous or impossible requests should be explained and should not produce dangerous commands.
"""


def _state_json(state: ScenarioConfig) -> str:
    if hasattr(state, "model_dump_json"):
        return state.model_dump_json(indent=2)
    return state.json(indent=2)


def _load_tool_arguments(raw_args):
    if isinstance(raw_args, dict):
        return raw_args
    raw = (raw_args or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidate = raw[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return parsed
    raise ValueError(f"Model returned invalid tool arguments: {raw[:300]}")


def _extract_steps(text: str) -> Optional[int]:
    patterns = [
        r"(?:train|retrain|training|steps?|timesteps?)\D{0,12}(\d{1,9})\s*(?:step|steps?|timesteps?)?",
        r"(\d{1,9})\s*(?:steps?|step|timesteps?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _extract_algo(text: str) -> Optional[str]:
    upper = (text or "").upper()
    if "BASELINE" in upper:
        return "Baseline"
    for algo in SUPPORTED_ALGOS:
        if re.search(rf"(?<![A-Z0-9]){algo}(?![A-Z0-9])", upper):
            return algo
    return None


def _is_evaluation_request(text: str) -> bool:
    text = text or ""
    explicit_evaluation = re.search(
        r"\b(evaluate|evaluation|test|compare|comparison|infer|inference)\b",
        text,
        re.IGNORECASE,
    )
    trained_model_use = _is_trained_model_reference(text) and re.search(
        r"\b(use|run|load|apply|deploy|execute)\b",
        text,
        re.IGNORECASE,
    )
    return bool(explicit_evaluation or trained_model_use)


def _is_trained_model_reference(text: str) -> bool:
    return bool(re.search(r"\b(trained|pretrained|pre-trained)\b", text or "", re.IGNORECASE))


def _is_train_request(text: str) -> bool:
    return bool(
        re.search(
            r"\b(train|training|retrain|retraining|learn|learning|finetune|fine-tune|fine-tuning)\b",
            text or "",
            re.IGNORECASE,
        )
    )


def _extract_multiplier(text: str, keywords: List[str]) -> Optional[float]:
    for keyword in keywords:
        match = re.search(rf"{keyword}.{{0,12}}?(\d+(?:\.\d+)?)\s*x", text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        percent = re.search(rf"{keyword}.{{0,12}}?(\d+(?:\.\d+)?)\s*%", text, re.IGNORECASE)
        if percent:
            return float(percent.group(1)) / 100.0
    return None


def _range_profile(start: int, end: int, value: float, default: float):
    data = {str(hour): default for hour in range(24)}
    for hour in range(start, end + 1):
        data[str(hour % 24)] = value
    return data


def _local_time_profile(text: str):
    lower = (text or "").lower()
    if "duck curve" in text or "duck curve" in lower:
        return {
            "profile_name": "duck_curve",
            "profile_description": "PV output is high at noon, and load and EV charging increase in the evening.",
            "load_multiplier_by_hour": _range_profile(18, 22, 1.5, 1.0),
            "pv_multiplier_by_hour": _range_profile(10, 14, 2.0, 0.2),
            "ev_multiplier_by_hour": _range_profile(18, 22, 2.5, 0.3),
        }
    if "evening" in lower and ("ev" in lower or "charge" in lower):
        return {
            "profile_name": "evening_ev_charging",
            "profile_description": "EV centralized charging in the evening.",
            "load_multiplier_by_hour": _range_profile(18, 22, 1.2, 1.0),
            "pv_multiplier_by_hour": _range_profile(18, 22, 0.1, 1.0),
            "ev_multiplier_by_hour": _range_profile(18, 22, 2.5, 0.5),
        }
    return None


def _local_skill_response(user_input: str, matched_skills: Optional[List[MatchedSkill]]) -> Optional[ChatResponse]:
    if not matched_skills:
        return None

    primary = matched_skills[0]
    skill = primary.skill
    if not skill:
        return None

    text = user_input or ""
    commands = []
    notes = [f"{skill.skill_id}/{skill.name_cn} has been hit."]
    time_profile = _local_time_profile(text)

    for field, rule in skill.recommended_parameters.items():
        if time_profile and field in {"global_pv_multiplier", "global_load_multiplier", "global_ev_multiplier"}:
            continue
        value = rule.default
        if field == "global_pv_multiplier":
            value = _extract_multiplier(text, ["pv", "PV"]) or value
        elif field == "global_load_multiplier":
            value = _extract_multiplier(text, ["load", "load"]) or value
        elif field == "global_ev_multiplier":
            value = _extract_multiplier(text, ["EV", "electric vehicle", "charging"]) or value
        if value is not None:
            commands.append({"action": "update_config", "target": field, "value": value})

    if time_profile:
        commands.append({"action": "update_config", "target": "time_profiles", "value": time_profile})
        notes.append("Time_profiles have been generated.")

    algo = _extract_algo(text)
    if _is_train_request(text):
        commands.append({"action": "update_config", "target": "execution_mode", "value": "train"})
        commands.append(
            {
                "action": "update_config",
                "target": "algo_name",
                "value": algo or skill.recommended_rl.default_algorithm,
            }
        )
        steps = _extract_steps(text)
        if steps is not None:
            commands.append(
                {
                    "action": "update_config",
                    "target": "rl_hyperparams",
                    "value": {"total_timesteps": steps},
                }
            )
    elif _is_evaluation_request(text):
        commands.append({"action": "update_config", "target": "execution_mode", "value": "evaluate"})
        if algo:
            commands.append({"action": "update_config", "target": "algo_name", "value": algo})
    elif algo:
        commands.append({"action": "update_config", "target": "algo_name", "value": algo})

    commands.extend(
        [
            {"action": "update_config", "target": "active_skill_ids", "value": [m.skill_id for m in matched_skills]},
            {"action": "update_config", "target": "scenario_name", "value": skill.name_cn},
            {"action": "update_config", "target": "scenario_description", "value": skill.description_cn},
        ]
    )

    if primary.skill_id == "S02_HEAVY_LOAD_END_NODES" and not re.search(r"\bb\d+\b", text, re.IGNORECASE):
        notes.append("When no node is specified, only the global load parameters are set first and no node disturbance is automatically added.")

    return ChatResponse(
        thoughts=" ".join(notes),
        delta_commands=commands,
        status="success",
    )


async def chat_with_agent(
    user_input: str,
    current_state: ScenarioConfig,
    api_config: dict,
    memory: list = None,
    matched_skills: Optional[List[MatchedSkill]] = None,
    skill_context: Optional[str] = None,
) -> ChatResponse:
    api_key = api_config.get("api_key", "").strip()
    base_url = api_config.get("base_url", "https://api.openai.com/v1").strip()
    model_name = api_config.get("model_name", "gpt-4o-mini").strip()

    if not api_key:
        local = _local_skill_response(user_input, matched_skills)
        if local:
            local.thoughts = "The API Key is not configured, and the Scenario Skill local default rules have been used to generate the minimum available configuration. " + local.thoughts
            return local
        return ChatResponse(
            thoughts="Please configure the model API Key first; when S01/S02 is not hit, local rules will not generate complex scene configurations without authorization.",
            status="error",
            error_msg="No API Key",
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Current ScenarioConfig:\n{_state_json(current_state)}"},
    ]
    if skill_context:
        messages.append({"role": "system", "content": skill_context})
    if memory:
        messages.extend(memory)
    messages.append({"role": "user", "content": user_input})

    try:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url if base_url else None)
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "generate_lvgs_actions"}},
            temperature=0.2,
        )
        message = response.choices[0].message
        if getattr(message, "tool_calls", None):
            args = _load_tool_arguments(message.tool_calls[0].function.arguments)
        else:
            args = _load_tool_arguments(getattr(message, "content", "") or "")
        return ChatResponse(
            thoughts=args.get("thoughts", "The scene configuration increment has been generated."),
            delta_commands=args.get("delta_commands", []),
            status="success",
        )
    except Exception as exc:
        local = _local_skill_response(user_input, matched_skills)
        if local:
            local.thoughts = f"The model structured call failed, and the Scenario Skill local default rules have been used to cover the problem. Error: {exc}\n\n{local.thoughts}"
            return local
        return ChatResponse(
            thoughts=f"Failed to communicate with the model service or parse structured output: {exc}",
            status="error",
            error_msg=str(exc),
        )
