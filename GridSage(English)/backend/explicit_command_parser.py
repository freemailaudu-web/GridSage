import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple


SUPPORTED_ALGOS = ("SAC", "PPO", "TD3", "DDPG")

_TIME_RANGE_PATTERN = (
    r"(?<![\d.])(\d{1,2})(?::[0-5]\d)?\s*(?:hours?|hrs?|h)?\s*"
    r"(?:to|through|until|[-~–—])\s*"
    r"(\d{1,2})(?::[0-5]\d)?\s*(?:hours?|hrs?|h)?(?!\d)"
)


def _number_near_keyword(text: str, keywords: List[str]) -> Optional[float]:
    text = text or ""
    for keyword in keywords:
        escaped = re.escape(keyword)
        keyword_pattern = rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
        multiplier_patterns = [
            rf"{keyword_pattern}.{{0,80}}?(-?\d+(?:\.\d+)?)\s*(x|times?|%)",
            rf"(-?\d+(?:\.\d+)?)\s*(x|times?|%).{{0,30}}?{keyword_pattern}",
        ]
        for pattern in multiplier_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue
            value = float(match.group(1))
            return value / 100.0 if match.group(2) == "%" else value

        text_without_time_ranges = re.sub(_TIME_RANGE_PATTERN, " ", text, flags=re.IGNORECASE)
        match = re.search(
            rf"{keyword_pattern}.{{0,30}}?"
            rf"(?:set(?:\s+to)?|change(?:\s+to)?|adjust(?:\s+to)?|raise(?:\s+to)?|lower(?:\s+to)?|to)"
            rf"[^\d-]{{0,8}}(-?\d+(?:\.\d+)?)",
            text_without_time_ranges,
            re.IGNORECASE,
        )
        if match:
            return float(match.group(1))
    return None


def _merge_time_profile_value(existing: Any, updates: Any) -> Any:
    if not isinstance(existing, dict) or not isinstance(updates, dict):
        return deepcopy(updates)
    merged = deepcopy(existing)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = deepcopy(value)
    return merged


def _upsert_command(commands: List[Dict[str, Any]], target: str, value: Any, overwrite: bool = True) -> None:
    for command in commands:
        if command.get("action") == "update_config" and command.get("target") == target:
            if not overwrite:
                return
            if target == "time_profiles":
                command["value"] = _merge_time_profile_value(command.get("value"), value)
            else:
                command["value"] = value
            return
    commands.append({"action": "update_config", "target": target, "value": value})


def _range_profile(start: int, end: int, value: float, default: float) -> Dict[str, float]:
    profile = {str(hour): default for hour in range(24)}
    for hour in range(start, end + 1):
        profile[str(hour % 24)] = value
    return profile


def _merge_hour_ranges(base: Dict[str, float], start: int, end: int, value: float) -> Dict[str, float]:
    merged = dict(base)
    for hour in range(start, end + 1):
        merged[str(hour % 24)] = value
    return merged


def _has_time_expression(text: str) -> bool:
    lower = (text or "").lower()
    return bool(
        re.search(_TIME_RANGE_PATTERN, text or "", re.IGNORECASE)
        or any(word in lower for word in ["noon", "evening", "night", "early morning", "time profile", "timing", "mismatch", "duck curve"])
    )


def _hour_range(text: str, default_start: int, default_end: int) -> Tuple[int, int]:
    match = re.search(_TIME_RANGE_PATTERN, text or "", re.IGNORECASE)
    if match:
        start = max(0, min(23, int(match.group(1))))
        end = max(0, min(23, int(match.group(2))))
        if end < start:
            start, end = end, start
        return start, end
    return default_start, default_end


def _phrase_has_any(text: str, keywords: List[str]) -> bool:
    lower = (text or "").lower()
    return any(keyword.lower() in lower for keyword in keywords)


def _default_directional_value(text: str, high_value: float, low_value: float, neutral: float = 1.0) -> Optional[float]:
    if re.search(r"\b(increase|raise|high|large|concentrated|peak)\b", text or "", re.IGNORECASE):
        return high_value
    if re.search(r"\b(reduce|decrease|lower|low|cut)\b", text or "", re.IGNORECASE):
        return low_value
    return neutral


def _detect_time_profile(text: str) -> Optional[Dict[str, Any]]:
    lower = (text or "").lower()
    if "duck curve" in lower:
        return {
            "profile_name": "duck_curve",
            "profile_description": "PV output is high at noon, and load and EV charging increase in the evening.",
            "load_multiplier_by_hour": _merge_hour_ranges(_range_profile(18, 22, 1.5, 1.0), 0, 6, 0.7),
            "pv_multiplier_by_hour": _merge_hour_ranges(_range_profile(10, 14, 2.0, 0.2), 0, 6, 0.0),
            "ev_multiplier_by_hour": _range_profile(18, 22, 2.5, 0.3),
        }
    if "source load" in lower or "source-load" in lower or "mismatch" in lower:
        return {
            "profile_name": "custom_time_profile",
            "profile_description": "Source-load timing mismatch: PV is high at noon, load and EV charging are high in the evening.",
            "load_multiplier_by_hour": _range_profile(18, 22, 1.5, 1.0),
            "pv_multiplier_by_hour": _range_profile(10, 14, 2.0, 1.0),
            "ev_multiplier_by_hour": _range_profile(18, 22, 2.5, 1.0),
        }
    if "evening" in lower and ("ev" in lower or "charg" in lower):
        return {
            "profile_name": "evening_ev_charging",
            "profile_description": "EV centralized charging in the evening.",
            "load_multiplier_by_hour": _range_profile(18, 22, 1.2, 1.0),
            "pv_multiplier_by_hour": _range_profile(18, 22, 0.1, 1.0),
            "ev_multiplier_by_hour": _range_profile(18, 22, 2.5, 0.5),
        }

    if not _has_time_expression(text):
        return None

    profile: Dict[str, Any] = {
        "profile_name": "custom_time_profile",
        "profile_description": "Time-sharing magnification generated by natural language analysis.",
    }
    has_component = False

    if _phrase_has_any(text, ["photovoltaic", "pv", "solar"]):
        start, end = _hour_range(text, 10, 14)
        pv_value = _number_near_keyword(text, ["photovoltaic", "pv", "solar"])
        if pv_value is None:
            pv_value = _default_directional_value(text, 2.0, 0.2)
        profile["pv_multiplier_by_hour"] = _range_profile(start, end, float(pv_value), 1.0)
        has_component = True

    if _phrase_has_any(text, ["load"]):
        default_range = (0, 6) if any(word in text for word in ["night", "early morning"]) else (18, 22)
        start, end = _hour_range(text, *default_range)
        load_value = _number_near_keyword(text, ["load"])
        if load_value is None:
            load_value = _default_directional_value(text, 1.5, 0.7)
        profile["load_multiplier_by_hour"] = _range_profile(start, end, float(load_value), 1.0)
        has_component = True

    if _phrase_has_any(text, ["ev", "electric vehicle", "charging"]):
        start, end = _hour_range(text, 18, 22)
        ev_value = _number_near_keyword(text, ["ev", "electric vehicle", "charging"])
        if ev_value is None:
            ev_value = _default_directional_value(text, 2.5, 0.5)
        profile["ev_multiplier_by_hour"] = _range_profile(start, end, float(ev_value), 1.0)
        has_component = True

    return profile if has_component else None


def _remove_bad_baseline_mode(commands: List[Dict[str, Any]]) -> bool:
    found = False
    kept = []
    for command in commands:
        target = str(command.get("target", ""))
        value = str(command.get("value", ""))
        if target == "mode" and value.lower() == "baseline":
            found = True
            continue
        kept.append(command)
    commands[:] = kept
    return found


def _remove_targets(commands: List[Dict[str, Any]], targets: set) -> None:
    commands[:] = [
        command
        for command in commands
        if not (command.get("action") == "update_config" and command.get("target") in targets)
    ]


def _has_modify_node_value(commands: List[Dict[str, Any]], node_id: str, key: str) -> bool:
    for command in commands:
        if command.get("action") != "modify_node" or command.get("target") != node_id:
            continue
        value = command.get("value")
        if isinstance(value, dict) and key in value:
            return True
    return False


def _number_before_units(text: str, units_pattern: str) -> Optional[float]:
    match = re.search(rf"(\d+(?:\.\d+)?)\s*(?:{units_pattern})", text or "", re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _extract_nodes(text: str) -> List[str]:
    text = text or ""
    raw_nodes: List[str] = []
    for pattern in [
        r"(?<![A-Za-z0-9])[bB](\d{1,4})(?![A-Za-z0-9])",
        r"(?:node|bus)\s*(\d{1,4})",
        r"(\d{1,4})\s*(?:node|bus)",
    ]:
        for match in re.finditer(pattern, text):
            raw_nodes.append(f"b{int(match.group(1))}")
    return list(dict.fromkeys(raw_nodes))


def _merge_modify_node_command(commands: List[Dict[str, Any]], node_id: str, payload: Dict[str, Any]) -> None:
    for command in commands:
        if command.get("action") == "modify_node" and command.get("target") == node_id:
            existing = command.get("value")
            if not isinstance(existing, dict):
                existing = {}
            existing.update(payload)
            command["value"] = existing
            return
    commands.append({"action": "modify_node", "target": node_id, "value": payload})


def _merge_disabled_devices_command(commands: List[Dict[str, Any]], node_id: str, device_type: str) -> None:
    value = {node_id: {device_type: ["*"]}}
    for command in commands:
        if command.get("action") == "update_config" and command.get("target") == "disabled_devices":
            existing = command.get("value")
            if not isinstance(existing, dict):
                existing = {}
            existing.setdefault(node_id, {})
            existing[node_id].setdefault(device_type, [])
            if "*" not in existing[node_id][device_type]:
                existing[node_id][device_type].append("*")
            command["value"] = existing
            return
    commands.append({"action": "update_config", "target": "disabled_devices", "value": value})


def _append_explicit_node_commands(text: str, commands: List[Dict[str, Any]]) -> None:
    text = text or ""
    nodes = _extract_nodes(text)
    if not nodes:
        return
    lower = text.lower()

    delete_request = bool(re.search(r"\b(delete|remove|deactivate|disable|turn off)\b", text, re.IGNORECASE))
    if delete_request:
        device_type = None
        if "generator" in lower or re.search(r"\bgen\b", lower):
            device_type = "generator"
        elif "pv" in lower or "photovoltaic" in lower or "solar" in lower:
            device_type = "pv"
        elif "wind" in lower:
            device_type = "wind"
        elif "ess" in lower or "battery" in lower or "energy storage" in lower:
            device_type = "ess"
        elif "charging station" in lower or "ev" in lower or "charging" in lower:
            device_type = "ev_station"
        elif "sop" in lower:
            device_type = "sop"
        elif "nop" in lower:
            device_type = "nop"
        if device_type:
            for node_id in nodes:
                _merge_disabled_devices_command(commands, node_id, device_type)
            return

    payload = {}
    if "pv" in lower or "photovoltaic" in lower or "solar" in lower:
        value = _number_before_units(text, r"kw|kilowatts?")
        payload["add_pv_kw"] = value if value is not None else 300.0
    if "wind" in lower:
        value = _number_before_units(text, r"kw|kilowatts?")
        payload["add_wind_kw"] = value if value is not None else 300.0
    if "ess" in lower or "battery" in lower or "energy storage" in lower:
        value = _number_before_units(text, r"kwh|kilowatt[- ]?hours?")
        power_value = _number_before_units(text, r"kw(?!h)|kilowatts?(?![- ]?hours?)")
        c_rate_match = re.search(r"(\d+(?:\.\d+)?)\s*c\b", lower)
        payload["add_ess_kwh"] = value if value is not None else 300.0
        if power_value is not None:
            payload["add_ess_power_kw"] = power_value
        if c_rate_match:
            payload["add_ess_c_rate"] = float(c_rate_match.group(1))
    if "ev" in lower or "charg" in lower or "electric vehicle" in lower:
        value = _number_before_units(
            text,
            r"(?:ev\s*)?(?:charging\s*)?(?:spots?|chargers?|ports?|points?|piles?|stalls?|parking\s+spaces?|units?)",
        )
        payload["add_ev_spots"] = int(value) if value is not None else 5
    if "load" in lower:
        value = _number_before_units(text, r"kw|kilowatts?")
        payload["add_load_kw"] = value if value is not None else 300.0
    if payload:
        for node_id in nodes:
            missing_payload = {
                key: value
                for key, value in payload.items()
                if not _has_modify_node_value(commands, node_id, key)
            }
            if missing_payload:
                _merge_modify_node_command(commands, node_id, missing_payload)


def _explicit_algo(text: str, commands: List[Dict[str, Any]]) -> Optional[str]:
    lower = (text or "").lower()
    upper = (text or "").upper()

    if "baseline" in lower or "baseline" in text:
        return "Baseline"

    for command in commands:
        target = str(command.get("target", ""))
        value = str(command.get("value", ""))
        if target in {"algo_name", "mode"} and value.lower() == "baseline":
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
    text = text or ""
    return bool(
        re.search(
            r"\b(train|training|retrain|retraining|learn|learning|finetune|fine-tune|fine-tuning)\b",
            text,
            re.IGNORECASE,
        )
    )


def _extract_steps(text: str) -> Optional[int]:
    patterns = [
        r"(?<![A-Za-z0-9])(\d{1,9})\s*(?:steps?|step|timesteps?)",
        r"(?:train|retrain|training|retraining)\D{0,40}?(?<![A-Za-z])(\d{1,9})(?![A-Za-z])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _model_reference_steps(text: str) -> Optional[int]:
    if not _is_evaluation_request(text):
        return None
    if not re.search(r"(model|model|checkpoint|weight)", text or "", re.IGNORECASE):
        return None
    return _extract_steps(text)


def _append_unique(items: List[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _requested_comparison_targets(text: str, commands: List[Dict[str, Any]]) -> List[str]:
    text = text or ""
    has_eval = _is_evaluation_request(text)
    targets: List[str] = []
    token_pattern = r"baseline|SAC|PPO|TD3|DDPG"
    for match in re.finditer(token_pattern, text, re.IGNORECASE):
        token = match.group(0)
        if token.lower() == "baseline":
            _append_unique(targets, "Baseline")
        else:
            _append_unique(targets, token.upper())

    has_explicit_targets = False
    for command in commands:
        if command.get("target") == "evaluation_targets":
            value = command.get("value")
            if isinstance(value, list):
                has_explicit_targets = len(value) > 1
                for item in value:
                    algo = item.get("algo") if isinstance(item, dict) else item
                    algo_text = str(algo or "").upper()
                    if algo_text == "BASELINE":
                        _append_unique(targets, "Baseline")
                    elif algo_text in SUPPORTED_ALGOS:
                        _append_unique(targets, algo_text)

    if (has_eval or has_explicit_targets) and len(targets) >= 2:
        return targets
    return []


def merge_explicit_global_commands(user_input: str, commands: List[Any]) -> List[Dict[str, Any]]:
    """Make explicit user settings auditable even if the LLM omits or misroutes them."""

    merged = [
        command.model_dump() if hasattr(command, "model_dump") else dict(command)
        for command in (commands or [])
    ]
    text = user_input or ""
    _append_explicit_node_commands(text, merged)

    bad_baseline_mode = _remove_bad_baseline_mode(merged)
    algo = _explicit_algo(text, merged)
    eval_request = _is_evaluation_request(text)
    train_request = _is_train_request(text)
    train_steps = _extract_steps(text)
    model_steps = _model_reference_steps(text)
    comparison_targets = _requested_comparison_targets(text, merged)
    if comparison_targets:
        _upsert_command(merged, "execution_mode", "evaluate")
        _upsert_command(
            merged,
            "evaluation_targets",
            [{"algo": target} for target in comparison_targets],
        )
    elif algo == "Baseline" or bad_baseline_mode:
        _upsert_command(merged, "algo_name", "Baseline")
        _upsert_command(merged, "execution_mode", "evaluate")
        _upsert_command(merged, "rl_hyperparams", {})
        _upsert_command(merged, "target_model_steps", None)
        _upsert_command(merged, "specific_model_name", "")
    elif eval_request:
        _upsert_command(merged, "execution_mode", "evaluate")
        if algo:
            _upsert_command(merged, "algo_name", algo)
        if model_steps is not None:
            _upsert_command(merged, "target_model_steps", model_steps)
    elif algo:
        _upsert_command(merged, "algo_name", algo)

    if train_request and not eval_request and algo != "Baseline" and not bad_baseline_mode:
        _upsert_command(merged, "execution_mode", "train")
        _upsert_command(merged, "specific_model_name", "")
        _upsert_command(merged, "target_model_steps", None)
        if train_steps is not None:
            existing_hyperparams = {}
            for command in merged:
                if command.get("action") == "update_config" and command.get("target") == "rl_hyperparams":
                    if isinstance(command.get("value"), dict):
                        existing_hyperparams = dict(command["value"])
                    break
            existing_hyperparams["total_timesteps"] = train_steps
            _upsert_command(merged, "rl_hyperparams", existing_hyperparams)

    time_profile = _detect_time_profile(text)
    pv_value = _number_near_keyword(text, ["photovoltaic", "pv", "solar"])
    load_value = _number_near_keyword(text, ["load"])
    ev_value = _number_near_keyword(text, ["ev", "electric vehicle", "charging"])

    if pv_value is not None and not time_profile:
        _upsert_command(merged, "global_pv_multiplier", pv_value)
    if load_value is not None and not time_profile:
        _upsert_command(merged, "global_load_multiplier", load_value)
    if ev_value is not None and not time_profile:
        _upsert_command(merged, "global_ev_multiplier", ev_value)

    if time_profile:
        _upsert_command(merged, "time_profiles", time_profile)
        if "scene" in text.lower() or "duck curve" in text.lower():
            scenario_name = "Duck Curve Timing Scenario" if time_profile.get("profile_name") == "duck_curve" else "Time-sharing Timing Scenario"
            _upsert_command(merged, "scenario_name", scenario_name)
            _upsert_command(merged, "scenario_description", time_profile.get("profile_description", "Time-sharing load, photovoltaic and EV rate scenarios."))
            _upsert_command(merged, "active_skill_ids", [])

    return merged
