import hashlib
import json


SCENARIO_FIELDS = (
    "grid_model",
    "start_hour",
    "end_hour",
    "step_minutes",
    "use_pv",
    "use_wind",
    "use_ess",
    "use_sop",
    "use_nop",
    "reconfiguration_mode",
    "selected_reconfiguration_plan_id",
    "reconfiguration_constraints",
    "global_pv_multiplier",
    "global_load_multiplier",
    "global_ev_multiplier",
    "time_profiles",
    "node_overrides",
    "disabled_devices",
)


def _normalize(value):
    if isinstance(value, dict):
        return {str(k): _normalize(value[k]) for k in sorted(value.keys(), key=str)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 10)
    return str(value).strip()


def _get_energy_flag(settings, flag_name, nested_name, default=True):
    if flag_name in settings:
        return bool(settings.get(flag_name))
    distributed = settings.get("distributed_energy") or {}
    if nested_name in distributed:
        return bool(distributed.get(nested_name))
    return default


def build_scenario_signature(settings):
    settings = settings or {}
    signature = {
        "grid_model": str(settings.get("grid_model", "ieee33")).strip().lower(),
        "start_hour": _normalize(settings.get("start_hour", 0)),
        "end_hour": _normalize(settings.get("end_hour", 24)),
        "step_minutes": _normalize(settings.get("step_minutes", 60)),
        "use_pv": _get_energy_flag(settings, "use_pv", "pv"),
        "use_wind": _get_energy_flag(settings, "use_wind", "wind"),
        "use_ess": _get_energy_flag(settings, "use_ess", "ess"),
        "use_sop": bool(settings.get("use_sop", settings.get("sop_nodes_active", True))),
        "use_nop": bool(settings.get("use_nop", settings.get("nop_nodes_active", True))),
        "reconfiguration_mode": _normalize(settings.get("reconfiguration_mode", "radial_reconfiguration")),
        "selected_reconfiguration_plan_id": _normalize(settings.get("selected_reconfiguration_plan_id", "R0")),
        "reconfiguration_constraints": _normalize(settings.get("reconfiguration_constraints") or {}),
        "global_pv_multiplier": _normalize(settings.get("global_pv_multiplier", 1.0)),
        "global_load_multiplier": _normalize(settings.get("global_load_multiplier", 1.0)),
        "global_ev_multiplier": _normalize(settings.get("global_ev_multiplier", 1.0)),
        "time_profiles": _normalize(settings.get("time_profiles") or {}),
        "node_overrides": _normalize(settings.get("node_overrides") or {}),
        "disabled_devices": _normalize(settings.get("disabled_devices") or {}),
    }
    return {field: signature[field] for field in SCENARIO_FIELDS}


def build_scenario_fingerprint(settings):
    payload = json.dumps(
        build_scenario_signature(settings),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
