import sys
import os
import io
import asyncio
import json
import re
from contextlib import redirect_stdout, redirect_stderr

# Compatibility note: vgridsim_core remains the legacy runtime directory name to avoid breaking imports.
# Use insert(0) so the bundled vgridsim_core package takes priority over modules with the same name.
_CORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vgridsim_core")
if _CORE_PATH not in sys.path:
    sys.path.insert(0, _CORE_PATH)

import config
from scenario_fingerprint import build_scenario_fingerprint, build_scenario_signature
from .schema import ScenarioConfig
from .experiment_logger import log_run_finish
from .result_interpreter import interpret_results_with_skills


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _prepare_training_subprocess_env(train_steps) -> dict:
    # Compatibility note: LVGS_* keys remain unchanged because child processes consume them.
    env_copy = os.environ.copy()
    env_copy["TRAINING_TOTAL_TIMESTEPS"] = str(train_steps)
    env_copy["PYTHONUTF8"] = "1"
    env_copy["PYTHONIOENCODING"] = "utf-8"
    env_copy["TQDM_DISABLE"] = "1"
    env_copy["LVGS_SUPPRESS_WINDOWS_ERROR_DIALOG"] = "1"
    env_copy["LVGS_FAST_TRAINING"] = "1"
    return env_copy


def _clean_process_log_line(line: str) -> str:
    text = _ANSI_RE.sub("", str(line or "")).strip()
    return text.replace("\ufffd", "")


def _extract_steps(value):
    """Extract Training Steps from int/string, such as 1000, "1000 steps", "sac_2stage_1000steps"."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value)
    m = re.search(r"(\d+)\s*(?:steps?|timesteps?)?", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _read_model_manifest(item_path: str) -> dict:
    manifest_path = os.path.join(item_path, "model_manifest.json") if os.path.isdir(item_path) else ""
    if manifest_path and os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _pick_zip_from_item(item_path: str):
    """Given the files/folders in the models directory, return the actual loadable model zip."""
    if os.path.isfile(item_path) and item_path.lower().endswith(".zip"):
        return item_path
    if not os.path.isdir(item_path):
        return None

    # The new version directory will give priority to loading the final model at the end of training; the old directory will be compatible with best_model.zip.
    preferred = ["final_model.zip", "best_model.zip"]
    for filename in preferred:
        cand = os.path.join(item_path, filename)
        if os.path.exists(cand):
            return cand

    sub_files = [os.path.join(item_path, f) for f in os.listdir(item_path) if f.lower().endswith(".zip")]
    if not sub_files:
        return None
    sub_files.sort(key=os.path.getmtime, reverse=True)
    return sub_files[0]


def _find_requested_model(
    models_dir: str,
    algo: str,
    target_steps=None,
    specific_model_name: str = "",
    scenario_fingerprint: str = "",
    require_scenario_match: bool = False,
):
    """
    User-specified Training Steps must be respected when finding models.
    - If target_steps is specified but cannot be found, an error will be reported directly and no longer silently return to "latest model".
    - If the number of steps/name is not specified, the latest model of the Algorithm is returned.
    """
    if not os.path.exists(models_dir):
        return None

    algo_key = str(algo).upper()
    algo_lower = str(algo).lower()
    specific = (specific_model_name or "").strip().lower()
    target_steps = _extract_steps(target_steps) or _extract_steps(specific)

    all_items = [os.path.join(models_dir, item) for item in os.listdir(models_dir)]
    all_items.sort(key=os.path.getmtime, reverse=True)

    matches = []
    for item_path in all_items:
        model_zip = _pick_zip_from_item(item_path)
        if not model_zip:
            continue

        item_name = os.path.basename(item_path).lower()
        zip_name = os.path.basename(model_zip).lower()
        manifest = _read_model_manifest(item_path)
        manifest_text = json.dumps(manifest, ensure_ascii=False).lower() if manifest else ""

        manifest_algo = str(manifest.get("algo", "")).upper()
        algo_match = (manifest_algo == algo_key) or (algo_lower in item_name) or (algo_lower in zip_name)
        if not algo_match:
            continue

        if target_steps is not None:
            manifest_steps = _extract_steps(manifest.get("total_timesteps"))
            name_steps = _extract_steps(item_name) or _extract_steps(zip_name)
            if manifest_steps != target_steps and name_steps != target_steps:
                continue
        elif specific:
            if specific not in item_name and specific not in zip_name and specific not in manifest_text:
                continue

        if require_scenario_match:
            manifest_fingerprint = str(manifest.get("scenario_fingerprint", "") or "")
            if not manifest_fingerprint or manifest_fingerprint != scenario_fingerprint:
                continue

        matches.append(model_zip)

    if target_steps is not None and not matches and require_scenario_match:
        return None

    if target_steps is not None and not matches:
        raise FileNotFoundError(
            f"The user specified the {target_steps} step model of {algo_key}, but no model with the step version tag was found in the models directory."
            f"Please train {target_steps} steps first, or confirm that the model directory name/manifest contains {target_steps}steps."
        )

    return matches[0] if matches else None


def _find_recent_trained_model_after(models_dir: str, algo: str, target_steps, started_at: float):
    if not os.path.exists(models_dir):
        return None

    algo_lower = str(algo).lower()
    target_steps = _extract_steps(target_steps)
    candidates = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.getmtime(item_path) < started_at:
            continue
        manifest = _read_model_manifest(item_path)
        manifest_steps = _extract_steps(manifest.get("total_timesteps"))
        item_steps = _extract_steps(item)
        manifest_algo = str(manifest.get("algo", "")).lower()
        if manifest_algo and manifest_algo != algo_lower:
            continue
        if algo_lower not in item.lower() and manifest_algo != algo_lower:
            continue
        if target_steps is not None and manifest_steps != target_steps and item_steps != target_steps:
            continue
        model_zip = _pick_zip_from_item(item_path)
        if model_zip:
            candidates.append(model_zip)

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None


def _state_to_scenario_settings(state: ScenarioConfig) -> dict:
    if hasattr(state, "model_dump"):
        return state.model_dump()
    return state.dict()


def _assert_model_matches_current_scenario(model_path: str, state: ScenarioConfig, algo: str, label: str) -> dict:
    manifest = _read_model_manifest(os.path.dirname(model_path)) if model_path else {}
    current_settings = _state_to_scenario_settings(state)
    current_fingerprint = build_scenario_fingerprint(current_settings)
    model_fingerprint = str(manifest.get("scenario_fingerprint", "") or "")

    if not model_fingerprint:
        raise ValueError(
            f"{algo} model {os.path.basename(model_path)} found, but the model does not have a scene fingerprint,"
            f"It is impossible to confirm whether it was trained under the current distribution network equipment configuration. In order to avoid applying the old scene model to the new scene,"
            f"This evaluation has been stopped. Please retrain {algo} in the current scenario before performing the evaluation."
        )

    if model_fingerprint != current_fingerprint:
        trained_signature = manifest.get("scenario_signature") or {}
        current_signature = build_scenario_signature(current_settings)
        changed_fields = [
            field for field in sorted(current_signature.keys())
            if trained_signature.get(field) != current_signature.get(field)
        ]
        changed_text = ", ".join(changed_fields) if changed_fields else "unknown scenario fields"
        raise ValueError(
            f"{label}'s model scene fingerprint is inconsistent with the current scene, and evaluation has been blocked."
            f"Inconsistent field: {changed_text}."
            f"Please retrain {algo} with the current distribution network equipment/rate/node disturbance before evaluating."
        )

    return manifest


def _format_metric_cell(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _has_ev_demand(state: ScenarioConfig) -> bool:
    if float(getattr(state, "global_ev_multiplier", 1.0) or 0.0) > 0:
        return True
    for params in getattr(state, "node_overrides", {}).values():
        if float(params.get("add_ev_spots", 0) or 0) > 0:
            return True
    return False


def _filter_metrics_for_state(metrics: dict, state: ScenarioConfig) -> dict:
    if _has_ev_demand(state):
        return metrics
    return {key: value for key, value in metrics.items() if "EV Charging Satisfaction Rate" not in key}


def _write_gui_settings_for_state(state: ScenarioConfig) -> None:
    settings_path = config.PATHS.get("gui_settings")
    settings = {}
    if settings_path and os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
        except Exception:
            settings = {}

    settings.update({
        "grid_model": state.grid_model,
        "solver": state.solver,
        "start_hour": int(state.start_hour),
        "end_hour": int(state.end_hour),
        "step_minutes": int(state.step_minutes),
        "use_pv": bool(state.use_pv),
        "use_wind": bool(state.use_wind),
        "use_ess": bool(state.use_ess),
        "use_sop": bool(state.use_sop),
        "use_nop": bool(state.use_nop),
        "reconfiguration_mode": state.reconfiguration_mode,
        "selected_reconfiguration_plan_id": state.selected_reconfiguration_plan_id,
        "available_reconfiguration_plans": getattr(state, "available_reconfiguration_plans", []) or [],
        "reconfiguration_constraints": getattr(state, "reconfiguration_constraints", {}) or {},
        "reward_mode": state.reward_mode,
        "global_pv_multiplier": float(state.global_pv_multiplier),
        "global_load_multiplier": float(state.global_load_multiplier),
        "global_ev_multiplier": float(state.global_ev_multiplier),
        "time_profiles": getattr(state, "time_profiles", {}) or {},
        "node_overrides": getattr(state, "node_overrides", {}) or {},
        "disabled_devices": getattr(state, "disabled_devices", {}) or {},
    })

    if state.rl_hyperparams:
        settings.setdefault("rl_common", {})
        for key in ["learning_rate", "batch_size", "gamma"]:
            if key in state.rl_hyperparams:
                settings["rl_common"][key] = state.rl_hyperparams[key]
        algo_key = str(state.algo_name or "").upper()
        if algo_key:
            settings.setdefault("rl_specific", {}).setdefault(algo_key, {})
            for key, value in state.rl_hyperparams.items():
                if key not in {"learning_rate", "batch_size", "gamma", "total_timesteps"}:
                    settings["rl_specific"][algo_key][key] = value

    if settings_path:
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)


def _as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _hour_from_time(t) -> int:
    value = _as_float(t, 0.0) or 0.0
    if value > 24:
        value = value / 3600.0
    return int(value) % 24


def _profile_lookup(profile: dict, key: str, hour: int, default: float) -> float:
    values = (profile or {}).get(key) or {}
    if isinstance(values, list):
        return _as_float(values[hour], default) if hour < len(values) else default
    if isinstance(values, dict):
        for candidate in (hour, str(hour), f"{hour:02d}"):
            if candidate in values:
                return _as_float(values[candidate], default)
        for range_key, value in values.items():
            text = str(range_key)
            if "-" in text:
                start, end = text.split("-", 1)
                start_hour = int(_as_float(start, -1))
                end_hour = int(_as_float(end, -1))
                if start_hour <= hour <= end_hour:
                    return _as_float(value, default)
    return default


def _profile_average(profile: dict, key: str, default: float) -> float:
    values = []
    for hour in range(24):
        values.append(_profile_lookup(profile, key, hour, default))
    return sum(values) / len(values) if values else default


def _apply_time_or_global_profiles(env, state: ScenarioConfig) -> str:
    profile = getattr(state, "time_profiles", None) or {}
    has_profile = bool(
        profile.get("load_multiplier_by_hour")
        or profile.get("pv_multiplier_by_hour")
        or profile.get("ev_multiplier_by_hour")
    )
    load_default = float(state.global_load_multiplier)
    pv_default = float(state.global_pv_multiplier)
    ev_default = float(state.global_ev_multiplier)

    if has_profile:
        return (
            f"time_profiles active: {profile.get('profile_name') or 'custom'} "
            f"(avg load x{_profile_average(profile, 'load_multiplier_by_hour', load_default):.2f}, "
            f"avg pv x{_profile_average(profile, 'pv_multiplier_by_hour', pv_default):.2f}, "
            f"avg ev x{_profile_average(profile, 'ev_multiplier_by_hour', ev_default):.2f})"
        )

    if state.global_load_multiplier != 1.0 or state.global_pv_multiplier != 1.0:
        return f"global multipliers active: load x{load_default}, pv x{pv_default}"

    return "default multipliers active"


def _normalize_evaluation_targets(state: ScenarioConfig):
    """Normalize state.evaluation_targets to [{algo, steps, label, specific_model_name}]."""
    raw_targets = getattr(state, "evaluation_targets", None) or []
    targets = []
    if isinstance(raw_targets, list):
        for idx, raw in enumerate(raw_targets):
            if isinstance(raw, dict):
                algo = str(raw.get("algo") or raw.get("algo_name") or "").strip()
                steps = _extract_steps(raw.get("steps") or raw.get("target_model_steps"))
                specific = str(raw.get("specific_model_name") or raw.get("model") or "").strip()
                label = str(raw.get("label") or "").strip()
            else:
                algo = str(raw or "").strip()
                steps = None
                specific = ""
                label = ""
            if not algo:
                continue
            if algo.lower() == "baseline":
                algo = "Baseline"
            if steps is None and algo.upper() != "BASELINE":
                steps = _extract_steps(getattr(state, "target_model_steps", None))
                if steps is None:
                    steps = _extract_steps((getattr(state, "rl_hyperparams", {}) or {}).get("total_timesteps"))
            if not label:
                label = algo.upper() if algo.upper() == "BASELINE" else f"{algo.upper()}-{steps}steps" if steps else f"{algo.upper()}-latest"
            targets.append({
                "algo": algo.upper(),
                "steps": steps,
                "specific_model_name": specific,
                "label": label,
            })

    # When there is no batch target, maintain the original single model behavior.
    if not targets:
        algo = str(getattr(state, "algo_name", "Baseline") or "Baseline").upper()
        steps = _extract_steps(getattr(state, "target_model_steps", None))
        specific = str(getattr(state, "specific_model_name", "") or "").strip()
        label = algo if algo == "BASELINE" else f"{algo}-{steps}steps" if steps else f"{algo}-latest"
        targets.append({"algo": algo, "steps": steps, "specific_model_name": specific, "label": label})

    return targets


def _build_comparison_message(comparison_metrics: dict, state: ScenarioConfig = None) -> str:
    """Organize the multi-model evaluation results into Markdown tables to facilitate direct display on the front end."""
    if not comparison_metrics:
        return "⚠️ No comparable evaluation results were generated."

    display_metrics = {
        label: _filter_metrics_for_state(metrics, state) if state is not None else metrics
        for label, metrics in comparison_metrics.items()
    }

    priority = [
        "Total Cost", "Total Objective Value", "Total Penalty Cost", "Grid Purchase Cost", "Generation Cost", "SOP Loss Cost", "ESS Discharge Cost",
        "Voltage Penalty Cost", "OpenDSS Failure Penalty Cost", "EV Unmet Demand Penalty Cost",
        "Power Balance Slack Penalty", "EV Undercharge Penalty", "SOP Capacity Slack Penalty", "NOP Voltage Slack Penalty",
        "Exact Total Grid Loss (kW)", "Voltage Compliance Rate(%)", "Minimum Node Voltage (pu)", "Maximum Node Voltage (pu)",
        "Maximum Voltage Deviation (pu)", "Voltage Range (pu)", "Maximum Line Power Flow (pu)", "Line with Maximum Power Flow",
        "Renewable Energy Absorption Rate (%)", "Renewable Energy Curtailment (kWh)", "EV Charging Satisfaction Rate (%)",
        "Cumulative Environment Reward (Scaled)", "Cumulative Environment Reward (Unscaled)",
    ]
    all_keys = []
    for metrics in display_metrics.values():
        for key in metrics.keys():
            if key not in all_keys and not key.startswith("_"):
                all_keys.append(key)
    selected_keys = [k for k in priority if k in all_keys]
    selected_keys += [k for k in all_keys if k not in selected_keys]

    header = ["Model/Algorithm"] + selected_keys
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for label, metrics in display_metrics.items():
        row = [label] + [_format_metric_cell(metrics.get(k, "-")) for k in selected_keys]
        lines.append("| " + " | ".join(row) + " |")

    best_line = ""
    candidates = []
    for label, metrics in display_metrics.items():
        value = metrics.get("Total Objective Value", metrics.get("Total Cost"))
        if isinstance(value, (int, float)):
            candidates.append((float(value), label))
    if candidates:
        best_objective, best_label = min(candidates, key=lambda x: x[0])
        baseline_candidates = [
            value for value, label in candidates
            if str(label).strip().upper() == "BASELINE"
        ]
        if baseline_candidates and str(best_label).strip().upper() != "BASELINE":
            baseline_objective = baseline_candidates[0]
            if best_objective < baseline_objective - 1e-6:
                best_line = (
                    f"\n\n⚠️ **Objective consistency warning**: Total Objective Value of {best_label} {best_objective:.4f} "
                    f"Lower than Baseline {baseline_objective:.4f}. Baseline is a global optimization baseline,"
                    f"At this time, you should not directly determine the optimal RL. Please check the penalty term and objective function caliber first."
                )
            else:
                best_line = f"\n\n🏆 Calculated according to the lowest **Total Objective Value**, the current optimal is: **{best_label}**, Total Objective Value {best_objective:.4f}."
        else:
            best_line = f"\n\n🏆 Calculated according to the lowest **Total Objective Value**, the current optimal is: **{best_label}**, Total Objective Value {best_objective:.4f}."

    return "✅ **Multi-model/multi-Algorithm evaluation completed!**\n\n📊 **The comparison results are as follows:**\n\n" + "\n".join(
        lines) + best_line + "\n\n💡 All targets are evaluated using the same scene configuration and the same random seed seed=0 to facilitate horizontal comparison."


# Compatibility note: run_lvgs_simulation remains the legacy callable name to avoid breaking imports.
async def run_lvgs_simulation(session_id: str, state: ScenarioConfig, task_states: dict):
    """
    Bridge executor called by the underlying simulation.
    Responsible for mapping Pydantic's ScenarioConfig back to the config global variable that GridSage-back is accustomed to reading, and calling the environment.
    """
    task_states[session_id]["logs"].append("=> Start building and mapping the latest GridSage configuration to the underlying engine...")
    if getattr(state, "active_skill_ids", None):
        task_states[session_id]["logs"].append("=> Active Scenario Skills: " + ", ".join(state.active_skill_ids))

    # 1. Map configuration to original config structure
    try:
        def to_scalar(v):
            if isinstance(v, (list, tuple)):
                return v[0] if v else 0
            return v

        # Correctly map CORE_PARAMS - use exactly the same flat key structure as in config.py
        config.CORE_PARAMS["grid_model"] = state.grid_model
        config.CORE_PARAMS["reward_mode"] = state.reward_mode
        config.CORE_PARAMS["distributed_energy"]["pv"] = state.use_pv
        config.CORE_PARAMS["distributed_energy"]["wind"] = state.use_wind
        config.CORE_PARAMS["distributed_energy"]["ess"] = state.use_ess
        config.CORE_PARAMS["sop_nodes_active"] = state.use_sop
        config.CORE_PARAMS["nop_nodes_active"] = state.use_nop
        config.CORE_PARAMS["reconfiguration_mode"] = state.reconfiguration_mode
        config.CORE_PARAMS["selected_reconfiguration_plan_id"] = state.selected_reconfiguration_plan_id
        config.CORE_PARAMS["available_reconfiguration_plans"] = getattr(state, "available_reconfiguration_plans", []) or []
        config.CORE_PARAMS["reconfiguration_constraints"] = getattr(state, "reconfiguration_constraints", {}) or {}

        # Timing and step size mounting (forced to scalar)
        config.CORE_PARAMS["start_hour"] = int(to_scalar(state.start_hour))
        config.CORE_PARAMS["end_hour"] = int(to_scalar(state.end_hour))
        config.CORE_PARAMS["step_minutes"] = int(to_scalar(state.step_minutes))

        # [FIX 1] Solver field mapping - passes user-selected solver to underlying engine
        config.CORE_PARAMS["solver"] = state.solver
        task_states[session_id]["logs"].append(f"=>Bound solver: {state.solver}")

        # [FIX 3] EV scale scaling factor mapping - write global_ev_multiplier to core parameters (forced to scalar)
        config.CORE_PARAMS["global_load_multiplier"] = float(to_scalar(state.global_load_multiplier))
        config.CORE_PARAMS["global_pv_multiplier"] = float(to_scalar(state.global_pv_multiplier))
        config.CORE_PARAMS["time_profiles"] = getattr(state, "time_profiles", {}) or {}
        config.CORE_PARAMS["node_overrides"] = getattr(state, "node_overrides", {}) or {}
        config.CORE_PARAMS["disabled_devices"] = getattr(state, "disabled_devices", {}) or {}
        config.CORE_PARAMS["global_ev_multiplier"] = float(to_scalar(state.global_ev_multiplier))
        task_states[session_id]["logs"].append(
            f"=>Bound EV scale scaling factor: x{config.CORE_PARAMS['global_ev_multiplier']}")

        # Reinforcement learning hyperparameter mounting
        if state.rl_hyperparams:
            algo_key = state.algo_name.strip().upper()
            if algo_key not in config.RL_HYPERPARAMS.get("algo_specific", {}):
                config.RL_HYPERPARAMS["algo_specific"][algo_key] = {}

            for k, v in state.rl_hyperparams.items():
                if k in ["learning_rate", "batch_size", "gamma"]:
                    config.RL_HYPERPARAMS["common"][k] = v
                else:
                    config.RL_HYPERPARAMS["algo_specific"][algo_key][k] = v
            task_states[session_id]["logs"].append(f"=> Mounted RL custom hyperparameters: {len(state.rl_hyperparams)} items")

        task_states[session_id]["logs"].append(f"=>Number of mounted node mutations: {len(state.node_overrides)}")
        task_states[session_id]["logs"].append(f"=> Number of disabled default device nodes: {len(getattr(state, 'disabled_devices', {}) or {})}")
        _write_gui_settings_for_state(state)
        task_states[session_id]["logs"].append("=> The training/evaluation sub-process scenario configuration gui_settings.json has been written")
    except Exception as map_err:
        task_states[session_id]["logs"].append(f"=> Agent layer parameter mapping failed: {map_err}")

    # 2. Run core evaluation in the thread pool (redirect_stdout must be within the thread, not across threads)
    task_states[session_id]["logs"].append(f"=> Allocating background computing threads and starting {state.algo_name}...")

    try:
        loop = asyncio.get_running_loop()
        real_metrics, captured_logs = await loop.run_in_executor(
            None, _run_core_evaluation_sync, state, session_id, task_states
        )

        # Transfer the print log captured in the thread to the front end
        if captured_logs:
            task_states[session_id]["logs"].extend(captured_logs[-50:])

            # Distinguish between pure training mode and evaluation mode based on the tags in the indicator
            if real_metrics.get("is_train_only", False):
                task_states[session_id]["status"] = "train_completed"
                task_states[session_id]["result"] = {
                    "message": f"Successfully completed training of {state.algo_name} model.",
                    "status": "TrainSuccess"
                }
                task_states[session_id]["logs"].append(f"=> ✔ The training task is over! The simulation evaluation will not be triggered this time.")
            elif real_metrics.get("_error"):
                task_states[session_id]["status"] = "error"
                task_states[session_id]["result"] = {
                    "message": f"Simulation failed to complete: {real_metrics.get('_error')}",
                    "status": "SimulationFailed",
                    "all_metrics": real_metrics,
                }
                task_states[session_id]["logs"].append(f"=> ✘ Simulation failed: {real_metrics.get('_error')}")
            else:
                task_states[session_id]["status"] = "completed"

                if real_metrics.get("is_comparison", False):
                    comparison_metrics = real_metrics.get("comparison_metrics", {})
                    detailed_message = _build_comparison_message(comparison_metrics, state)

                    # Populate the legacy frontend summary field with the first item, and place the complete comparison data in comparison_metrics/all_metrics.
                    first_metrics = next(iter(comparison_metrics.values()), {}) if comparison_metrics else {}
                    task_states[session_id]["result"] = {
                        "message": detailed_message,
                        "status": "ComparisonSuccess",
                        "cost": float(first_metrics.get("Total Cost", 0.0)) if isinstance(first_metrics.get("Total Cost", 0.0),
                                                                                      (int, float)) else 0.0,
                        "voltage_pass_rate": float(first_metrics.get("Voltage Compliance Rate(%)", 100.0)) if isinstance(
                            first_metrics.get("Voltage Compliance Rate(%)", 100.0), (int, float)) else 100.0,
                        "grid_loss_kw": float(first_metrics.get("Exact Total Grid Loss (kW)", 0.0)) if isinstance(
                            first_metrics.get("Exact Total Grid Loss (kW)", 0.0), (int, float)) else 0.0,
                        "comparison_metrics": comparison_metrics,
                        "all_metrics": real_metrics
                    }
                    task_states[session_id]["logs"].append(
                        f"=> ✔ Multi-model evaluation completed! A total of {len(comparison_metrics)} targets were compared.")
                else:
                    # ================= [New] Dynamically splicing all evaluation indicators =================
                    # Traverse the real_metrics dictionary and convert all items into Markdown list format (retaining 4 decimal places)
                    display_metrics = _filter_metrics_for_state(real_metrics, state)
                    metrics_details = "\n".join([
                        f"- **{k}**: {round(v, 4) if isinstance(v, float) else v}"
                        for k, v in display_metrics.items()
                        if k != "is_train_only"
                    ])

                    # Build a new rich version of reply text
                    detailed_message = (
                        f"✅ **The simulation operation ended successfully!**\n\n"
                        f"📊 **The detailed evaluation indicators are as follows:**\n{metrics_details}\n\n"
                        f"💡 *You can continue to use natural language to initiate the next parameter modification.*"
                    )
                    # ===============================================================

                    task_states[session_id]["result"] = {
                        "message": detailed_message, # Pass rich text to the message field of the front end
                        "status": "Success",
                        "cost": float(real_metrics.get("Total Cost", 0.0)),
                        "voltage_pass_rate": float(real_metrics.get("Voltage Compliance Rate(%)", 100.0)),
                        "grid_loss_kw": float(real_metrics.get("Exact Total Grid Loss (kW)", 0.0)),
                        "all_metrics": real_metrics # The complete data is passed to the front end intact, which is convenient for the front end to draw pictures or make tables in the future.
                    }
                    task_states[session_id]["logs"].append(
                        f"=> ✔ The simulation ran successfully! Total Cost: {real_metrics.get('Total Cost', 0.0):.2f}")

        comparison_metrics = real_metrics.get("comparison_metrics", {}) if isinstance(real_metrics, dict) else {}
        skill_interpretation = interpret_results_with_skills(
            getattr(state, "active_skill_ids", []),
            real_metrics,
            state=state,
            comparison_metrics=comparison_metrics,
            task_status=task_states[session_id].get("status", ""),
        )
        if skill_interpretation and task_states[session_id].get("result", {}).get("message"):
            task_states[session_id]["result"]["message"] += skill_interpretation
        log_run_finish(
            session_id=session_id,
            state=state,
            timestamp_start=task_states[session_id].get("timestamp_start"),
            task_state=task_states[session_id],
            metrics=real_metrics,
        )

    except FileNotFoundError as err:
        task_states[session_id]["status"] = "error"
        task_states[session_id]["logs"].append(f"❌ The key weight file is missing: {str(err)}")
        task_states[session_id]["logs"].append("=> Please say to the big model: 'Switch back to Baseline optimizer and rerun'")
        log_run_finish(
            session_id=session_id,
            state=state,
            timestamp_start=task_states[session_id].get("timestamp_start"),
            task_state=task_states[session_id],
            error_message=str(err),
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        with open("error_dump.txt", "w", encoding="utf-8") as err_f:
            err_f.write(tb)
        task_states[session_id]["status"] = "error"
        task_states[session_id]["logs"].append(f"❌ A fatal crash occurred in the experimental execution layer: {str(e)}")
        #Write the complete stack to the log to facilitate precise positioning
        tb_lines = tb.split('\n')
        task_states[session_id]["logs"].extend([f"  {l}" for l in tb_lines if l.strip()][-20:])
        log_run_finish(
            session_id=session_id,
            state=state,
            timestamp_start=task_states[session_id].get("timestamp_start"),
            task_state=task_states[session_id],
            error_message=str(e),
        )


def _run_core_evaluation_sync(state: ScenarioConfig, session_id: str = None, task_states: dict = None):
    """
    Synchronous isolation zone, running in a separate thread. Capture stdout here, returning a (metrics, log_lines) tuple.
    """
    import traceback
    import io
    from contextlib import redirect_stdout, redirect_stderr

    f = io.StringIO()

    with redirect_stdout(f), redirect_stderr(f):
        # --- Path diagnostic log ---
        # Compatibility note: vgridsim_core remains the runtime directory name to avoid breaking imports.
        _CORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vgridsim_core")
        if _CORE_PATH not in sys.path:
            sys.path.insert(0, _CORE_PATH)
        print(f"[GridSage-Engine] GridSage-back core path: {_CORE_PATH}")
        print(f"[GridSage-Engine] path_exists: {os.path.exists(_CORE_PATH)}")

        # --- Verify Excel file ---
        from config import PATHS
        excel_path = PATHS["grid_params_excel"]
        print(f"[GridSage-Engine] Excel path: {excel_path}")
        print(f"[GridSage-Engine] Excel exists: {os.path.exists(excel_path)}")

        if not os.path.exists(excel_path):
            lines = [l for l in f.getvalue().split('\n') if l.strip()]
            raise FileNotFoundError(f"Excel parameter file does not exist: {excel_path}")

        # --- Hard core test module introduces real environment variables ---
        print(f"[GridSage-Engine] Worker Python Exe: {sys.executable}")
        try:
            import openpyxl
            print(f"[GridSage-Engine] Openpyxl loaded from: {openpyxl.__file__}")
        except Exception as oe:
            raise RuntimeError(f"The bottom layer cannot import openpyxl at all! Real error={oe}\n{traceback.format_exc()}")

        try:
            import pandas as pd
            df_test = pd.read_excel(excel_path, sheet_name="EVStation", engine="openpyxl")
            print(f"[GridSage-Engine] EVStation rows: {len(df_test)}")
        except Exception as ev_err:
            tb = traceback.format_exc()
            with open("error_dump.txt", "w", encoding="utf-8") as err_f:
                err_f.write(f"EVStation worksheet reading failed: {ev_err}\n{tb}")
            raise RuntimeError(f"EVStation worksheet read failed: {ev_err}\n{tb}")

        from power_grid_env import PowerGridEnv
        from evaluate_agents import evaluate_baseline, evaluate_rl_agent, discover_and_load_algorithms, \
            plot_and_save_results, plot_voltage_snapshots, plot_line_flow_snapshots_comparison
        from rl_normalization import load_eval_vecnormalize
    from config import PATHS

    # The system uses the two-stage power flow mode (DistFlow + OpenDSS) by default
    use_two_stage = True
    print(f"[GridSage-Engine] Environment building for grid: {state.grid_model}, mode=two_stage")
    env = PowerGridEnv(gui_params=config.CORE_PARAMS, use_two_stage_flow=use_two_stage)
    seed = 0
    env.reset(seed=seed)

    # =========================================================
    # Global Multiplier Control
    # This step is extremely important. Global parameters must be covered before node-level mutation.
    # =========================================================
    profile_mode = _apply_time_or_global_profiles(env, state)
    print(f"[GridSage-Engine] Applying scenario profile mode: {profile_mode}")

    # =========================================================
    # Scene disturbance (NodeOverrides)
    # Node-level load/PV/ESS/EV parking space expansion is rebuilt uniformly by PowerGridEnv.reset().
    # Avoid Baseline, RL reset, and multi-model comparison in different scenarios.
    # =========================================================
    print(f"[GridSage-Engine] Node overrides bound to env reset path: {len(state.node_overrides)} item(s).")

    # =========================================================
    # Algorithm isolation dispatch (Algorithmic Dispatching)
    # =========================================================
    metrics = {}
    time_series_log_for_this_seed = {}

    try:
        # Training remains in single Algorithm mode: if the user says "train SAC for 1000 steps", only state.algo_name will be trained.
        if getattr(state, "execution_mode", "evaluate") == "train":
            algo = state.algo_name.strip().upper()
            if algo == "BASELINE" or algo == "":
                raise ValueError("Baseline is an optimization/baseline solver and does not require training. Please specify PPO/SAC/TD3/DDPG, etc. RL Algorithm.")

            print(f"[GridSage-Engine] The user has explicitly specified to enter training mode and {algo} training is about to start!")
            train_script = os.path.join(_CORE_PATH, f"train_{algo.lower()}.py")
            if not os.path.exists(train_script):
                raise FileNotFoundError(f"The corresponding training script cannot be found: {train_script}")

            train_steps = state.rl_hyperparams.get("total_timesteps", 1000)
            print(f"[GridSage-Engine] is training in the background... (Set the number of steps: {train_steps})")
            import subprocess
            import time
            env_copy = _prepare_training_subprocess_env(train_steps)
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            train_started_at = time.time()

            # Use Popen for streaming reading and enter the progress into the front end in real time
            process = subprocess.Popen(
                [sys.executable, train_script],
                env=env_copy,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=_CORE_PATH,
                creationflags=creationflags
            )

            captured_out = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    clean_line = _clean_process_log_line(line)
                    captured_out.append(clean_line)
                    if session_id and task_states and clean_line:
                        task_states[session_id]["logs"].append(clean_line)
                        if len(task_states[session_id]["logs"]) > 100:
                            task_states[session_id]["logs"] = task_states[session_id]["logs"][-100:]

            returncode = process.wait()
            if returncode != 0:
                saved_model = _find_recent_trained_model_after(
                    config.PATHS.get("models_dir", ""), algo, train_steps, train_started_at
                )
                if saved_model:
                    print(
                        f"[GridSage-Engine] training model has been saved, but the child process exit code is {returncode};"
                        f"Processed as successful. Model: {saved_model}"
                    )
                    returncode = 0
            if returncode != 0:
                raise RuntimeError(f"Training failed!\nSTDOUT:\n{chr(10).join(captured_out[-50:])}")

            print(f"[GridSage-Engine] {algo} training completed! Since it is in pure training mode, this task is over.")
            metrics = {"is_train_only": True}
            return metrics, [l for l in f.getvalue().split('\n') if l.strip()] + captured_out[-20:]

        # Evaluation supports single model and batch comparison of evaluation_targets.
        targets = _normalize_evaluation_targets(state)
        is_comparison_request = bool(getattr(state, "evaluation_targets", None)) or len(targets) > 1
        # Baseline may rewrite the grid object; try to put it into the last evaluation to avoid affecting the evaluation of the RL policy in the same scenario.
        targets = [t for t in targets if t["algo"] != "BASELINE"] + [t for t in targets if t["algo"] == "BASELINE"]

        print(f"[GridSage-Engine] Evaluation targets: {targets}")
        needs_rl_models = any(str(t.get("algo", "")).strip().upper() not in {"", "BASELINE"} for t in targets)
        model_classes = discover_and_load_algorithms() if needs_rl_models else {}
        models_dir = PATHS.get("models_dir", "")
        comparison_metrics = {}
        current_scenario_fingerprint = build_scenario_fingerprint(_state_to_scenario_settings(state))

        for target in targets:
            target_algo = str(target.get("algo", "")).strip().upper()
            target_steps = _extract_steps(target.get("steps"))
            target_specific_model = str(target.get("specific_model_name", "") or "").strip()
            label = str(target.get("label") or target_algo).strip()

            if target_algo == "BASELINE" or target_algo == "":
                print(f"[GridSage-Engine] Running Baseline for comparison label={label}...")
                env.reset(seed=seed)
                target_metrics, ts_data = evaluate_baseline(config.CORE_PARAMS, seed, env.stations_list, env.grid,
                                                            use_two_stage=use_two_stage)
                comparison_metrics[label] = target_metrics
                time_series_log_for_this_seed[label] = ts_data
                continue

            print(
                f"[GridSage-Engine] Preparing RL inference environment for {label} ({target_algo}, steps={target_steps or 'latest'})...")
            model_path = _find_requested_model(
                models_dir,
                target_algo,
                target_steps,
                target_specific_model,
                scenario_fingerprint=current_scenario_fingerprint,
                require_scenario_match=True,
            )

            if model_path:
                print(
                    f"[GridSage-Engine] Target={label}; requested steps={target_steps if target_steps is not None else 'latest'}")
                print(f"[GridSage-Engine] Target={label}; selected model path={model_path}")

            if not model_path:
                requested = f"{target_steps} steps" if target_steps is not None else "current scene"
                raise FileNotFoundError(
                    f"The current scenario has not trained a {target_algo} model that can be used for evaluation ({requested})."
                    f"The system has blocked the direct application of old scene models and will not automatically train in evaluation mode."
                    f"Please switch the execution mode to train and train {target_algo} first, and then evaluate after the training is completed."
                )

            model_class = next((v for k, v in model_classes.items() if k.upper() == target_algo), None)
            if not model_class:
                raise ValueError(f"Unsupported or inactive RL Algorithm: {target_algo}")

            manifest = _assert_model_matches_current_scenario(model_path, state, target_algo, label)

            print(f"[GridSage-Engine] Executing Stage 1 & 2 via RL Policy Network for {label}...")
            eval_normalizer = load_eval_vecnormalize(
                env,
                os.path.dirname(model_path),
                training_config=config.TRAINING_CONFIG,
                manifest=manifest,
            )
            model_env = eval_normalizer if eval_normalizer is not None else env
            model = model_class.load(model_path, env=model_env)
            target_metrics, ts_data = evaluate_rl_agent(
                model,
                env,
                seed,
                obs_normalizer=eval_normalizer,
            )
            target_metrics["_model_path"] = model_path
            target_metrics["_algo"] = target_algo
            target_metrics["_steps"] = target_steps if target_steps is not None else "latest"
            target_metrics["_scenario_fingerprint"] = manifest.get("scenario_fingerprint")
            comparison_metrics[label] = target_metrics
            time_series_log_for_this_seed[label] = ts_data

        if not comparison_metrics:
            raise RuntimeError("No models were successfully evaluated.")

        if is_comparison_request:
            metrics = {"is_comparison": True, "comparison_metrics": comparison_metrics}
        else:
            metrics = next(iter(comparison_metrics.values()))
    except Exception as exec_err:
        raise RuntimeError(f"{str(exec_err)}\n\n[SOLVER & STDOUT LOGS]:\n{f.getvalue()}") from exec_err

    print(f"[GridSage-Engine] Generating result charts in backend...")
    if time_series_log_for_this_seed and any(v for v in time_series_log_for_this_seed.values() if v):
        try:
            plot_and_save_results(time_series_log_for_this_seed, seed, config.CORE_PARAMS)
            plot_voltage_snapshots(time_series_log_for_this_seed, seed, config.CORE_PARAMS)
            plot_line_flow_snapshots_comparison(time_series_log_for_this_seed, seed, config.CORE_PARAMS)
        except Exception as plot_err:
            print(f"[WARNING] Chart generation failed: {plot_err}")

    print(f"[GridSage-Engine] Aggregating results. Total mapped cost: {metrics.get('Total Cost', 0):.2f}")
    return metrics, [l for l in f.getvalue().split('\n') if l.strip()]
