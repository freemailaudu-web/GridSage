from typing import Tuple

from .schema import ScenarioConfig


def _valid_buses(grid_model: str):
    try:
        import os
        import sys

        core_path = os.path.join(os.path.dirname(__file__), "vgridsim_core")
        if core_path not in sys.path:
            sys.path.insert(0, core_path)
        from grid_model import create_grid

        test_grid = create_grid(model=grid_model, gui_params=None)
        if hasattr(test_grid, "BusNames"):
            return set(test_grid.BusNames)
        if hasattr(test_grid, "_bnames"):
            return set(test_grid._bnames)
        return set(bus.ID for bus in test_grid.Buses)
    except Exception:
        fallback_sizes = {"ieee33": 33, "ieee69": 69, "ieee123": 123}
        return {f"b{i}" for i in range(1, fallback_sizes.get(grid_model, 0) + 1)}


def validate_scenario(scenario: ScenarioConfig) -> Tuple[bool, str]:
    """General physics-aware validation before VGridSim execution."""

    valid_grids = {"ieee33", "ieee69", "ieee123"}
    if scenario.grid_model not in valid_grids:
        return False, f"Unsupported grid_model '{scenario.grid_model}'. Use one of {sorted(valid_grids)}."

    if not (0 <= int(scenario.start_hour) <= 23):
        return False, "start_hour must be between 0 and 23."
    if not (1 <= int(scenario.end_hour) <= 24):
        return False, "end_hour must be between 1 and 24."
    if int(scenario.end_hour) <= int(scenario.start_hour):
        return False, "end_hour must be greater than start_hour."
    if int(scenario.step_minutes) not in {5, 10, 15, 30, 60}:
        return False, "step_minutes must be one of 5, 10, 15, 30, 60."

    valid_algos = {"Baseline", "PPO", "SAC", "TD3", "DDPG"}
    if scenario.algo_name not in valid_algos:
        return False, f"Unsupported algo_name '{scenario.algo_name}'. Use one of {sorted(valid_algos)}."
    if scenario.execution_mode not in {"train", "evaluate"}:
        return False, "execution_mode must be 'train' or 'evaluate'."
    if scenario.execution_mode == "train" and scenario.algo_name == "Baseline":
        return False, "Baseline does not support training. Use PPO, SAC, TD3, or DDPG."
    if scenario.reconfiguration_mode not in {"none", "radial_reconfiguration"}:
        return False, "reconfiguration_mode must be 'none' or 'radial_reconfiguration'."
    plan_id = str(scenario.selected_reconfiguration_plan_id or "").strip()
    if not plan_id:
        return False, "selected_reconfiguration_plan_id cannot be empty. Use R0 for the original topology."
    constraints = scenario.reconfiguration_constraints or {}
    if int(constraints.get("max_switch_operations", 1)) > 2:
        return False, "First-version NOP reconfiguration allows at most one NOP close and one base-line open."
    if bool(constraints.get("allow_multi_nop", False)):
        return False, "First-version NOP reconfiguration does not support multiple NOPs at the same time."

    total_timesteps = scenario.rl_hyperparams.get("total_timesteps")
    if total_timesteps is not None and int(total_timesteps) <= 0:
        return False, "rl_hyperparams.total_timesteps must be greater than 0."

    multiplier_errors = []
    if scenario.global_pv_multiplier < 0:
        multiplier_errors.append("global_pv_multiplier cannot be negative.")
    if scenario.global_load_multiplier < 0:
        multiplier_errors.append("global_load_multiplier cannot be negative.")
    if scenario.global_ev_multiplier < 0:
        multiplier_errors.append("global_ev_multiplier cannot be negative.")
    if scenario.global_pv_multiplier > 3.0:
        multiplier_errors.append("global_pv_multiplier is above the safety boundary 3.0.")
    if scenario.global_load_multiplier > 3.0:
        multiplier_errors.append("global_load_multiplier is above the safety boundary 3.0.")
    if scenario.global_ev_multiplier > 5.0:
        multiplier_errors.append("global_ev_multiplier is above the safety boundary 5.0.")
    if multiplier_errors:
        return False, " ".join(multiplier_errors)

    for profile_key, upper_bound in {
        "load_multiplier_by_hour": 3.0,
        "pv_multiplier_by_hour": 3.0,
        "ev_multiplier_by_hour": 5.0,
    }.items():
        values = (scenario.time_profiles or {}).get(profile_key)
        if values is None:
            continue
        if isinstance(values, list):
            iterable = enumerate(values)
        elif isinstance(values, dict):
            iterable = values.items()
        else:
            return False, f"time_profiles.{profile_key} must be a list or object."
        for hour_key, raw_value in iterable:
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                return False, f"time_profiles.{profile_key}[{hour_key}] must be numeric."
            if value < 0:
                return False, f"time_profiles.{profile_key}[{hour_key}] cannot be negative."
            if value > upper_bound:
                return False, f"time_profiles.{profile_key}[{hour_key}] exceeds safety boundary {upper_bound}."

    try:
        valid_buses = _valid_buses(scenario.grid_model)
    except Exception as exc:
        return False, f"Could not load grid model '{scenario.grid_model}': {exc}"

    allowed_node_keys = {
        "add_pv_kw",
        "add_wind_kw",
        "add_load_kw",
        "add_ev_spots",
        "add_ess_kwh",
        "add_ess_power_kw",
        "add_ess_c_rate",
    }
    for node_id, params in scenario.node_overrides.items():
        if node_id not in valid_buses:
            preview = sorted(valid_buses)[:12]
            return False, f"Node '{node_id}' does not exist in {scenario.grid_model}. Example valid nodes: {preview}."
        for key, raw_value in params.items():
            if key not in allowed_node_keys:
                return False, f"Unsupported node override '{key}'. Use {sorted(allowed_node_keys)}."
            value = float(raw_value)
            if value < 0:
                return False, f"Node '{node_id}' parameter '{key}' cannot be negative."
            if key == "add_ev_spots" and int(value) != value:
                return False, f"Node '{node_id}' add_ev_spots must be a non-negative integer."
            if key in {"add_pv_kw", "add_wind_kw", "add_load_kw"} and value > 2000:
                return False, f"Node '{node_id}' {key}={value} exceeds the current 2000 kW safety boundary."
            if key == "add_ess_power_kw" and value > 2000:
                return False, f"Node '{node_id}' add_ess_power_kw={value} exceeds the current 2000 kW safety boundary."
            if key == "add_ess_c_rate" and value > 4:
                return False, f"Node '{node_id}' add_ess_c_rate={value} exceeds the current 4C safety boundary."
            if key == "add_ev_spots" and value > 200:
                return False, f"Node '{node_id}' add_ev_spots={value} exceeds the current 200 spots safety boundary."

    allowed_device_types = {"generator", "pv", "wind", "ess", "ev_station", "sop", "nop"}
    for node_id, devices in scenario.disabled_devices.items():
        if node_id not in valid_buses:
            preview = sorted(valid_buses)[:12]
            return False, f"Node '{node_id}' does not exist in {scenario.grid_model}. Example valid nodes: {preview}."
        if not isinstance(devices, dict):
            return False, f"disabled_devices['{node_id}'] must be an object."
        for device_type, ids in devices.items():
            if device_type not in allowed_device_types:
                return False, f"Unsupported disabled device type '{device_type}'. Use {sorted(allowed_device_types)}."
            if not isinstance(ids, list) or not all(isinstance(item, str) and item.strip() for item in ids):
                return False, f"disabled_devices['{node_id}']['{device_type}'] must be a list of device ids or ['*']."

    return True, "Scenario validation passed."
