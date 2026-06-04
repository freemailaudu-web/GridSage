import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
from power_grid_env import PowerGridEnv
from baseline import solve_baseline
from grid_model import create_grid, load_electricity_price, load_station_info
from reconfiguration import apply_reconfiguration_plan
import traceback
import matplotlib as mpl
from visualization import plot_ev_spot_powers, plot_voltage_snapshots, plot_line_flow_snapshots_comparison
from fpowerkit.soldss import OpenDSSSolver
from fpowerkit.solbase import GridSolveResult
from two_stage_powerflow import fix_bus_voltage_limits, update_grid_from_model
from fpowerkit import Generator
from copy import deepcopy
from config import CORE_PARAMS, EVALUATION_CONFIG, PATHS, TRAINING_CONFIG
import importlib.util
import glob
from rl_normalization import load_eval_vecnormalize, predict_with_optional_normalization, unwrap_power_grid_env
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import numpy as np
import re

def _iter_numeric_values(data):
    if data is None:
        return
    if isinstance(data, dict):
        for value in data.values():
            yield from _iter_numeric_values(value)
        return
    if isinstance(data, (list, tuple, np.ndarray)):
        for value in data:
            yield from _iter_numeric_values(value)
        return
    try:
        value = float(data)
    except (TypeError, ValueError):
        return
    if np.isfinite(value):
        yield value


def _add_voltage_metrics(metrics, voltages_data):
    values = list(_iter_numeric_values(voltages_data))
    if not values:
        metrics.setdefault("Minimum Node Voltage (pu)", np.nan)
        metrics.setdefault("Maximum Node Voltage (pu)", np.nan)
        metrics.setdefault("Maximum Voltage Deviation (pu)", np.nan)
        return
    metrics["Minimum Node Voltage (pu)"] = min(values)
    metrics["Maximum Node Voltage (pu)"] = max(values)
    metrics["Maximum Voltage Deviation (pu)"] = max(abs(v - 1.0) for v in values)
    metrics["Voltage Range (pu)"] = max(values) - min(values)


def _series_by_id(series_data, item_id, index, length=0):
    if isinstance(series_data, dict):
        return list(series_data.get(item_id, []))
    arr = np.asarray(series_data) if series_data is not None else np.asarray([])
    if arr.ndim == 2 and arr.shape[1] > index:
        return list(arr[:, index])
    if length > 0:
        return [0.0] * length
    return []


def _add_pv_curtailment_metrics(metrics, grid, pvw_dispatch_data, gui_params, pvw_available_data=None):
    pvw_devices = list(grid.PVWinds) if hasattr(grid, "PVWinds") else []
    if not pvw_devices:
        metrics["Renewable Energy Absorption Rate (%)"] = 100.0
        metrics["Renewable Energy Curtailment (kWh)"] = 0.0
        return

    step_minutes = gui_params["step_minutes"]
    step_h = step_minutes / 60.0
    step_seconds = step_minutes * 60
    fallback_steps = int((gui_params["end_hour"] - gui_params["start_hour"]) * (60 // step_minutes))
    sb_kw = grid.SB * 1000

    renewable_actual = 0.0
    renewable_potential = 0.0
    renewable_curtail = 0.0
    for idx, pvw in enumerate(pvw_devices):
        actual_series = _series_by_id(pvw_dispatch_data, pvw.ID, idx, fallback_steps)
        steps = len(actual_series) if actual_series else fallback_steps
        if not actual_series:
            actual_series = [0.0] * steps
        potential_series = _series_by_id(pvw_available_data, pvw.ID, idx, steps) if pvw_available_data else []
        if not potential_series:
            potential_series = [
                (pvw.P(t * step_seconds) if callable(pvw.P) else pvw.P)
                for t in range(steps)
            ]
        if len(actual_series) < steps:
            actual_series = list(actual_series) + [0.0] * (steps - len(actual_series))

        for actual, potential in zip(actual_series[:steps], potential_series[:steps]):
            potential_pu = max(0.0, float(potential))
            actual_pu = min(max(0.0, float(actual)), potential_pu)
            renewable_actual += actual_pu * sb_kw * step_h
            renewable_potential += potential_pu * sb_kw * step_h
            renewable_curtail += max(0.0, potential_pu - actual_pu) * sb_kw * step_h

    metrics["Renewable Energy Absorption Rate (%)"] = (
        renewable_actual / renewable_potential * 100.0
    ) if renewable_potential > 1e-9 else 100.0
    metrics["Renewable Energy Curtailment (kWh)"] = renewable_curtail


def _add_line_loading_metrics(metrics, grid, line_powers_data):
    max_flow = 0.0
    max_line_id = ""
    for line_id, series in (line_powers_data or {}).items():
        for value in _iter_numeric_values(series):
            flow = abs(value)
            if flow > max_flow:
                max_flow = flow
                max_line_id = line_id
    metrics["Maximum Line Power Flow (pu)"] = max_flow
    if max_line_id:
        metrics["Line with Maximum Power Flow"] = max_line_id


def _add_operational_metrics(metrics, grid, gui_params, voltages_data, line_powers_data, pvw_dispatch_data,
                             pvw_available_data=None):
    _add_voltage_metrics(metrics, voltages_data)
    _add_line_loading_metrics(metrics, grid, line_powers_data)
    _add_pv_curtailment_metrics(metrics, grid, pvw_dispatch_data, gui_params, pvw_available_data)


def calc_station_operator_step_metrics(ev_power_kw, price_t, step_minutes, ev_params, station_cfg, info):
    import numpy as np

    ev_power_kw = np.asarray(ev_power_kw, dtype=float)
    charge_kw = np.clip(ev_power_kw, 0.0, None)
    discharge_kw = np.clip(-ev_power_kw, 0.0, None)

    step_h = step_minutes / 60.0
    pi_uc = float(station_cfg.get("charge_service_price", 1.20))
    pi_ud = float(station_cfg.get("v2g_subsidy_price", 0.80))
    eta_c = max(ev_params.charge_efficiency, 1e-6)
    eta_d = max(ev_params.discharge_efficiency, 1e-6)

    charge_profit = float(np.sum((pi_uc - price_t) * charge_kw * eta_c * step_h))
    discharge_profit = float(np.sum((price_t - pi_ud) * discharge_kw / eta_d * step_h))
    gross_profit = charge_profit + discharge_profit

    extra_cost = 0.0
    if station_cfg.get("include_grid_cost", False):
        extra_cost += info.get("grid_purchase_cost", 0.0)
    if station_cfg.get("include_generation_cost", True):
        extra_cost += info.get("generation_cost", 0.0)
    if station_cfg.get("include_ess_cost", True):
        extra_cost += info.get("ess_discharge_cost", 0.0)
    if station_cfg.get("include_sop_loss_cost", True):
        extra_cost += info.get("sop_loss_cost", 0.0)

    penalty_cost = 0.0
    if station_cfg.get("include_penalty_cost", True):
        penalty_cost += -info.get("voltage_penalty_unscaled", 0.0)
        penalty_cost += -info.get("opendss_failure_penalty_unscaled", 0.0)
        penalty_cost += -info.get("ev_shortage_penalty_unscaled", 0.0)

    net_profit = gross_profit - extra_cost - penalty_cost

    return {
        "charge_service_profit": charge_profit,
        "v2g_spread_profit": discharge_profit,
        "station_gross_profit": gross_profit,
        "station_extra_cost": extra_cost,
        "station_penalty_cost": penalty_cost,
        "station_net_profit": net_profit,
    }
# This function is used to discover and load all Algorithm plug-ins
def discover_and_load_algorithms():
    """
    Scan custom_algorithms and stable_baselines3 directories to dynamically load all model classes.
    """
    model_class_registry = {}

    # 1. Load the built-in Algorithm of stable-baselines3 (as a basis)
    from stable_baselines3 import PPO, DDPG, SAC, TD3
    model_class_registry.update({"PPO": PPO, "DDPG": DDPG, "SAC": SAC, "TD3": TD3})

    # 2. Scan all python files in the custom_algorithms directory
    plugin_dir = "custom_algorithms"
    plugin_files = glob.glob(os.path.join(plugin_dir, "*.py"))

    for plugin_file in plugin_files:
        module_name = os.path.basename(plugin_file)[:-3]
        try:
            # Dynamically import plug-in modules
            spec = importlib.util.spec_from_file_location(module_name, plugin_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check and call the registered function
            if hasattr(module, 'register_algorithm'):
                registration_info = module.register_algorithm()
                algo_name = registration_info['name']
                algo_class = registration_info['class']
                model_class_registry[algo_name] = algo_class
                print(f" [Plugin Loader] successfully loaded custom Algorithm: '{algo_name}' from {plugin_file}")
            else:
                print(f" [Plugin Loader] Warning: File {plugin_file} is not a valid plugin (missing register_algorithm function).")
        except Exception as e:
            print(f" [Plugin Loader] Error: Loading plugin {plugin_file} failed: {e}")

    return model_class_registry

def run_baseline_stage_two(grid: object, baseline_data: object, gui_params: object) -> object:
    """
    - Correction: Before calling print_power_audit, load the AUDIT_TIMESTEP state for the grid object.
    """
    AUDIT_TIMESTEP = 5
    print("\n" + "=" * 20 + " Baseline Stage 2: OpenDSS precise calculation begins " + "=" * 17)
    sb_mva = grid.SB
    step_minutes = gui_params['step_minutes']
    step_duration_h = step_minutes / 60.0
    price = load_electricity_price(gui_params=gui_params)
    time_steps = int((gui_params['end_hour'] - gui_params['start_hour']) * (60 // step_minutes))
    source_bus_id = gui_params.get('slack_bus', 'b1')

    VIRTUAL_GEN_ID = 'gen_for_slack_bus'
    slack_generator = grid.Gen(VIRTUAL_GEN_ID) if VIRTUAL_GEN_ID in grid.GenNames else None
    if not slack_generator:
        slack_generator = Generator(id=VIRTUAL_GEN_ID, busid=source_bus_id, pmax_pu=9999, pmin_pu=-9999, qmax_pu=9999,
                                    qmin_pu=-9999, costA=0, costB=0, costC=0)
        grid.AddGen(slack_generator)

    sop_gen_map = {}
    if hasattr(grid, 'SOPs'):
        for sop_id, sop in grid.SOPs.items():
            gen1_id = f"gen_for_{sop_id}_bus1";
            gen2_id = f"gen_for_{sop_id}_bus2"
            if gen1_id in grid.GenNames: grid.DelGen(gen1_id)
            if gen2_id in grid.GenNames: grid.DelGen(gen2_id)
            gen1 = Generator(id=gen1_id, busid=sop.Bus1, pmax_pu=9999, pmin_pu=-9999, qmax_pu=9999, qmin_pu=-9999,
                             costA=0, costB=0, costC=0)
            gen2 = Generator(id=gen2_id, busid=sop.Bus2, pmax_pu=9999, pmin_pu=-9999, qmax_pu=9999, qmin_pu=-9999,
                             costA=0, costB=0, costC=0)
            grid.AddGen(gen1);
            grid.AddGen(gen2)
            sop_gen_map[sop_id] = (gen1, gen2)

    voltages_stage1_data = baseline_data.get('bus_voltages', {})
    total_generation_cost = baseline_data.get('generation_cost', 0)
    total_ess_discharge_cost = baseline_data.get('ess_discharge_cost', 0)
    fix_bus_voltage_limits(grid)
    opendss_solver = OpenDSSSolver(grid, source_bus=source_bus_id)

    spot_to_bus_map = dict(baseline_data.get("spot_to_bus_map", {}))
    if not spot_to_bus_map:
        spot_counter = 0
        stations_info = load_station_info()
        # The actual bus name in the current power grid
        valid_buses = set(getattr(grid, "_bnames", [b.ID for b in grid.Buses]))

        for i, info in enumerate(stations_info):
            bus_id = str(info['Bus_ID'])
            station_id = info.get('Station_ID', f"ST_{i}")

            for j in range(info['Num_Spots']):
                global_spot_idx = spot_counter + j

                # Only when the bus exists in the current network frame, the mapping will be established
                if bus_id in valid_buses:
                    spot_to_bus_map[global_spot_idx] = bus_id
                else:
                    # You can leave a prompt here for easy viewing on the console.
                    print(f"Warning: The bus {bus_id} of the charging station {station_id} is not in the current distribution network, skip the mapping of these parking spaces.")

            # Note: Whether there is mapping or not, spot_counter must be accumulated to keep the index consistent with the Baseline
            spot_counter += info['Num_Spots']


    # Before calling the audit function, prepare the power grid status specifically for AUDIT_TIMESTEP
    if AUDIT_TIMESTEP is not None and AUDIT_TIMESTEP < time_steps:
        # 1. Update the status of all regular devices at t=AUDIT_TIMESTEP
        update_grid_from_model(grid, baseline_data, AUDIT_TIMESTEP)

        # 2. Update the status of SOP virtual generator at t=AUDIT_TIMESTEP
        sop_flows_data = baseline_data.get('sop_flows', {})
        for sop_id, flow_data in sop_flows_data.items():
            if sop_id in sop_gen_map:
                gen1, gen2 = sop_gen_map[sop_id]
                p_transfer = flow_data['P1'][AUDIT_TIMESTEP]
                q1_injection = flow_data['Q1'][AUDIT_TIMESTEP]
                q2_injection = flow_data.get('Q2', flow_data['Q1'])[AUDIT_TIMESTEP]

                loss_transfer = flow_data['Loss'][AUDIT_TIMESTEP]
                gen1._p, gen1._q = -p_transfer, q1_injection
                gen2._p, gen2._q = p_transfer - loss_transfer, q2_injection # The power at the injection end must be subtracted from the loss




    # --- Main simulation loop starts ---
    voltages_stage2_log = []
    line_powers_stage2_log = []
    step_costs_log = []
    total_precise_grid_cost = 0.0
    total_loss_W = 0.0
    stage2_inflow_p_log_pu = []

    for t in range(time_steps):
        # Inside the loop, update the status for the current time step t
        update_grid_from_model(grid, baseline_data, t)

        sop_flows_data = baseline_data.get('sop_flows', {})
        for sop_id, flow_data in sop_flows_data.items():
            if sop_id in sop_gen_map:
                gen1, gen2 = sop_gen_map[sop_id]
                p_transfer = flow_data['P1'][t]
                q1_injection = flow_data['Q1'][t]
                q2_injection = flow_data.get('Q2', flow_data['Q1'])[t]
                # ▼▼▼ Core correction points ▼▼▼
                loss_transfer = flow_data['Loss'][t]
                gen1._p, gen1._q = -p_transfer, q1_injection
                gen2._p, gen2._q = p_transfer - loss_transfer, q2_injection # The power at the injection end must be subtracted from the loss
                # ▲▲▲ Correction completed ▲▲▲

        spot_powers_data = baseline_data.get('spot_powers', {})
        bus_ev_load_pu = {b.ID: 0.0 for b in grid.Buses}
        for spot_id_num, power_list in spot_powers_data.items():
            # As long as the power is not 0 (no matter positive or negative), process it
            if t < len(power_list) and abs(power_list[t]) > 1e-6:
                bus_id = spot_to_bus_map.get(spot_id_num)
                # It is required to have a bus_id, and it is also required that this bus_id is really a bus of the current network frame.
                if bus_id and bus_id in bus_ev_load_pu:
                    # Charge (positive) and V2G (negative) will be accumulated
                    bus_ev_load_pu[bus_id] += power_list[t]

        original_load_funcs = {}
        try:
            for bus_id, ev_load in bus_ev_load_pu.items():
                # This way negative ev_load (V2G) can also be applied
                if abs(ev_load) > 1e-6:
                    bus = grid.Bus(bus_id)
                    original_func = bus.Pd
                    original_load_funcs[bus_id] = original_func
                    # V2G (negative value) will reduce Pd, charging (positive value) will increase Pd
                    bus.Pd = lambda time, _of=original_func, _el=ev_load: _of(time) + _el
            time_in_seconds = t * step_minutes * 60
            opendss_result, opendss_value = opendss_solver.solve(time_in_seconds)
        finally:
            for bus_id, original_func in original_load_funcs.items():
                grid.Bus(bus_id).Pd = original_func

        step_cost_t = 0
        if opendss_result == GridSolveResult.OK:
            total_loss_W += opendss_value
            voltages_stage2_log.append({bus.ID: bus.V for bus in grid.Buses if bus.V is not None})
            line_powers_stage2_log.append({line.ID: line.P for line in grid.Lines if line.P is not None})
            precise_inflow_pu = slack_generator.P if slack_generator.P is not None else 0
            stage2_inflow_p_log_pu.append(precise_inflow_pu)
            precise_inflow_pu = max(0.0, precise_inflow_pu) # Disable negative power from being included in Grid Purchase Cost
            precise_grid_cost_step = price[t] * precise_inflow_pu * sb_mva * 1000 * step_duration_h
            total_precise_grid_cost += precise_grid_cost_step
            step_cost_t += precise_grid_cost_step
        else:
            print(f" - Warning: OpenDSS solution failed at time step {t}")
            stage2_inflow_p_log_pu.append(np.nan)
            voltages_stage2_log.append({})
            line_powers_stage2_log.append({})
            step_cost_t = np.nan
        step_costs_log.append(step_cost_t)

    stage1_inflow_p_log_pu = baseline_data.get('grid_inflow_p', [0] * time_steps)


    metrics = {
        "Grid Purchase Cost": total_precise_grid_cost,
        "Generation Cost": total_generation_cost,
        "SOP Loss Cost": baseline_data.get('sop_loss_cost', 0),
        "ESS Discharge Cost": total_ess_discharge_cost,
        "Exact Total Grid Loss (kW)": total_loss_W / 1000.0
    }
    if baseline_data.get("solver_note"):
        metrics["solver note"] = baseline_data["solver_note"]
    metrics["Total Cost"] = sum(
        m for k, m in metrics.items()
        if 'Grid Loss' not in k and isinstance(m, (int, float))
    )
    metrics["Power Balance Slack Penalty"] = baseline_data.get('slack_penalty', 0)
    metrics["EV Undercharge Penalty"] = baseline_data.get('ev_not_full_penalty', 0)
    metrics["SOP Capacity Slack Penalty"] = baseline_data.get('sop_slack_penalty', 0)
    metrics["NOP Voltage Slack Penalty"] = baseline_data.get('nop_slack_penalty', 0)
    metrics["Total Penalty Cost"] = (
        metrics["Power Balance Slack Penalty"]
        + metrics["EV Undercharge Penalty"]
        + metrics["SOP Capacity Slack Penalty"]
        + metrics["NOP Voltage Slack Penalty"]
    )

    stage1_total_obj = baseline_data.get('objective_value', 0)
    stage1_purchase_cost = baseline_data.get('grid_purchase_cost', 0)
    final_hybrid_obj_value = (stage1_total_obj - stage1_purchase_cost) + total_precise_grid_cost
    metrics["Total Objective Value"] = final_hybrid_obj_value

    standards = EVALUATION_CONFIG["standards"]
    violations = sum(1 for v_dict in voltages_stage2_log for v in v_dict.values()
                     if not (standards["voltage_min_pu"] <= v <= standards["voltage_max_pu"]))
    total_checks = sum(len(v_dict) for v_dict in voltages_stage2_log)
    metrics["Voltage Compliance Rate(%)"] = (1 - violations / total_checks) * 100 if total_checks > 0 else 100.0

    satisfied_count = baseline_data.get('charged_ev_count', 0)
    total_count = baseline_data.get('total_ev_count', 0)
    metrics["EV Charging Satisfaction Rate (%)"] = (satisfied_count / total_count) * 100 if total_count > 0 else 100


    def format_log_data(log):
        data_final = {}
        if log:
            all_keys = set().union(*(d.keys() for d in log if d))
            for key in all_keys: data_final[key] = [step.get(key, np.nan) for step in log]
        return data_final

    time_series_data = {
        "voltages_data_stage1": voltages_stage1_data,
        "voltages_data_stage2": format_log_data(voltages_stage2_log),
        "line_powers_data_stage1": baseline_data.get('line_powers', {}),
        "line_powers_data_stage2": format_log_data(line_powers_stage2_log),
        "step_costs": step_costs_log,
        "raw_baseline_data": baseline_data,
    }
    _add_operational_metrics(
        metrics,
        grid,
        gui_params,
        time_series_data["voltages_data_stage2"],
        time_series_data["line_powers_data_stage2"],
        baseline_data.get("pvw_powers", {}),
        baseline_data.get("pvw_available_powers", {}),
    )
    return metrics, time_series_data

def evaluate_rl_agent(model, env, seed, obs_normalizer=None):
    """
    Evaluates a single RL agent and returns normalized metrics and timing data.
    - Fixed data logging and return structures to ensure functions always return correctly.
    """
    base_env = unwrap_power_grid_env(env)
    obs, info = env.reset(seed=seed)
    base_env = unwrap_power_grid_env(env)
    standards = EVALUATION_CONFIG["standards"]
    metrics = {
        "Grid Purchase Cost": 0.0,
        "Generation Cost": 0.0,
        "SOP Loss Cost": 0.0,
        "ESS Discharge Cost": 0.0,
        "Power Balance Slack Penalty": 0.0,
        "SOP Capacity Slack Penalty": 0.0,
        "NOP Voltage Slack Penalty": 0.0,
        "EV Undercharge Penalty": 0.0,
        "ev_urgency_penalty": 0.0,
        "EV Urgency Penalty Cost": 0.0,
        "Voltage Penalty Cost": 0.0,
        "OpenDSS Failure Penalty Cost": 0.0,
        "EV Unmet Demand Penalty Cost": 0.0,
        "Cumulative Environment Reward (Scaled)": 0.0,
        "Cumulative Environment Reward (Unscaled)": 0.0,
    }

    reward_mode = base_env.params.get("reward_mode", "grid_operator")
    station_cfg = base_env.params.get("station_operator", {})

    if reward_mode == "station_operator":
        metrics.update({
            "Charging Service Spread Profit": 0.0,
            "V2G Spread Profit": 0.0,
            "Operator Gross Revenue": 0.0,
            "Operator Additional Cost": 0.0,
            "Operator Penalty Cost": 0.0,
            "Operator Net Revenue": 0.0,
        })

    # Initialize all loggers
    voltages_stage1_log, voltages_stage2_log = [], []
    line_powers_stage1_log, line_powers_stage2_log = [], []
    total_loss_kW = 0.0
    pvw_pu_log, ess_soc_log, spot_kw_log = [], [], []
    step_costs_log = []
    sop_p_log, sop_q1_log, sop_q2_log, nop_status_log = [], [], [], []
    total_episode_reward = 0.0

    terminated = False
    while not terminated:
        current_ess_soc = [ess.SOC * 100 for ess in base_env.ess_list] if base_env.ess_list else []
        ess_soc_log.append(current_ess_soc)

        action, _ = predict_with_optional_normalization(
            model, obs, obs_normalizer=obs_normalizer, deterministic=True
        )
        action = np.asarray(action)
        if action.ndim > 1:
            action = action[0]

        action_idx = base_env.total_spots + len(base_env.ess_list) + len(base_env.pvw_list)
        sop_p_action = action[action_idx: action_idx + len(base_env.sop_list)]
        action_idx += len(base_env.sop_list)
        sop_q1_action = action[action_idx: action_idx + len(base_env.sop_list)]
        action_idx += len(base_env.sop_list)
        sop_q2_action = action[action_idx: action_idx + len(base_env.sop_list)]
        action_idx += len(base_env.sop_list)
        nop_action = action[action_idx: action_idx + len(base_env.nop_list)]

        sop_p_log.append([a * sop.PMax for a, sop in zip(sop_p_action, base_env.sop_list)])
        sop_q1_log.append([a * sop.QMax for a, sop in zip(sop_q1_action, base_env.sop_list)])
        sop_q2_log.append([a * sop.QMax for a, sop in zip(sop_q2_action, base_env.sop_list)])
        nop_status_log.append(np.round(np.clip(nop_action, 0, 1)).astype(int))

        obs, reward, terminated, truncated, info = env.step(action)

        total_episode_reward += reward
        metrics["Cumulative Environment Reward (Scaled)"] += reward
        metrics["Cumulative Environment Reward (Unscaled)"] += info.get("reward_unscaled", reward)
        metrics["Voltage Penalty Cost"] += max(0.0, -float(info.get("voltage_penalty_unscaled", 0.0) or 0.0))
        metrics["OpenDSS Failure Penalty Cost"] += max(
            0.0,
            -float(info.get("opendss_failure_penalty_unscaled", 0.0) or 0.0)
        )
        ev_shortage_cost = max(0.0, -float(info.get("ev_shortage_penalty_unscaled", 0.0) or 0.0))
        metrics["EV Unmet Demand Penalty Cost"] += ev_shortage_cost
        metrics["EV Undercharge Penalty"] += ev_shortage_cost
        
        urgency_penalty = float(info.get("ev_urgency_penalty", 0.0) or 0.0)
        metrics["ev_urgency_penalty"] += urgency_penalty
        metrics["EV Urgency Penalty Cost"] += max(0.0, -urgency_penalty)
        metrics["Power Balance Slack Penalty"] += float(info.get("slack_penalty", 0.0) or 0.0)
        metrics["SOP Capacity Slack Penalty"] += float(info.get("sop_slack_penalty", 0.0) or 0.0)
        metrics["NOP Voltage Slack Penalty"] += float(info.get("nop_slack_penalty", 0.0) or 0.0)

        grid_cost = info.get('grid_purchase_cost', 0)
        gen_cost = info.get('generation_cost', 0)
        sop_cost = info.get('sop_loss_cost', 0)
        ess_cost = info.get('ess_discharge_cost', 0)

        metrics["Grid Purchase Cost"] += grid_cost
        metrics["Generation Cost"] += gen_cost
        metrics["SOP Loss Cost"] += sop_cost
        metrics["ESS Discharge Cost"] += ess_cost
        physical_step_cost = grid_cost + gen_cost + sop_cost + ess_cost
        step_costs_log.append(physical_step_cost)

        if reward_mode == "station_operator":
            step_station = {}
            if "station_net_profit" in info:
                step_station = {
                    "charge_service_profit": info.get("charge_service_profit", 0.0),
                    "v2g_spread_profit": info.get("v2g_spread_profit", 0.0),
                    "station_gross_profit": info.get("station_gross_profit", 0.0),
                    "station_extra_cost": info.get("station_extra_cost", 0.0),
                    "station_penalty_cost": info.get("station_penalty_cost", 0.0),
                    "station_net_profit": info.get("station_net_profit", 0.0),
                }
            else:
                price_t = base_env.price[base_env.current_step - 1]
                step_station = calc_station_operator_step_metrics(
                    ev_power_kw=info.get("ev_power_kw", []),
                    price_t=price_t,
                    step_minutes=base_env.params["step_minutes"],
                    ev_params=base_env.ev_params,
                    station_cfg=station_cfg,
                    info=info
                )

            metrics["Charging Service Spread Profit"] += step_station["charge_service_profit"]
            metrics["V2G Spread Profit"] += step_station["v2g_spread_profit"]
            metrics["Operator Gross Revenue"] += step_station["station_gross_profit"]
            metrics["Operator Additional Cost"] += step_station["station_extra_cost"]
            metrics["Operator Penalty Cost"] += step_station["station_penalty_cost"]
            metrics["Operator Net Revenue"] += step_station["station_net_profit"]

        # Record the voltage and power flow in the two stages respectively
        voltages_stage1_log.append(info.get('voltages_stage1', {}))
        voltages_stage2_log.append(info.get('voltages_stage2', {}))
        line_powers_stage1_log.append(info.get('line_powers_stage1', {}))
        line_powers_stage2_log.append(info.get('line_powers_stage2', {}))

        # Accumulate accurate Grid Loss
        total_loss_kW += info.get('opendss_loss_W', 0.0) / 1000.0

        pvw_pu_log.append(info.get('pvw_power_pu', [0] * len(base_env.pvw_list)))
        spot_kw_log.append(info.get('ev_power_kw', [0] * base_env.total_spots))

    final_ess_soc = [ess.SOC * 100 for ess in base_env.ess_list] if base_env.ess_list else []
    ess_soc_log.append(final_ess_soc)

    metrics["Total Cost"] = sum(metrics[k] for k in ["Grid Purchase Cost", "Generation Cost", "SOP Loss Cost", "ESS Discharge Cost"])
    metrics["Total Penalty Cost"] = sum(
        metrics[k] for k in [
            "Power Balance Slack Penalty", "SOP Capacity Slack Penalty", "NOP Voltage Slack Penalty",
            "EV Undercharge Penalty", "EV Urgency Penalty Cost", "Voltage Penalty Cost", "OpenDSS Failure Penalty Cost",
        ]
    )
    if reward_mode == "station_operator":
        metrics["Total Objective Value"] = metrics["Operator Net Revenue"]
    else:
        metrics["Total Objective Value"] = -metrics["Cumulative Environment Reward (Unscaled)"]
    metrics["Exact Total Grid Loss (kW)"] = total_loss_kW


    # --- EV Charging Satisfaction Rate calculation ---
    session_refs = []
    spot_offset = 0
    start_hour = base_env.params['start_hour']
    end_hour = base_env.params['end_hour']
    steps_per_hour = 60 // base_env.params['step_minutes']
    for station in base_env.stations_list:
        for session in station.daily_sessions:
            if not (session.arrival_hour < end_hour and session.departure_hour > start_hour):
                continue

            effective_arrival_hour = max(session.arrival_hour, start_hour)
            effective_departure_hour = min(session.departure_hour, end_hour)
            arrival_step = int(round((effective_arrival_hour - start_hour) * steps_per_hour))
            departure_step = int(round((effective_departure_hour - start_hour) * steps_per_hour))
            arrival_step = max(0, min(arrival_step, base_env.total_timesteps - 1))
            departure_step = max(0, min(departure_step, base_env.total_timesteps))
            if departure_step <= arrival_step:
                continue

            session_refs.append((spot_offset + session.spot_id, departure_step))
        spot_offset += station.num_spots
    total_evs_in_scenario = len(session_refs)
    ev_satisfaction_count = 0
    if total_evs_in_scenario > 0:
        for global_spot_id, departure_step in session_refs:
            final_soc = base_env.ev_boc[global_spot_id, departure_step]
            if final_soc >= standards["ev_charged_soc_threshold"]:
                ev_satisfaction_count += 1
        metrics["EV Charging Satisfaction Rate (%)"] = (ev_satisfaction_count / total_evs_in_scenario) * 100
    else:
        metrics["EV Charging Satisfaction Rate (%)"] = 100.0

    # --- Voltage Compliance Rate calculation ---
    violations = sum(1 for v_dict in voltages_stage2_log for v in v_dict.values()
                     if not (standards["voltage_min_pu"] <= v <= standards["voltage_max_pu"])) # <-- Modify
    total_checks = sum(len(v_dict) for v_dict in voltages_stage2_log)
    metrics["Voltage Compliance Rate(%)"] = (1 - violations / total_checks) * 100 if total_checks > 0 else 100.0

    # --- Data formatting ---
    def format_log_data(log):
        data_final = {}
        if log:
            all_keys = set().union(*(d.keys() for d in log if d))
            for key in all_keys:
                data_final[key] = [step.get(key, np.nan) for step in log]
        return data_final

    voltages_s1_final = format_log_data(voltages_stage1_log)
    voltages_s2_final = format_log_data(voltages_stage2_log)
    line_powers_s1_final = format_log_data(line_powers_stage1_log)
    line_powers_s2_final = format_log_data(line_powers_stage2_log)

    # --- Construct the final returned data packet ---
    time_series_data = {
        "voltages_data_stage1": voltages_s1_final,
        "voltages_data_stage2": voltages_s2_final,
        "line_powers_data_stage1": line_powers_s1_final,
        "line_powers_data_stage2": line_powers_s2_final,
        "step_costs": step_costs_log,
        "pvw_pu": np.array(pvw_pu_log),
        "ess_soc_percent": np.array(ess_soc_log),
        "spot_kw": np.array(spot_kw_log),
        "sop_p_pu": np.array(sop_p_log),
        "sop_q_pu": np.array(sop_q1_log),
        "sop_q1_pu": np.array(sop_q1_log),
        "sop_q2_pu": np.array(sop_q2_log),
        "nop_status": np.array(nop_status_log),
    }
    _add_operational_metrics(
        metrics,
        base_env.grid,
        base_env.params,
        time_series_data["voltages_data_stage2"],
        time_series_data["line_powers_data_stage2"],
        time_series_data["pvw_pu"],
    )

    return metrics, time_series_data

def plot_sop_flows(all_ts_data, seed, gui_params):
    """
    Create side-by-side comparison graphs for the SOP's active power flow (P) and reactive power flow (Q).
    """
    # Find the number and ID of SOPs in the scene
    baseline_sops = all_ts_data.get('Baseline', {}).get('raw_baseline_data', {}).get('sop_flows', {})
    if not baseline_sops:
        print(f"Scenario (Seed: {seed}) has no SOP data, skip SOP flow drawing.")
        return
    sop_ids = list(baseline_sops.keys())
    num_sops = len(sop_ids)

    # Create timeline
    start_hour, end_hour = gui_params['start_hour'], gui_params['end_hour']
    # Check if there is 'P1' data and get the number of steps
    if not sop_ids or 'P1' not in baseline_sops[sop_ids[0]] or not baseline_sops[sop_ids[0]]['P1']:
        print(f"Scenario (Seed: {seed}) SOP data is incomplete, skip SOP flow drawing.")
        return
    num_steps = len(baseline_sops[sop_ids[0]]['P1'])
    time_axis = np.linspace(start_hour, end_hour, num_steps)

    # Create a row for each SOP, each row contains two subgraphs P and Q
    fig, axes = plt.subplots(num_sops, 2, figsize=(18, 5 * num_sops), sharex=True, squeeze=False)
    fig.suptitle(f'Comparison of active/reactive power scheduling of each AlgorithmSOP (scenario Seed: {seed})', fontsize=18, fontproperties="SimHei")

    for i, sop_id in enumerate(sop_ids):
        ax_p = axes[i, 0] # The graph on the left is used to draw the active power P
        ax_q = axes[i, 1] # The graph on the right is used to draw reactive power Q

        # --- Draw active power P --- in the left subgraph
        ax_p.set_title(f'SOP ID: {sop_id} - Active power (P)', fontproperties="SimHei")
        ax_p.set_ylabel('Active power P (pu)', fontproperties="SimHei")
        # Draw the P curve of Baseline
        ax_p.plot(time_axis, baseline_sops[sop_id]['P1'], label=f'Baseline P1', color='blue', linestyle='-')
        # Draw the P curve of RLAlgorithm
        for algo_name, ts_data in all_ts_data.items():
            if algo_name == 'Baseline' or 'sop_p_pu' not in ts_data:
                continue
            rl_sop_p_flows = ts_data['sop_p_pu']
            if rl_sop_p_flows.ndim == 2 and rl_sop_p_flows.shape[1] > i:
                ax_p.plot(time_axis, rl_sop_p_flows[:, i], label=f'{algo_name} P1', linestyle='--')
        ax_p.legend()
        ax_p.grid(True, linestyle=':')
        ax_p.axhline(0, color='black', linewidth=0.5)

        # --- Draw reactive power Q in the right subgraph ---
        ax_q.set_title(f'SOP ID: {sop_id} - Reactive power (Q)', fontproperties="SimHei")
        ax_q.set_ylabel('Reactive power Q (pu)', fontproperties="SimHei")
        # Draw the Q curve of Baseline
        ax_q.plot(time_axis, baseline_sops[sop_id]['Q1'], label=f'Baseline Q1', color='green', linestyle='-')
        if 'Q2' in baseline_sops[sop_id]:
            ax_q.plot(time_axis, baseline_sops[sop_id]['Q2'], label=f'Baseline Q2', color='purple', linestyle='-')
        # Draw the Q curve of RLAlgorithm
        for algo_name, ts_data in all_ts_data.items():
            if algo_name == 'Baseline':
                continue
            rl_sop_q1_flows = ts_data.get('sop_q1_pu', ts_data.get('sop_q_pu', np.array([])))
            rl_sop_q2_flows = ts_data.get('sop_q2_pu', np.array([]))
            if rl_sop_q1_flows.ndim == 2 and rl_sop_q1_flows.shape[1] > i:
                ax_q.plot(time_axis, rl_sop_q1_flows[:, i], label=f'{algo_name} Q1', linestyle='--')
            if rl_sop_q2_flows.ndim == 2 and rl_sop_q2_flows.shape[1] > i:
                ax_q.plot(time_axis, rl_sop_q2_flows[:, i], label=f'{algo_name} Q2', linestyle=':')
        ax_q.legend()
        ax_q.grid(True, linestyle=':')
        ax_q.axhline(0, color='black', linewidth=0.5)

    # Set X-axis labels uniformly
    axes[-1, 0].set_xlabel('Time (hours)', fontproperties="SimHei")
    axes[-1, 1].set_xlabel('Time (hours)', fontproperties="SimHei")

    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'sop_flows_PQ_seed_{seed}.png')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path)
    plt.close()
    print(f"[SOP enhanced version] The SOP active/reactive power comparison chart of the scenario (Seed: {seed}) has been saved to: {save_path}")


def plot_nop_status(all_ts_data, seed, gui_params):
    """Create a comparison diagram for the switching status of the NOP (0-open, 1-closed)."""
    baseline_nops = all_ts_data.get('Baseline', {}).get('raw_baseline_data', {}).get('nop_status', {})
    if not baseline_nops:
        print(f"Scene (Seed: {seed}) has no NOP data, skip NOP status drawing.")
        return
    nop_ids = list(baseline_nops.keys())
    num_nops = len(nop_ids)

    start_hour, end_hour = gui_params['start_hour'], gui_params['end_hour']
    num_steps = len(baseline_nops[nop_ids[0]])
    time_axis = np.linspace(start_hour, end_hour, num_steps)

    fig, axes = plt.subplots(num_nops, 1, figsize=(12, 3 * num_nops), sharex=True, squeeze=False)
    fig.suptitle(f'Comparison of switch status of each AlgorithmNOP (scenario Seed: {seed})', fontsize=16, fontproperties="SimHei")

    for i, nop_id in enumerate(nop_ids):
        ax = axes[i, 0]
        # Use step plot to draw the switch status, which is more intuitive
        ax.step(time_axis, baseline_nops[nop_id], where='post', label='Baseline', color='blue')

        for algo_name, ts_data in all_ts_data.items():
            if algo_name == 'Baseline' or 'nop_status' not in ts_data:
                continue

            rl_nop_status = ts_data['nop_status']
            if rl_nop_status.shape[1] > i:
                ax.step(time_axis, rl_nop_status[:, i], where='post', label=algo_name, linestyle='--')

        ax.set_title(f'NOP ID: {nop_id}', fontproperties="SimHei")
        ax.set_ylabel('switch status', fontproperties="SimHei")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['open', 'closed'])
        ax.legend()
        ax.grid(True, axis='x', linestyle=':')

    axes[-1, 0].set_xlabel('Time (hours)', fontproperties="SimHei")

    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'nop_status_seed_{seed}.png')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path)
    plt.close()
    print(f"The NOP status comparison chart of scene (Seed: {seed}) has been saved to: {save_path}")


def print_baseline_status_monitor(data, grid, gui_params):
    """Print the detailed status of each step for the Baseline running results"""
    # Normally this function is not needed and is used for debugging.
    print("\n" + "=" * 25 + " Baseline running status monitoring " + "=" * 25)

    # --- Initialization parameters ---
    sb_mva_kw = grid.SB * 1000
    step_seconds = gui_params['step_minutes'] * 60
    num_steps = len(data.get('grid_inflow_p', []))

    if num_steps == 0:
        print("No valid time series data available for monitoring.")
        print("=" * 75)
        return

    # --- Print monitoring information in a loop according to time steps ---
    for t in range(num_steps):
        print(f"\n---------- Step {t} Status Monitoring (Baseline) ----------")

        # --- 1. Demand side (no change) ---
        base_load_kw = sum(bus.Pd(t * step_seconds) for bus in grid.Buses) * sb_mva_kw
        ev_charge_kw = sum(powers[t] * sb_mva_kw for powers in data.get('spot_powers', {}).values() if powers[t] > 0)
        ess_charge_kw = sum(powers[t] * sb_mva_kw for powers in data.get('ess_powers', {}).values() if powers[t] > 0)
        total_demand_kw = base_load_kw + ev_charge_kw + ess_charge_kw
        print(
            f"[Demand side] Total demand: {total_demand_kw:.2f} kW (base load: {base_load_kw:.2f}, EV charging: {ev_charge_kw:.2f}, ESS charging: {ess_charge_kw:.2f})")

        # --- 2. Power supply side (the calculation here does not change, it is regarded as the actual supply) ---
        pvw_devices = list(grid.PVWinds) if hasattr(grid, 'PVWinds') else []
        pvw_powers_data = data.get('pvw_powers', {})
        pv_gen_kw = sum(pvw_powers_data[dev.ID][t] * sb_mva_kw for dev in pvw_devices if
                        dev.Tag == 'pv' and dev.ID in pvw_powers_data)
        wind_gen_kw = sum(pvw_powers_data[dev.ID][t] * sb_mva_kw for dev in pvw_devices if
                          dev.Tag == 'wind' and dev.ID in pvw_powers_data)
        ess_discharge_kw = -sum(
            powers[t] * sb_mva_kw for powers in data.get('ess_powers', {}).values() if powers[t] < 0)
        ev_discharge_kw = -sum(
            powers[t] * sb_mva_kw for powers in data.get('spot_powers', {}).values() if powers[t] < 0)
        total_free_supply_kw = pv_gen_kw + wind_gen_kw + ess_discharge_kw + ev_discharge_kw

        grid_supply_kw = data['grid_inflow_p'][t] * sb_mva_kw
        gen_supply_kw = sum(powers[t] * sb_mva_kw for powers in data.get('gen_powers', {}).values())
        total_paid_supply_kw = grid_supply_kw + gen_supply_kw

        total_supply_kw = total_free_supply_kw + total_paid_supply_kw
        slack_supply_kw = sum(powers[t] * sb_mva_kw for powers in data.get('slack_powers', {}).values())

        print(f"[Power supply side](actual supply) total: {total_supply_kw + slack_supply_kw:.2f} kW")
        print(f" ├─ Free supply: {total_free_supply_kw:.2f} kW")
        print(f" │ ├─ PV output: {pv_gen_kw:.2f} kW")
        print(f" │ ├─ Wind output: {wind_gen_kw:.2f} kW")
        print(f" │ ├─ ESS discharge: {ess_discharge_kw:.2f} kW")
        print(f"    │   └─ V2G: {ev_discharge_kw:.2f} kW")
        print(
            f" └─ Paid supply: {total_paid_supply_kw:.2f} kW (grid purchase: {grid_supply_kw:.2f}, local generation: {gen_supply_kw:.2f})")
        if abs(slack_supply_kw) > 0.01:
            print(f" ⚠️ Warning: There is a power gap in the system! Slack Power: {slack_supply_kw:.2f} kW")

        # --- 3. Network conditioning equipment (no change) ---
        sop_flows_data = data.get('sop_flows', {})
        nop_flows_data = data.get('nop_flows', {})
        total_sop_transfer_kw = sum(abs(flow['P1'][t]) * sb_mva_kw for flow in sop_flows_data.values())
        total_nop_transfer_kw = sum(abs(flow['P'][t]) * sb_mva_kw for flow in nop_flows_data.values())

        if total_sop_transfer_kw > 0.01 or total_nop_transfer_kw > 0.01:
            print(f"[Network Adjustment]")
            if total_sop_transfer_kw > 0.01:
                sop_details = "; ".join(
                    [f"{sop_id}: {flow['P1'][t] * sb_mva_kw:.2f} kW" for sop_id, flow in sop_flows_data.items()])
                print(f" ├─ SOP transmission active power: {sop_details}")
            if total_nop_transfer_kw > 0.01:
                nop_details = "; ".join(
                    [f"{nop_id}: {flow['P'][t] * sb_mva_kw:.2f} kW" for nop_id, flow in nop_flows_data.items()])
                print(f" └─ NOP transmission active power: {nop_details}")

        # Renewable energy utilization and abandonment analysis
        time_in_seconds = t * step_seconds

        # Calculate the theoretical maximum output
        total_potential_pv_kw = sum(dev.P(time_in_seconds) * sb_mva_kw for dev in pvw_devices if dev.Tag == 'pv')
        total_potential_wind_kw = sum(dev.P(time_in_seconds) * sb_mva_kw for dev in pvw_devices if dev.Tag == 'wind')

        # Calculate light and air abandonment (theoretical maximum - actual output)
        curtailed_pv_kw = total_potential_pv_kw - pv_gen_kw
        curtailed_wind_kw = total_potential_wind_kw - wind_gen_kw

        # Only print this part when light or wind abandonment occurs
        if curtailed_pv_kw > 0.01 or curtailed_wind_kw > 0.01:
            print(f"[Renewable energy utilization]")
            if curtailed_pv_kw > 0.01:
                util_rate = (pv_gen_kw / total_potential_pv_kw * 100) if total_potential_pv_kw > 0 else 100
                print(f" ├─ Photovoltaic consumption: {pv_gen_kw:.2f} / {total_potential_pv_kw:.2f} kW (utilization rate: {util_rate:.1f}%)")
                print(f" │ └─ Curtailed optical power: {curtailed_pv_kw:.2f} kW")
            if curtailed_wind_kw > 0.01:
                util_rate = (wind_gen_kw / total_potential_wind_kw * 100) if total_potential_wind_kw > 0 else 100
                print(
                    f" └─ Wind power consumption: {wind_gen_kw:.2f} / {total_potential_wind_kw:.2f} kW (utilization rate: {util_rate:.1f}%)")
                print(f" └─ Curtailed wind power: {curtailed_wind_kw:.2f} kW")

        print("--------------------------------------------------")

    print("\n" + "=" * 75 + "\n")


def print_sop_monitor(data):
    """
    A special status monitor that prints SOP for Baseline running results.
    """
    print("\n" + "=" * 25 + " SOP running status monitoring " + "=" * 26)

    sop_flows = data.get('sop_flows')
    sop_slacks = data.get('sop_slacks')

    if not sop_flows:
        print("Report: No running data for SOP found in model results.")
        print("=" * 75)
        return

    num_steps = 0
    # Determine the total number of steps through the data of any SOP
    if sop_flows:
        first_sop_id = next(iter(sop_flows))
        num_steps = len(sop_flows[first_sop_id]['P1'])

    if num_steps == 0:
        print("Report: SOP data is empty and cannot be monitored.")
        print("=" * 75)
        return

    print(f"{'Time':<6}{'SOP_ID':<8}{'P1(pu)':<12}{'Q1(pu)':<12}{'Q2(pu)':<12}{'Loss(pu)':<12}{'Cap_Slack':<12}")
    print("-" * 75)

    for t in range(num_steps):
        for sop_id, flows in sop_flows.items():
            p1 = flows['P1'][t]
            q1 = flows['Q1'][t]
            q2 = flows.get('Q2', [0] * num_steps)[t]
            loss = flows['Loss'][t]
            slack = sop_slacks.get(sop_id, [0] * num_steps)[t]

            # Only print when there is any sign of activity in the SOP to avoid screen swiping
            if abs(p1) > 1e-4 or abs(q1) > 1e-4 or abs(q2) > 1e-4 or abs(loss) > 1e-4 or abs(slack) > 1e-4:
                print(f"{t:<6}{sop_id:<8}{p1:<12.4f}{q1:<12.4f}{q2:<12.4f}{loss:<12.4f}{slack:<12.4f}")

    print("=" * 75 + "\n")


def evaluate_baseline(gui_params, seed, stations_list, grid, use_two_stage=False):
    """
    - In two_stage mode, temporarily adds a virtual generator to the slack node for accurate power flow calculations.
    - Ensures that a fully compatible data structure with the RL Agent is returned in all cases.
    """
    VIRTUAL_GEN_ID = 'gen_for_slack_bus'
    slack_bus_id = gui_params.get('slack_bus', 'b1')

    # Temporarily added virtual generator object
    slack_generator = None

    # --- Phase 1: Always run Linear DistFlow optimization first ---
    print("\n--- Running Baseline Stage 1 (Linear DistFlow Optimization) ---")
    selected_plan_request = str(gui_params.get("selected_reconfiguration_plan_id", "R0") or "R0").strip().lower()
    if selected_plan_request in {"auto", "best", "enumerate"}:
        print("[Reconfiguration] Enumerating legal radial plans for Baseline...")
        best = None
        plans = getattr(grid, "available_reconfiguration_plans", []) or []
        for plan in plans:
            if not (plan.get("is_connected") and plan.get("is_radial")):
                continue
            candidate_params = dict(gui_params)
            candidate_params["selected_reconfiguration_plan_id"] = plan["plan_id"]
            candidate_grid = deepcopy(grid)
            try:
                apply_reconfiguration_plan(candidate_grid, candidate_params)
                result_i, data_i, model_i = solve_baseline(candidate_grid, stations_list, candidate_params)
            except Exception as exc:
                print(f"[Reconfiguration] Plan {plan['plan_id']} failed: {exc}")
                continue
            if not data_i:
                continue
            objective_i = data_i.get("objective_value", float("inf"))
            print(f"[Reconfiguration] Plan {plan['plan_id']} objective={objective_i}")
            if best is None or objective_i < best[0]:
                best = (objective_i, result_i, data_i, model_i, candidate_grid, candidate_params)
        if best is None:
            print("!!!!!! Baseline reconfiguration enumeration failed, no feasible plan found !!!!!!")
            result, stage_one_data, _ = GridSolveResult.Failed, {}, None
        else:
            _, result, stage_one_data, _, grid, gui_params = best
            print(
                f"[Reconfiguration] Baseline selected plan "
                f"{stage_one_data.get('reconfiguration_plan', {}).get('plan_id', 'R0')}"
            )
    else:
        result, stage_one_data, _ = solve_baseline(grid, stations_list, gui_params)

    if not stage_one_data:
        print("!!!!!! Baseline Stage 1 solution failed, cannot continue !!!!!!")
        metrics = {
            "Total Cost": 0.0,
            "Grid Purchase Cost": 0.0,
            "Generation Cost": 0.0,
            "SOP Loss Cost": 0.0,
            "ESS Discharge Cost": 0.0,
            "Voltage Compliance Rate(%)": 0.0,
            "EV Charging Satisfaction Rate (%)": 0.0,
            "_error": "Baseline Stage 1 solution failed",
        }
        return metrics, {}

    if use_two_stage:
        metrics, time_series_data = run_baseline_stage_two(grid, stage_one_data, gui_params)
        reconf = stage_one_data.get("reconfiguration_plan", {}) or {}
        metrics["Reconstruction Plan ID"] = reconf.get("plan_id", "R0")
        metrics["closed NOP"] = reconf.get("close_nop_id") or "none"
        metrics["Open branch"] = reconf.get("open_line_id") or "None"
        metrics["topology radial"] = bool(reconf.get("is_radial", True))
        metrics["topology connected"] = bool(reconf.get("is_connected", True))

        #Extract data from the results of the first stage to remain compatible with the data structure of RL Agent and single-stage mode, so that the drawing function can work properly.
        time_series_data["pvw_pu"] = np.array(list(stage_one_data.get('pvw_powers', {}).values())).T
        time_series_data["ess_soc_percent"] = np.array(list(stage_one_data.get('ess_soc', {}).values())).T * 100
        time_series_data["spot_kw"] = np.array(list(stage_one_data.get('spot_powers', {}).values())).T * (
                    grid.SB * 1000)

        # Print the total amount of electricity purchased in the second stage 2. EV does not support the battery aging model, nor does it support V2G scheduling logic based on user wishes. V2G is only called for the lowest cost of the power grid.
        total_stage2_inflow_kwh = (metrics.get('Grid Purchase Cost', 0) / load_electricity_price(gui_params=gui_params)[
            0]) if load_electricity_price(gui_params=gui_params) else 0
        # More accurate calculation method
        if time_series_data and 'raw_baseline_data' in time_series_data:
            total_stage2_inflow_kwh = sum(
                p * (gui_params['step_minutes'] / 60.0) * grid.SB * 1000
                for p in time_series_data['raw_baseline_data'].get('grid_inflow_p_stage2', []) # Assume that the stage2 results exist here
            ) if 'grid_inflow_p_stage2' in time_series_data['raw_baseline_data'] else metrics.get('Grid Purchase Cost', 0) / (
                        sum(load_electricity_price(gui_params=gui_params)) / len(
                    load_electricity_price(gui_params=gui_params)))



        return metrics, time_series_data
    else:
        # The logic of single-stage mode remains unchanged
        print("--- Baseline evaluation mode: Single-Stage, building complete data package ---")
        metrics = {
                "Grid Purchase Cost": stage_one_data.get('grid_purchase_cost', 0),
                "Generation Cost": stage_one_data.get('generation_cost', 0),
                "SOP Loss Cost": stage_one_data.get('sop_loss_cost', 0),
                "ESS Discharge Cost": stage_one_data.get('ess_discharge_cost', 0),
        }
        if stage_one_data.get("solver_note"):
            metrics["solver note"] = stage_one_data["solver_note"]
        metrics["Total Cost"] = sum(v for v in metrics.values() if isinstance(v, (int, float)))
        metrics["Power Balance Slack Penalty"] = stage_one_data.get('slack_penalty', 0)
        metrics["EV Undercharge Penalty"] = stage_one_data.get('ev_not_full_penalty', 0)
        metrics["SOP Capacity Slack Penalty"] = stage_one_data.get('sop_slack_penalty', 0)
        metrics["NOP Voltage Slack Penalty"] = stage_one_data.get('nop_slack_penalty', 0)
        metrics["Total Penalty Cost"] = (
            metrics["Power Balance Slack Penalty"]
            + metrics["EV Undercharge Penalty"]
            + metrics["SOP Capacity Slack Penalty"]
            + metrics["NOP Voltage Slack Penalty"]
        )
        #Add the complete objective function value including all penalties to metrics
        metrics["Total Objective Value"] = stage_one_data.get('objective_value', metrics["Total Cost"])
        reconf = stage_one_data.get("reconfiguration_plan", {}) or {}
        metrics["Reconstruction Plan ID"] = reconf.get("plan_id", "R0")
        metrics["closed NOP"] = reconf.get("close_nop_id") or "none"
        metrics["Open branch"] = reconf.get("open_line_id") or "None"
        metrics["topology radial"] = bool(reconf.get("is_radial", True))
        metrics["topology connected"] = bool(reconf.get("is_connected", True))
        violations = 0
        total_checks = sum(len(v) for v in stage_one_data.get('bus_voltages', {}).values())
        if total_checks > 0:
            for voltages in stage_one_data.get('bus_voltages', {}).values():
                standards = EVALUATION_CONFIG["standards"]
                for v in voltages:
                    if not (standards["voltage_min_pu"] <= v <= standards["voltage_max_pu"]):
                        violations += 1
            metrics["Voltage Compliance Rate(%)"] = (1 - violations / total_checks) * 100
        else:
            metrics["Voltage Compliance Rate(%)"] = 100.0

        satisfied_count = stage_one_data.get('charged_ev_count', 0)
        total_count = stage_one_data.get('total_ev_count', 0)
        metrics["EV Charging Satisfaction Rate (%)"] = (satisfied_count / total_count) * 100 if total_count > 0 else 100

        num_steps = len(stage_one_data.get('grid_inflow_p', []))
        step_costs = [0] * num_steps
        if num_steps > 0:
            price = load_electricity_price(gui_params=gui_params)
            sb_mva = grid.SB
            step_h = gui_params['step_minutes'] / 60.0
            for t in range(num_steps):
                grid_cost_t = price[t] * stage_one_data.get('grid_inflow_p', [0] * num_steps)[
                    t] * sb_mva * 1000 * step_h
                gen_cost_t = 0
                for gen_id, p_list in stage_one_data.get('gen_powers', {}).items():
                    gen = grid.Gen(gen_id)
                    if gen:
                        p_mw = p_list[t] * sb_mva
                        gen_cost_t += (gen.CostA(t) * p_mw ** 2 + gen.CostB(t) * p_mw + gen.CostC(t)) * step_h
                sop_loss_cost_t = sum(
                    price[t] * flow_data['Loss'][t] * sb_mva * 1000 * step_h
                    for sop_id, flow_data in stage_one_data.get('sop_flows', {}).items()
                )
                step_costs[t] = grid_cost_t + gen_cost_t + sop_loss_cost_t

        voltages_data = stage_one_data.get("bus_voltages", {})

        line_powers_data = stage_one_data.get("line_powers", {})
        time_series_data = {
                "step_costs": step_costs,
                "voltages_data_stage1": voltages_data,
                "voltages_data_stage2": voltages_data,
                # Copy power flow data into two new keys to match what the plot function expects
                "line_powers_data_stage1": line_powers_data,
                "line_powers_data_stage2": line_powers_data,
                "raw_baseline_data": stage_one_data,
                "pvw_pu": np.array(list(stage_one_data.get('pvw_powers', {}).values())).T,
                "ess_soc_percent": np.array(list(stage_one_data.get('ess_soc', {}).values())).T * 100,
                "spot_kw": np.array(list(stage_one_data.get('spot_powers', {}).values())).T * (
                        grid.SB * 1000)
        }
        _add_operational_metrics(
            metrics,
            grid,
            gui_params,
            time_series_data["voltages_data_stage2"],
            time_series_data["line_powers_data_stage2"],
            stage_one_data.get("pvw_powers", {}),
            stage_one_data.get("pvw_available_powers", {}),
        )
        return metrics, time_series_data
def plot_accumulated_costs(all_metrics, all_ts_data, seed, gui_params, grid_for_params):
    """
    Draw the cumulative cost comparison chart of all Algorithms.
    """
    plt.figure(figsize=(12, 8))

    start_hour, end_hour = gui_params['start_hour'], gui_params['end_hour']

    # Preload parameters required for cost calculation
    price = load_electricity_price(gui_params=gui_params)
    sb_mva = grid_for_params.SB
    step_h = gui_params['step_minutes'] / 60.0
    gen_costs_A = {g.ID: g.CostA(0) for g in grid_for_params.Gens}
    gen_costs_B = {g.ID: g.CostB(0) for g in grid_for_params.Gens}
    gen_costs_C = {g.ID: g.CostC(0) for g in grid_for_params.Gens}

    for algo_name, metrics in all_metrics.items():
        ts_data = all_ts_data.get(algo_name)
        if not ts_data or 'step_costs' not in ts_data:
            print(f"Warning: Algorithm {algo_name} is missing 'step_costs' data and cannot plot cumulative costs.")
            continue

        step_costs = ts_data['step_costs']
        if len(step_costs) == 0:
            continue

        # Calculate cumulative cost
        accumulated_costs = np.cumsum(step_costs)
        time_axis = np.linspace(start_hour, end_hour, len(accumulated_costs))

        # Draw curve
        plt.plot(time_axis, accumulated_costs, label=f"{algo_name} (Total Cost: {metrics.get('Total Cost', 0):.2f} yuan)",
                 marker='o', markersize=3, linestyle='-')

    plt.title(f'Comparison of cumulative running costs of each Algorithm (scenario Seed: {seed})', fontproperties="SimHei", size=16)
    plt.xlabel('Time (hours)', fontproperties="SimHei")
    plt.ylabel('Cumulative cost (yuan)', fontproperties="SimHei")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'accumulated_cost_seed_{seed}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"The cumulative cost comparison chart of scenario (Seed: {seed}) has been saved to: {save_path}")

def plot_aggregated_ev_power(all_ts_data, seed, gui_params):
    """
    Draw a total EV power graph containing 6 independent curves.
    - Separate the charging/discharging power of each pile first, and then sum them separately to avoid power hedging.
    """
    # Check if there is valid data
    algorithms = list(all_ts_data.keys())
    if not algorithms or not all_ts_data[algorithms[0]]:
        print(f"Scenario (Seed: {seed}) has no valid charging pile timing data, skip the total power drawing.")
        return

    # Create timeline
    start_hour, end_hour = gui_params['start_hour'], gui_params['end_hour']
    try:
        num_steps = len(next(iter(all_ts_data.values()))['spot_kw'])
        if num_steps == 0: raise IndexError
    except (StopIteration, KeyError, IndexError):
        print(f"The number of data steps is zero, skipping the total power plot of the scene (Seed: {seed}).")
        return
    time_axis = np.linspace(start_hour, end_hour, num_steps)

    # Start drawing
    fig, ax = plt.subplots(figsize=(15, 8))

    # —— Color mapping + automatic assignment ——
    algorithms = list(all_ts_data.keys())

    def _normalize_algo_name(name: str) -> str:
        s = name.strip()
        s = re.sub(r'(?i)_(two|single)_?stage', '', s)
        s = re.sub(r'(?i)\b(two|single)\s*stage\b', '', s)
        s = re.sub(r'[_\-\s]+seed\d+', '', s)
        s = re.sub(r'[_\-\s]+v\d+', '', s)
        s = re.sub(r'[_\-\s]+', '_', s).upper()
        s = s.replace('STABLE_BASELINES3_', '')
        return s

    BASE_COLOR_MAP = {
        'BASELINE': 'blue',
        'DDPG': '#e41a1c',
        'PPO': '#377eb8',
        'SAC': '#4daf4a',
        'TD3': '#984ea3',
        'A2C': '#ff7f00',
        'TRPO': '#a65628',
        'DQN': '#f781bf',
    }

    FALLBACK_PALETTE = [
        '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e',
        '#e6ab02', '#a6761d', '#666666', '#a6cee3', '#fb9a99'
    ]

    unknown_keys = sorted({
        _normalize_algo_name(a)
        for a in algorithms
        if _normalize_algo_name(a) not in BASE_COLOR_MAP and _normalize_algo_name(a) != 'BASELINE'
    })

    algo_colors = {}
    for a in algorithms:
        key = _normalize_algo_name(a)
        if key == 'BASELINE':
            algo_colors[a] = BASE_COLOR_MAP['BASELINE']
            continue
        color = BASE_COLOR_MAP.get(key)
        if color is None:
            idx = list(unknown_keys).index(key) % len(FALLBACK_PALETTE)
            color = FALLBACK_PALETTE[idx]
        algo_colors[a] = color

    for algo_name, ts_data in all_ts_data.items():
        if 'spot_kw' not in ts_data or ts_data['spot_kw'].size == 0:
            continue

        # The dimensions of spot_powers are (number of time steps, number of charging piles)
        spot_powers = ts_data['spot_kw']

        # 1. First, separate the charging power (positive value) and discharging power (negative value) at the level of each charging pile.
        all_charge_powers = np.maximum(spot_powers, 0)
        all_discharge_powers = np.minimum(spot_powers, 0)

        # 2. Then, calculate the sum along the dimensions of the charging pile (axis=1)
        total_charge_curve = np.sum(all_charge_powers, axis=1)
        total_discharge_curve = np.sum(all_discharge_powers, axis=1)

        # Get the basic color of this Algorithm
        color = algo_colors.get(algo_name, 'gray')

        # Draw two independent curves: one for charging and one for discharging
        # Use [solid line] to draw the charging power curve
        ax.plot(time_axis, total_charge_curve, color=color, linestyle='-', linewidth=2, label=f'{algo_name} charging power')

        # Use [dotted line] to draw the discharge power curve
        ax.plot(time_axis, total_discharge_curve, color=color, linestyle='--', linewidth=2, label=f'{algo_name} discharge power')

    # --- Chart beautification ---
    ax.axhline(0, color='black', linewidth=1.0)
    ax.set_title(f'Fine comparison of EV charging/discharging total power under each Algorithm (scenario Seed: {seed})', fontproperties="SimHei", size=16)
    ax.set_xlabel('Time (hours)', fontproperties="SimHei")
    ax.set_ylabel('Total power (kW)', fontproperties="SimHei")
    ax.legend(fontsize=12, ncol=3)
    ax.grid(True, linestyle=':', alpha=0.6)

    # --- Save chart ---
    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'aggregated_ev_power_corrected_{seed}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"[Logic corrected] EV charging and discharging separation comparison chart of scenario (Seed: {seed}) has been saved to: {save_path}")
def plot_and_save_results(all_ts_data, seed, gui_params):
    """Draw a comparison chart of all Algorithm time series data and save it to Excel"""

    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, f'comparison_data_seed_{seed}.xlsx')

    algorithms = list(all_ts_data.keys())
    if not algorithms or not all_ts_data[algorithms[0]]:
        print(f"Scene (Seed: {seed}) has no valid timing data, skip drawing and saving.")
        return

    # Create timeline
    start_hour, end_hour = gui_params['start_hour'], gui_params['end_hour']
    try:
        # Determine the number of steps based on the first valid data
        num_steps = len(next(iter(all_ts_data.values()))['pvw_pu'])
        if num_steps == 0: raise IndexError
    except (StopIteration, KeyError, IndexError):
        print(f"The number of data steps is zero, skip drawing and saving of scene (Seed: {seed}).")
        return

    start_hour, end_hour = gui_params['start_hour'], gui_params['end_hour']
    time_axis = np.linspace(start_hour, end_hour, num_steps)

    # --- Start drawing ---
    # 1. Total photovoltaic and wind power output
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        total_pvw_output = np.sum(all_ts_data[algo]['pvw_pu'], axis=1)
        plt.plot(time_axis, total_pvw_output, label=algo, marker='o', linestyle='--')
    plt.title(f'Comparison of total output of photovoltaic and wind power (scenario Seed: {seed})', fontproperties="SimHei", size=16)
    plt.xlabel('Time (hours)', fontproperties="SimHei")
    plt.ylabel('Total output (pu)', fontproperties="SimHei")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'pvw_total_output_seed_{seed}.png'))
    plt.close()

    # 2. ESS SOC changes
    num_ess = next(iter(all_ts_data.values())).get('ess_soc_percent', np.array([])).shape[1]
    if num_ess > 0:
        fig, axes = plt.subplots(num_ess, 1, figsize=(12, 4 * num_ess), sharex=True, squeeze=False)
        fig.suptitle(f'Comparison of SOC changes of energy storage system (ESS) (scenario Seed: {seed})', fontsize=16, fontproperties="SimHei")
        for i in range(num_ess):
            for algo in algorithms:
                soc_data_to_plot = all_ts_data[algo]['ess_soc_percent'][:-1, i]
                axes[i, 0].plot(time_axis, soc_data_to_plot, label=algo, marker='.')
            axes[i, 0].set_title(f'ESS #{i + 1}', fontproperties="SimHei")
            axes[i, 0].set_ylabel('SOC (%)', fontproperties="SimHei")
            axes[i, 0].legend()
            axes[i, 0].grid(True)
        axes[-1, 0].set_xlabel('Time (hours)', fontproperties="SimHei")
        plt.savefig(os.path.join(output_dir, f'ess_soc_seed_{seed}.png'))
        plt.close()

    # 3. Charging power of each charging pile
    num_spots = next(iter(all_ts_data.values())).get('spot_kw', np.array([])).shape[1]
    if num_spots > 0:
        fig, axes = plt.subplots(num_spots, 1, figsize=(12, 3 * num_spots), sharex=True, squeeze=False)
        fig.suptitle(f'Charging pile power change comparison (scenario Seed: {seed})', fontsize=16, fontproperties="SimHei")
        for i in range(num_spots):
            for algo in algorithms:
                axes[i, 0].plot(time_axis, all_ts_data[algo]['spot_kw'][:, i], label=algo, alpha=0.8)
            axes[i, 0].set_title(f'Charging pile #{i + 1}', fontproperties="SimHei")
            axes[i, 0].set_ylabel('Power (kW)', fontproperties="SimHei")
            axes[i, 0].legend()
            axes[i, 0].grid(True)
            axes[i, 0].axhline(0, color='black', linewidth=0.5)
        axes[-1, 0].set_xlabel('Time (hours)', fontproperties="SimHei")
        plt.savefig(os.path.join(output_dir, f'spot_power_seed_{seed}.png'))
        plt.close()

    print(f"The comparison picture of the scene (Seed: {seed}) has been saved to the folder: {output_dir}")

    # --- Start saving data to Excel ---
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Worksheet 1: Total photovoltaic and wind power output
        pvw_df_data = {'Time(h)': time_axis}
        for algo in algorithms:
            pvw_pu_data = all_ts_data[algo].get('pvw_pu', np.full((num_steps, 0), np.nan))
            pvw_df_data[f'{algo}_Total_PVW_Output(pu)'] = np.sum(pvw_pu_data,
                                                                 axis=1) if pvw_pu_data.ndim > 1 else pvw_pu_data
        pd.DataFrame(pvw_df_data).to_excel(writer, sheet_name='PV_Wind_Total_Output', index=False)

        # Sheet 2: ESS SOC
        if num_ess > 0:
            ess_df_data = {'Time(h)': time_axis}
            for i in range(num_ess):
                for algo in algorithms:
                    ess_soc_data = all_ts_data[algo].get('ess_soc_percent', np.full((num_steps, num_ess), np.nan))
                    soc_data_to_save = ess_soc_data[:-1, i]
                    ess_df_data[f'{algo}_ESS_{i + 1}_SOC(%)'] = soc_data_to_save
            pd.DataFrame(ess_df_data).to_excel(writer, sheet_name='ESS_SOC', index=False)

        # Worksheet 3: EV charging pile power
        if num_spots > 0:
            spot_df_data = {'Time(h)': time_axis}
            for i in range(num_spots):
                for algo in algorithms:
                    spot_kw_data = all_ts_data[algo].get('spot_kw', np.full((num_steps, num_spots), np.nan))
                    spot_df_data[f'{algo}_Spot_{i + 1}_Power(kW)'] = spot_kw_data[:, i]
            pd.DataFrame(spot_df_data).to_excel(writer, sheet_name='EV_Spot_Powers', index=False)

            #Save the voltage data of all nodes
            # [Bus voltage]
            print("...Writing to 'Bus_Voltages' worksheet")
            all_buses = set()
            for data in all_ts_data.values():
                voltages_data_dict = data.get('voltages_data_stage2', {})
                # Check if it is a dictionary (dict) and update the bus list from the dictionary's keys
                if voltages_data_dict and isinstance(voltages_data_dict, dict):
                    all_buses.update(voltages_data_dict.keys())

            if all_buses:
                def get_sort_key(bus_id):
                    try:
                        # Try to sort numerically (e.g. 'b10' comes after 'b2')
                        return int(''.join(filter(str.isdigit, bus_id)))
                    except:
                        return bus_id # If there is no number, sort by the original string

                sorted_buses = sorted(list(all_buses), key=get_sort_key)
                voltage_df_data = {'Time_Step': list(range(num_steps))}

                for bus_id in sorted_buses:
                    for algo_name, ts_data in all_ts_data.items():
                        col_name = f"{algo_name}_{bus_id}_V(pu)"
                        voltages_data_dict = ts_data.get('voltages_data_stage2', {})
                        # The data is now dict[list], directly use bus_id as the key to get the voltage list
                        voltage_list = voltages_data_dict.get(bus_id, [np.nan] * num_steps)
                        # Make sure the list length is consistent with the time step
                        voltage_df_data[col_name] = voltage_list[:num_steps]

                pd.DataFrame(voltage_df_data).to_excel(writer, sheet_name='Bus_Voltages', index=False)

            # 【Line Trend】
            print("...Writing to 'Line_Flows' worksheet")
            all_lines = set()
            for data in all_ts_data.values():
                flows_data_dict = data.get('line_powers_data_stage2', {})
                if flows_data_dict and isinstance(flows_data_dict, dict):
                    all_lines.update(flows_data_dict.keys())

            if all_lines:
                def get_line_sort_key(line_id):
                    try:
                        return int(''.join(filter(str.isdigit, line_id)))
                    except:
                        return line_id

                sorted_lines = sorted(list(all_lines), key=get_line_sort_key)
                flow_df_data = {'Time_Step': list(range(num_steps))}

                for line_id in sorted_lines:
                    for algo_name, ts_data in all_ts_data.items():
                        col_name = f"{algo_name}_{line_id}_P(pu)"
                        flow_data_dict = ts_data.get('line_powers_data_stage2', {})
                        flow_list = flow_data_dict.get(line_id, [np.nan] * num_steps)
                        flow_df_data[col_name] = flow_list[:num_steps]

                pd.DataFrame(flow_df_data).to_excel(writer, sheet_name='Line_Flows', index=False)
            print("...Writing to 'SOP_Flows' worksheet")
            baseline_sops = all_ts_data.get('Baseline', {}).get('raw_baseline_data', {}).get('sop_flows', {})
            if baseline_sops:
                sop_ids = list(baseline_sops.keys())
                sop_df_data = {'Time_Step': list(range(num_steps))}
                for sop_id in sop_ids:
                    # Add Baseline data
                    sop_df_data[f"Baseline_{sop_id}_P(pu)"] = baseline_sops[sop_id]['P1']
                    sop_df_data[f"Baseline_{sop_id}_Q1(pu)"] = baseline_sops[sop_id]['Q1']
                    if 'Q2' in baseline_sops[sop_id]:
                        sop_df_data[f"Baseline_{sop_id}_Q2(pu)"] = baseline_sops[sop_id]['Q2']

                    # Add RL data
                    for algo_name, ts_data in all_ts_data.items():
                        if algo_name != 'Baseline':
                            sop_p_data = ts_data.get('sop_p_pu', np.array([]))
                            sop_q1_data = ts_data.get('sop_q1_pu', ts_data.get('sop_q_pu', np.array([])))
                            sop_q2_data = ts_data.get('sop_q2_pu', np.array([]))
                            sop_idx = sop_ids.index(sop_id)
                            if sop_p_data.ndim == 2 and sop_p_data.shape[1] > sop_idx:
                                sop_df_data[f"{algo_name}_{sop_id}_P(pu)"] = sop_p_data[:, sop_idx]
                            if sop_q1_data.ndim == 2 and sop_q1_data.shape[1] > sop_idx:
                                sop_df_data[f"{algo_name}_{sop_id}_Q1(pu)"] = sop_q1_data[:, sop_idx]
                            if sop_q2_data.ndim == 2 and sop_q2_data.shape[1] > sop_idx:
                                sop_df_data[f"{algo_name}_{sop_id}_Q2(pu)"] = sop_q2_data[:, sop_idx]
                pd.DataFrame(sop_df_data).to_excel(writer, sheet_name='SOP_Flows', index=False)

            # NOP data saving
            print("...Writing to 'NOP_Status' worksheet")
            baseline_nops = all_ts_data.get('Baseline', {}).get('raw_baseline_data', {}).get('nop_status', {})
            if baseline_nops:
                nop_ids = list(baseline_nops.keys())
                nop_df_data = {'Time_Step': list(range(num_steps))}
                for nop_id in nop_ids:
                    # Add Baseline data
                    nop_df_data[f"Baseline_{nop_id}_Status"] = baseline_nops[nop_id]

                    # Add RL data
                    for algo_name, ts_data in all_ts_data.items():
                        if algo_name != 'Baseline':
                            nop_status_data = ts_data.get('nop_status', np.array([]))
                            nop_idx = nop_ids.index(nop_id)
                            if nop_status_data.shape[1] > nop_idx:
                                nop_df_data[f"{algo_name}_{nop_id}_Status"] = nop_status_data[:, nop_idx]
                pd.DataFrame(nop_df_data).to_excel(writer, sheet_name='NOP_Status', index=False)

    print(f"The detailed time series data of scene (Seed: {seed}) has been saved to Excel file: {os.path.abspath(excel_path)}")


def plot_ev_occupancy(env_instance, seed, gui_params):
    """
    Draw the changing curve of the number of vehicles present in the charging station in real time under a specific scenario.
    This is a diagnostic tool to verify that the charging power matches the number of vehicles.
    """
    if not hasattr(env_instance, 'ev_present'):
        print("Diagnostic error: Unable to find ev_present data from environment.")
        return

    # Get the ev_present data from the environment (this is a 0/1 matrix of stakes x time steps)
    ev_present_matrix = env_instance.ev_present

    # Sum along the dimension of the charging pile (axis=0) to get the total number of vehicles present at each time step
    num_cars_present_per_step = np.sum(ev_present_matrix, axis=0)

    # Create timeline
    start_hour, end_hour = gui_params['start_hour'], gui_params['end_hour']
    num_steps = len(num_cars_present_per_step)
    time_axis = np.linspace(start_hour, end_hour, num_steps)

    # Drawing
    plt.figure(figsize=(15, 6))
    plt.plot(time_axis, num_cars_present_per_step, label='Real-time number of vehicles present', color='purple', drawstyle='steps-post')

    plt.title(f'Real-time changes in the number of vehicles present at the charging station (scenario Seed: {seed})', fontproperties="SimHei", size=16)
    plt.xlabel('Time (hours)', fontproperties="SimHei")
    plt.ylabel('Number of vehicles present (vehicles)', fontproperties="SimHei")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()

    # Save chart
    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'ev_occupancy_seed_{seed}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"[Diagnostic Tool] The graph of the number of vehicles present in the scene (Seed: {seed}) has been saved to: {save_path}")


if __name__ == '__main__':
    # =================================================================================
    # 1. Load core configuration from config.py
    # =================================================================================
    EVALUATION_MODE = EVALUATION_CONFIG["flow_mode"]
    gui_params = CORE_PARAMS
    num_test_episodes = EVALUATION_CONFIG["num_test_episodes"]

    print(f"\n{'=' * 35}")
    print(f"Current evaluation mode: {EVALUATION_MODE.upper()}")
    print(f"{'=' * 35}\n")

    # =================================================================================
    # 2. Dynamically discover and load all available Algorithm classes
    # - First load the built-in Algorithm of stable_baselines3 as the basis.
    # - Then scan the custom_algorithms/ folder and load all user-defined Algorithm plug-ins.
    # =================================================================================
    print(f"{'=' * 35}")
    print("Dynamicly discovering and loading all available Algorithms...")


    # Define a function to perform discovery and loading logic to keep the main process clear
    def discover_and_load_algorithms():
        model_class_registry = {}

        # 2.1. Load the built-in Algorithm of stable-baselines3
        from stable_baselines3 import PPO, DDPG, SAC, TD3
        model_class_registry.update({"PPO": PPO, "DDPG": DDPG, "SAC": SAC, "TD3": TD3})

        # 2.2. Scan all python files in the custom_algorithms directory
        plugin_dir = "custom_algorithms"
        if not os.path.isdir(plugin_dir):
            print(f" Message: Plugin directory 'izer{plugin_dir}' not found, only built-in Algorithm will be used.")
            return model_class_registry

        import glob
        import importlib.util
        plugin_files = glob.glob(os.path.join(plugin_dir, "*.py"))

        for plugin_file in plugin_files:
            module_name = os.path.basename(plugin_file)[:-3]
            try:
                spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, 'register_algorithm'):
                    registration_info = module.register_algorithm()
                    algo_name = registration_info['name']
                    algo_class = registration_info['class']
                    model_class_registry[algo_name] = algo_class
                    print(f" [plug-in loader] successfully loaded custom Algorithm: '{algo_name}'")
                else:
                    print(
                        f" [plugin loader] warning: File {os.path.basename(plugin_file)} is not a valid plug-in (missing register_algorithm function).")
            except Exception as e:
                print(f" [Plugin Loader] Error: Loading plugin {os.path.basename(plugin_file)} failed: {e}")

        return model_class_registry


    # Perform discovery operation
    model_name_to_class = discover_and_load_algorithms()
    print(f" Loading completed! Currently available Algorithm: {list(model_name_to_class.keys())}")
    print(f"{'=' * 35}\n")

    # =================================================================================
    # 3. [Automation Core] Automatically discover all trained models in the models/ directory
    # - Instead of reading the hardcoded list in config.py, scan the folder directly.
    # =================================================================================
    print(f"{'=' * 35}")
    print("Automatically scanning the 'models/' directory to discover trained models...")
    models_to_evaluate = {}
    model_folders = [f for f in os.listdir(PATHS["models_dir"]) if os.path.isdir(os.path.join(PATHS["models_dir"], f))]

    for folder_name in model_folders:
        # e.g., folder_name = "best_brilliantalgo_two_stage" -> display_name = "BrilliantAlgo_Two_Stage"
        display_name = folder_name.replace("best_", "").replace("_", " ").title().replace(" ", "_")
        model_path = os.path.join(PATHS["models_dir"], folder_name, "best_model.zip")

        if not os.path.exists(model_path):
            print(f" Warning: 'best_model.zip' not found in folder '{folder_name}', skipping.")
            continue

        # Infer the type from the model name and find the corresponding class in the registered Algorithm
        model_type_key = next((key for key in model_name_to_class if key in display_name.upper()), None)

        if model_type_key:
            model_class = model_name_to_class[model_type_key]
            models_to_evaluate[display_name] = (model_path, model_class)
            print(f"[ModelFinder] found model '{display_name}' and associated it to class '{model_type_key}'.")
        else:
            print(f" Warning: No class matching '{display_name}' found in registered Algorithm, skipping this model.")
    print(f" Discovery complete! The following models will be evaluated: {list(models_to_evaluate.keys())}")
    print(f"{'=' * 35}\n")

    # =================================================================================
    # 4. Initialize environment and result recorder
    # =================================================================================
    all_results_metrics = []
    try:
        use_two_stage = (EVALUATION_MODE == 'two_stage')
        rl_env = PowerGridEnv(gui_params=gui_params, use_two_stage_flow=use_two_stage)
    except Exception as e:
        print(f"Failed to initialize PowerGridEnv: {e}")
        traceback.print_exc()
        exit()

    # =================================================================================
    # 5. Main evaluation loop
    # - Traverse all test scenarios (episodes).
    # - In each scenario, run Baseline and all discovered RL models in sequence.
    # - Generate graphs and data files for each scenario.
    # =================================================================================
    for i in range(num_test_episodes):
        seed = i
        print(f"\n--- Start evaluating scenario {i + 1}/{num_test_episodes} (seed={seed}) ---\n")

        all_metrics_for_this_seed = {}
        time_series_log_for_this_seed = {}

        # Core synchronization logic: first use the RL environment with seed to generate standard scenarios for use by Baseline and RL Agent.
        rl_env.reset(seed=seed)
        stations_list_for_this_seed = rl_env.stations_list
        grid_instance_for_this_seed = rl_env.grid

        # 5.1. Evaluate Baseline
        try:
            grid_for_baseline = deepcopy(grid_instance_for_this_seed)
            baseline_metrics, baseline_ts = evaluate_baseline(gui_params, seed, stations_list_for_this_seed,
                                                              grid_for_baseline,
                                                              use_two_stage=use_two_stage)
            if baseline_metrics and not pd.isna(baseline_metrics.get("Total Cost")):
                baseline_metrics['Algorithm'] = 'Baseline'
                all_results_metrics.append(baseline_metrics)
                all_metrics_for_this_seed['Baseline'] = baseline_metrics
                time_series_log_for_this_seed['Baseline'] = baseline_ts
        except Exception as e:
            print(f"A serious error occurred while evaluating Baseline: {e}")
            traceback.print_exc()

        # 5.2. Evaluate all discovered RL models
        for model_name, (model_path, model_class) in models_to_evaluate.items():
            try:
                model_dir = os.path.dirname(model_path)
                manifest = {}
                manifest_path = os.path.join(model_dir, "model_manifest.json")
                if os.path.exists(manifest_path):
                    try:
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            manifest = json.load(f)
                    except Exception:
                        manifest = {}
                eval_normalizer = load_eval_vecnormalize(
                    rl_env,
                    model_dir,
                    training_config=TRAINING_CONFIG,
                    manifest=manifest,
                )
                model_env = eval_normalizer if eval_normalizer is not None else rl_env
                model = model_class.load(model_path, env=model_env)
                rl_metrics, rl_ts = evaluate_rl_agent(
                    model,
                    rl_env,
                    seed,
                    obs_normalizer=eval_normalizer,
                )
                rl_metrics['Algorithm'] = model_name
                all_results_metrics.append(rl_metrics)
                all_metrics_for_this_seed[model_name] = rl_metrics
                time_series_log_for_this_seed[model_name] = rl_ts
            except Exception as e:
                print(f"A serious error occurred while evaluating {model_name}: {e}")
                traceback.print_exc()

        # 5.3. Generate all comparison charts and data files for the current scene
        if time_series_log_for_this_seed:
            print("\n--- Generating visual reports and data files for the current scenario... ---")
            # (call all drawing functions)
            plot_and_save_results(time_series_log_for_this_seed, seed, gui_params)
            plot_accumulated_costs(all_metrics_for_this_seed, time_series_log_for_this_seed, seed, gui_params,
                                   grid_instance_for_this_seed)
            plot_voltage_snapshots(time_series_log_for_this_seed, seed, gui_params)
            plot_line_flow_snapshots_comparison(time_series_log_for_this_seed, seed, gui_params)
            plot_aggregated_ev_power(time_series_log_for_this_seed, seed, gui_params)
            plot_sop_flows(time_series_log_for_this_seed, seed, gui_params)
            plot_nop_status(time_series_log_for_this_seed, seed, gui_params)
            print("--- Report and data file generation completed. ---\n")
        else:
            print(f"Scenario (Seed: {seed}) has no valid evaluation results, skip drawing.")

    # =================================================================================
    # 6. Summary of final results
    # - Print a comparison table of average performance indicators for all scenarios.
    # =================================================================================
    if not all_results_metrics:
        print("\nAll evaluations were unsuccessful and the summary report cannot be generated.")
    else:
        results_df = pd.DataFrame(all_results_metrics)
        display_columns = [
            "Total Cost", "Grid Purchase Cost", "Generation Cost", "SOP Loss Cost", "ESS Discharge Cost",
            "Exact Total Grid Loss (kW)", "Voltage Compliance Rate(%)", "EV Charging Satisfaction Rate (%)"
        ]
        existing_display_columns = [col for col in display_columns if col in results_df.columns]
        summary = results_df.groupby('Algorithm')[existing_display_columns].mean()

        print("\n\n" + "=" * 110)
        print(" " * 45 + "Simulation evaluation results summary (Mean - average)" + " " * 45)
        print("=" * 110)
        print(summary.to_string(float_format="%.4f"))
        print("=" * 110)
