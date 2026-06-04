import gymnasium as gym
import numpy as np
import pandas as pd
import os
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, Binary, minimize, value, \
    NonNegativeReals
from config import PATHS
# Import modules in the project
from grid_model import create_grid, load_electricity_price, load_station_info
from gev_station import GEVStation, EVParameters, average_hour_multiplier
from sop_nop import SOP, NOP
from fpowerkit import Generator
# Import OpenDSS solver and voltage correction tool
from fpowerkit.soldss import OpenDSSSolver
from fpowerkit.solbase import GridSolveResult
from two_stage_powerflow import fix_bus_voltage_limits
from config import RL_ENV_CONFIG, CORE_PARAMS, EVALUATION_CONFIG, BASELINE_PARAMS

class PowerGridEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, gui_params: dict, use_two_stage_flow: bool = True):
        super().__init__()
        self.params = gui_params
        self.use_two_stage_flow = use_two_stage_flow
        self.obs_power_base_kw = self._positive_float(self.params.get("obs_power_base_kw", 1000.0), 1000.0)
        self.obs_price_base = self._positive_float(self.params.get("obs_price_base", 1.0), 1.0)
        # Reward weight
        self.reward_weights = dict(RL_ENV_CONFIG["reward_weights"])
        self.reward_weights.update(self.params.get("reward_weights", {}))
        self.reward_scale = RL_ENV_CONFIG.get("reward_scale", 1.0)
        self.failure_penalty = self.params.get("reward_weights", {}).get(
            "opendss_failure_penalty",
            RL_ENV_CONFIG["penalties"]["opendss_failure_penalty"]
        )

        # Reward mode / operator parameters
        self.reward_mode = self.params.get("reward_mode", "grid_operator")
        self.station_operator_cfg = self.params.get("station_operator", {})
        self.stations_info = load_station_info(gui_params=self.params)
        if not self.stations_info:
            raise ValueError("Failed to load any charging station information from parameter file!")
        ev_cfg = self.params.get("ev_params", {})
        self.node_overrides = self.params.get("node_overrides", {}) or {}

        self.stations_list = []
        for info in self.stations_info:
            station = GEVStation(
                station_id=info['Station_ID'],
                num_spots=info['Num_Spots'],
                ev_params=EVParameters(
                    capacity_kwh=ev_cfg.get("capacity_kwh", 70.0),
                    max_charge_kw=ev_cfg.get("max_charge_kw", 60.0),
                    max_discharge_kw=ev_cfg.get("max_discharge_kw", 25.0),
                    charge_efficiency=ev_cfg.get("charge_efficiency", 0.95),
                    discharge_efficiency=ev_cfg.get("discharge_efficiency", 0.90),
                )
            )
            station.bus_id = info['Bus_ID']
            self.stations_list.append(station)

        self.ev_params = self.stations_list[0].ev_params if self.stations_list else EVParameters()
        self._apply_ev_spot_overrides_to_stations()
        self._rebuild_spot_index()

        print(f"The RL environment has been initialized, with a total of {len(self.stations_list)} charging stations and {self.total_spots} charging piles.")
        print(f"Current power flow calculation mode: {'Two-stage (DistFlow+OpenDSS)' if self.use_two_stage_flow else 'Single-stage (DistFlow only)'}")

        self.grid_template = create_grid(model=self.params['grid_model'], gui_params=self.params)
        self.price = load_electricity_price(gui_params=self.params)
        self.total_timesteps = int(
            (self.params['end_hour'] - self.params['start_hour']) * (60 // self.params['step_minutes']))

        # 1: Define virtual generator template during initialization
        self.VIRTUAL_GEN_ID = 'gen_for_slack_bus'
        self.slack_bus_id = self.params.get('slack_bus', 'b1')
        self.slack_generator_template = Generator(
            id=self.VIRTUAL_GEN_ID, busid=self.slack_bus_id,
            pmax_pu=9999, pmin_pu=-9999, qmax_pu=9999, qmin_pu=-9999,
            costA=0, costB=0, costC=0
        )

        # Also add this attribute to the virtual generator template of the RL environment
        self.slack_generator_template.RealisticPmax = 9999

        # (Component lists, action/observation space definitions remain unchanged)
        self.ess_list = list(self.grid_template.ESSs) if hasattr(self.grid_template, 'ESSs') else []
        self.pvw_list = list(self.grid_template.PVWinds) if hasattr(self.grid_template, 'PVWinds') else []
        self.sop_list = list(self.grid_template.SOPs.values()) if hasattr(self.grid_template, 'SOPs') else []
        self.nop_metadata_list = list(self.grid_template.NOPs.values()) if hasattr(self.grid_template, 'NOPs') else []
        self.nop_list = []
        self._configure_spaces()
        self.grid = None
        self.current_step = 0
        self.ev_present = None
        self.ev_boc = None
        self.active_session_map = {}
        print(
            "[RL-Normalization] Observation internal scaling enabled: "
            f"obs_power_base_kw={self.obs_power_base_kw}, obs_price_base={self.obs_price_base}"
        )

    def _reset_episode_metrics(self):
        self.episode_metrics = {
            # Base rewards and costs
            "ep_reward_unscaled": 0.0,
            "ep_total_cost": 0.0,
            "ep_total_objective_cost": 0.0,
            "ep_grid_purchase_cost": 0.0,
            "ep_generation_cost": 0.0,
            "ep_ess_discharge_cost": 0.0,
            "ep_sop_loss_cost": 0.0,

            # DistFlow / Physical constraint relaxation penalty
            "ep_distflow_penalty": 0.0,
            "ep_slack_penalty": 0.0,
            "ep_sop_slack_penalty": 0.0,
            "ep_nop_slack_penalty": 0.0,

            # Voltage and OpenDSS
            "ep_voltage_penalty_cost": 0.0,
            "ep_voltage_violation_count": 0.0,
            "ep_opendss_failure_penalty_cost": 0.0,
            "ep_opendss_failure_count": 0.0,

            # EV
            "ep_ev_shortage_penalty_cost": 0.0,
            "ep_ev_urgency_penalty": 0.0,
            "ep_ev_charge_kwh": 0.0,
            "ep_ev_discharge_kwh": 0.0,

            # PV/Wind consumption
            "ep_pvw_available_pu": 0.0,
            "ep_pvw_actual_pu": 0.0,
            "ep_pvw_curtailment_pu": 0.0,

            # SOP Action Projection Diagnosis
            "ep_sop_projection_count": 0.0,
            "ep_sop_projection_scale_sum": 0.0,
            "ep_sop_projection_scale_min": 1.0,
            "ep_sop_pre_projection_norm_max": 0.0,
        }

    def _to_number(self, value, default=0.0):
        if isinstance(value, (list, tuple)):
            value = value[0] if value else default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _positive_float(self, value, default=1.0):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(default)
        return parsed if parsed > 0 else float(default)

    def _power_unit_to_kw(self):
        grid = self.grid if self.grid is not None else self.grid_template
        sb_mva = getattr(grid, "SB", self.params.get("base_power", 1.0))
        return self._positive_float(sb_mva, 1.0) * 1000.0

    def _scale_power_kw(self, value):
        return float(value) * self._power_unit_to_kw() / self.obs_power_base_kw

    def _scale_price(self, value):
        return float(value) / self.obs_price_base

    def _apply_ev_spot_overrides_to_stations(self):
        """Apply node-level EV spot expansion before deriving action/observation shapes."""
        for node_name, attrs in self.node_overrides.items():
            attrs_dict = vars(attrs) if hasattr(attrs, "__dict__") else (attrs or {})
            add_ev_spots = int(self._to_number(attrs_dict.get("add_ev_spots", 0), 0))
            if add_ev_spots <= 0:
                continue
            stations_at_bus = [s for s in self.stations_list if getattr(s, "bus_id", None) == node_name]
            if not stations_at_bus:
                station_id = f"scenario_ev_station_{node_name}"
                existing_ids = {s.station_id for s in self.stations_list}
                suffix = 2
                base_station_id = station_id
                while station_id in existing_ids:
                    station_id = f"{base_station_id}_{suffix}"
                    suffix += 1
                station = GEVStation(
                    station_id=station_id,
                    num_spots=add_ev_spots,
                    ev_params=self.ev_params,
                )
                station.bus_id = node_name
                self.stations_list.append(station)
                print(f"[Scenario] Node {node_name}: created EV station {station_id} with {add_ev_spots} spots.")
                continue
            for station in stations_at_bus:
                old_spots = station.num_spots
                station.num_spots += add_ev_spots
                print(f"[Scenario] Node {node_name}: EV spots {old_spots} -> {station.num_spots}.")

    def _rebuild_spot_index(self):
        self.total_spots = sum(station.num_spots for station in self.stations_list)
        self.spot_to_bus_map = {}
        spot_counter = 0
        for station in self.stations_list:
            for local_spot_idx in range(station.num_spots):
                self.spot_to_bus_map[spot_counter + local_spot_idx] = station.bus_id
            spot_counter += station.num_spots

    def _configure_spaces(self):
        action_dim = self.total_spots + len(self.ess_list) + len(self.pvw_list) + 3 * len(self.sop_list) + len(
            self.nop_list)
        low_bounds = np.array(
            [-1] * self.total_spots + [-1] * len(self.ess_list) + [0] * len(self.pvw_list) + [-1] * 3 * len(
                self.sop_list) + [0] * len(self.nop_list), dtype=np.float32)
        high_bounds = np.array([1] * action_dim, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
        obs_dim = (2 + 2 * len(self.grid_template.Buses) + len(self.pvw_list) + len(
            self.ess_list) + 4 * self.total_spots)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _apply_grid_node_overrides(self):
        # Grid-level node overrides are materialized in create_grid().
        return

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)

        self.current_step = 0
        # 1. Create a clean grid instance
        self.grid = create_grid(model=self.params['grid_model'], gui_params=self.params)

        # 2. Add all necessary virtual generators for this new grid instance
        if self.use_two_stage_flow:
            # Add virtual generator for balancing (create new instance)
            if self.VIRTUAL_GEN_ID not in self.grid.GenNames:
                new_slack_gen = Generator(
                    id=self.VIRTUAL_GEN_ID, busid=self.slack_bus_id,
                    pmax_pu=9999, pmin_pu=0, qmax_pu=9999, qmin_pu=-9999,
                    costA=0, costB=0, costC=0
                )
                new_slack_gen.RealisticPmax = 9999 # Consistent with baseline.py
                self.grid.AddGen(new_slack_gen)
            print(f"--- RL Env Reset: Created new virtual generator to '{self.slack_bus_id}' for two-phase mode. ---")

            # When resetting, clear the mapping first
            self.sop_gen_map = {}
            # Create a new equivalent generator for the SOP to avoid reusing old instances
            if hasattr(self.grid, 'SOPs'):
                print(f"--- RL Env Reset: Creating a new equivalent generator for SOP for grid instance {len(self.grid.SOPs)}... ---")
                for sop_id, sop in self.grid.SOPs.items():
                    g1 = Generator(id=f"gen_for_{sop.ID}_bus1", busid=sop.Bus1,
                                   pmin_pu=-sop.PMax, pmax_pu=sop.PMax,
                                   qmin_pu=-sop.QMax, qmax_pu=sop.QMax, costA=0, costB=0, costC=0)
                    g2 = Generator(id=f"gen_for_{sop.ID}_bus2", busid=sop.Bus2,
                                   pmin_pu=-sop.PMax, pmax_pu=sop.PMax,
                                   qmin_pu=-sop.QMax, qmax_pu=sop.QMax, costA=0, costB=0, costC=0)
                    g1.RealisticPmax = sop.PMax
                    g2.RealisticPmax = sop.PMax

                    self.grid.AddGen(g1)
                    self.grid.AddGen(g2)
                    self.sop_gen_map[sop.ID] = (g1, g2)

        self.original_bus_pds = {bus.ID: bus.Pd for bus in self.grid.Buses}
        self.original_bus_qds = {bus.ID: bus.Qd for bus in self.grid.Buses}
        fix_bus_voltage_limits(self.grid)

        self.ess_list = list(self.grid.ESSs) if hasattr(self.grid, 'ESSs') else []
        self.pvw_list = list(self.grid.PVWinds) if hasattr(self.grid, 'PVWinds') else []
        self.sop_list = list(self.grid.SOPs.values()) if hasattr(self.grid, 'SOPs') else []
        self.nop_metadata_list = list(self.grid.NOPs.values()) if hasattr(self.grid, 'NOPs') else []
        self.nop_list = []
        # Rebuild the NOP line mapping and bind it to the "current grid" instance
        self.nop_line_map = {}
        if self.nop_list:
            for nop in self.nop_list:
                line_id = f"line_for_{nop.ID}"
                # Ensure that the line corresponding to line_id exists in self.grid (grid_model.py ensures this)
                self.nop_line_map[nop.ID] = self.grid.Line(line_id)

        # Get data source selection from self.params (i.e. CORE_PARAMS)
        # This value is set by simulation_runner, the source is GUI
        ev_data_source_mode = self.params.get('ev_data_source', 'random') # The default is 'random'

        print("\n" + "=" * 25 + " RL environment scene loading " + "=" * 25)
        print(f" [Configuration information] EV data source selected by GUI: '{ev_data_source_mode}'")

        if ev_data_source_mode == 'external':
            # --- Mode 1: GUI select "Use external file" ---
            print(f"[Load mode] Try loading from external file...")
            ev_scenario_file = PATHS["ev_scenarios_csv"]

            # Strict file existence checks must be performed
            if not os.path.exists(ev_scenario_file):
                print(f" [FATAL ERROR] GUI selected 'Use external file', but file not found: {os.path.abspath(ev_scenario_file)}")
                print(
                    f" [FATAL ERROR] Please place the {os.path.basename(ev_scenario_file)} file in the data folder, or select 'Randomly generated' in the GUI.")
                print("=" * 75 + "\n")
                # Throw an exception and terminate the operation to prevent the program from using incorrect data.
                raise FileNotFoundError(f"EV scenario file not found: {ev_scenario_file}. GUI requested 'external'.")

            # If the file exists, load it
            print(f" [Diagnosis Result] File found! Scenario will be loaded from CSV: {ev_scenario_file}")
            print("=" * 75 + "\n")
            try:
                all_scenarios_df = pd.read_csv(ev_scenario_file)
                for station in self.stations_list:
                    station.load_scenarios_from_csv(all_scenarios_df)
            except Exception as e:
                print(f"[RL environment error]: Failed to load EV scene file {ev_scenario_file}: {e}")
                raise e

        else:
            # --- Mode 2: GUI select "Randomly generated" (or default) ---
            print(f"[Loading mode] Start random scene generator...")
            print("=" * 75 + "\n")
            # [FIX 3] Apply global EV multiplier to randomly generated traffic volume
            global_ev_multiplier = self.params.get('global_ev_multiplier', 1.0)
            ev_hour_multipliers = (self.params.get("time_profiles") or {}).get("ev_multiplier_by_hour")
            ev_profile_average = average_hour_multiplier(ev_hour_multipliers, 1.0)
            effective_ev_multiplier = ev_profile_average if ev_hour_multipliers else float(global_ev_multiplier)
            print(
                f" [Scene Control] Global EV multiplier: {global_ev_multiplier}, "
                f"Time-sharing EV average multiplier: {ev_profile_average:.3f}"
            )
            for i, station in enumerate(self.stations_list):
                station_info = self.stations_info[i] if i < len(self.stations_info) else {}
                base_spots = max(1, int(station_info.get('Num_Spots', station.num_spots)))
                default_num_evs = station.num_spots * 5
                base_num_evs = float(station_info.get('Num_EVs_to_Generate', default_num_evs)) * (
                    station.num_spots / base_spots
                )
                num_evs = int(base_num_evs * effective_ev_multiplier)
                station.generate_daily_scenarios(
                    num_evs_to_generate=num_evs,
                    arrival_hour_multipliers=ev_hour_multipliers,
                )

        self._prepare_ev_simulation_data()
        self._reset_episode_metrics()

        return self._get_observation(), {}

    def step(self, action: np.ndarray):

        physical_actions = self._apply_action(action)
        reward_unscaled = 0.0
        info = {}

        if self.use_two_stage_flow:
            distflow_results = self._solve_optimal_flow_for_step(physical_actions)
            voltages_stage1 = distflow_results.get('voltages', {})

            self._update_grid_for_opendss(physical_actions, distflow_results)
            nop_statuses = physical_actions.get('nop_status', [])
            for i, nop in enumerate(self.nop_list):
                if nop.ID in self.nop_line_map:
                    line_to_control = self.nop_line_map[nop.ID]
                    is_closed = (nop_statuses[i] == 1)
                    line_to_control.active = is_closed

            sop_losses_pu = distflow_results.get("sop_losses_pu", {})
            sop_p_powers = physical_actions.get('sop_p_power', [])
            sop_q1_powers = physical_actions.get('sop_q1_power', physical_actions.get('sop_q_power', []))
            sop_q2_powers = physical_actions.get('sop_q2_power', physical_actions.get('sop_q_power', []))
            for i, sop in enumerate(self.sop_list):
                if sop.ID in self.sop_gen_map:
                    gen1, gen2 = self.sop_gen_map[sop.ID]
                    p_target = sop_p_powers[i]
                    q1_target = sop_q1_powers[i]
                    q2_target = sop_q2_powers[i]
                    loss_pu = sop_losses_pu.get(sop.ID, 0.0)
                    gen1._p, gen1._q = -p_target, q1_target
                    gen2._p, gen2._q = p_target - loss_pu, q2_target

            original_load_funcs = {}
            sb_mva = self.params.get('base_power', 1.0)
            ev_powers_kw = physical_actions.get('ev_power', [])
            bus_ev_load_pu = {bus.ID: 0.0 for bus in self.grid.Buses}
            for spot_idx, power_kw in enumerate(ev_powers_kw):
                # Removed 'if power_kw > 0' condition
                if abs(power_kw) > 1e-6: # Check for non-zero
                    bus_id = self.spot_to_bus_map[spot_idx]
                    power_pu = power_kw / (sb_mva * 1000.0)
                    # Charge (positive) and V2G (negative) will be accumulated
                    bus_ev_load_pu[bus_id] += power_pu
            try:
                for bus_id, ev_load in bus_ev_load_pu.items():
                    # Change 'if ev_load > 0' to check for non-zero
                    if abs(ev_load) > 1e-6:
                        bus = self.grid.Bus(bus_id)
                        original_func = bus.Pd
                        original_load_funcs[bus_id] = original_func
                        # V2G (negative value) will reduce Pd, C charge (positive value) will increase Pd
                        bus.Pd = lambda t, _original_func=original_func, _ev_load=ev_load: _original_func(t) + _ev_load
                opendss_solver = OpenDSSSolver(self.grid, source_bus=self.params['slack_bus'])
                # Convert time steps into seconds, consistent with evaluate_agents.py
                time_in_seconds = self.current_step * self.params['step_minutes'] * 60
                opendss_result_status, opendss_loss_W = opendss_solver.solve(time_in_seconds)
            finally:
                for bus_id, original_func in original_load_funcs.items():
                    self.grid.Bus(bus_id).Pd = original_func

            generation_cost = distflow_results.get('generation_cost', 0)
            ess_discharge_cost = distflow_results.get('ess_discharge_cost', 0)
            sop_loss_cost = distflow_results.get('sop_loss_cost', 0)
            slack_penalty = distflow_results.get('slack_penalty', 0.0)
            sop_slack_penalty = distflow_results.get('sop_slack_penalty', 0.0)
            nop_slack_penalty = distflow_results.get('nop_slack_penalty', 0.0)
            distflow_penalty = slack_penalty + sop_slack_penalty + nop_slack_penalty

            if opendss_result_status == GridSolveResult.OK:
                voltages_stage2 = {bus.ID: bus.V for bus in self.grid.Buses if bus.V is not None}
                slack_generator = self.grid.Gen(self.VIRTUAL_GEN_ID)
                precise_inflow_pu = slack_generator.P if slack_generator and slack_generator.P is not None else 0
                precise_inflow_pu = max(0.0, precise_inflow_pu) # Disable negative power from being included in Grid Purchase Cost
                price_now = self.price[self.current_step]
                sb_mva_kw = self.params['base_power'] * 1000
                step_h = self.params['step_minutes'] / 60.0
                precise_grid_cost = price_now * precise_inflow_pu * sb_mva_kw * step_h
                total_cost = precise_grid_cost + generation_cost + ess_discharge_cost + sop_loss_cost
                total_objective_cost = total_cost + distflow_penalty

                standards = EVALUATION_CONFIG["standards"]
                violations = sum(1 for v in voltages_stage2.values()
                                 if not (standards["voltage_min_pu"] <= v <= standards["voltage_max_pu"]))
                voltage_penalty = self._get_voltage_penalty(voltages_stage2)
                base_reward = -total_cost * self.reward_weights["cost_penalty_factor"] + voltage_penalty

                info = {
                    'grid_purchase_cost': precise_grid_cost,
                    'generation_cost': generation_cost,
                    'ess_discharge_cost': ess_discharge_cost,
                    'sop_loss_cost': sop_loss_cost,
                    'slack_penalty': slack_penalty,
                    'sop_slack_penalty': sop_slack_penalty,
                    'nop_slack_penalty': nop_slack_penalty,
                    'distflow_penalty': distflow_penalty,
                    'voltages_stage1': voltages_stage1,
                    'voltages_stage2': voltages_stage2,
                    'line_powers_stage1': distflow_results.get('line_powers', {}),
                    'line_powers_stage2': {line.ID: line.P for line in self.grid.Lines if line.P is not None},
                    'pvw_power_pu': physical_actions.get('pvw_power', []),
                    'ev_power_kw': physical_actions.get('ev_power', []),
                    'opendss_loss_W': opendss_loss_W,
                    'total_cost': total_cost,
                    'total_objective_cost': total_objective_cost,
                    'voltage_penalty_unscaled': voltage_penalty,
                    'opendss_failure_penalty_unscaled': 0.0,
                    'base_reward_unscaled': base_reward,
                    'reward_mode': self.reward_mode,
                    'opendss_failed': False,
                }

                if self.reward_mode == "station_operator":
                    reward_unscaled = self._calculate_station_operator_reward(
                        physical_actions=physical_actions,
                        price_now=price_now,
                        info=info
                    )
                else:
                    reward_unscaled = base_reward
            else:
                total_cost = distflow_results.get('total_cost', 1e7)
                total_objective_cost = total_cost + distflow_penalty
                failure_penalty = self.failure_penalty
                base_reward = -total_cost * self.reward_weights["cost_penalty_factor"] + failure_penalty

                info = {
                    'grid_purchase_cost': distflow_results.get('grid_purchase_cost', 0),
                    'generation_cost': distflow_results.get('generation_cost', 0),
                    'ess_discharge_cost': distflow_results.get('ess_discharge_cost', 0),
                    'sop_loss_cost': distflow_results.get('sop_loss_cost', 0),
                    'slack_penalty': slack_penalty,
                    'sop_slack_penalty': sop_slack_penalty,
                    'nop_slack_penalty': nop_slack_penalty,
                    'distflow_penalty': distflow_penalty,
                    'voltages_stage1': voltages_stage1,
                    'voltages_stage2': voltages_stage1,
                    'line_powers_stage1': distflow_results.get('line_powers', {}),
                    'line_powers_stage2': {},
                    'pvw_power_pu': physical_actions.get('pvw_power', []),
                    'ev_power_kw': physical_actions.get('ev_power', []),
                    'opendss_loss_W': 0.0,
                    'total_cost': total_cost,
                    'total_objective_cost': total_objective_cost,
                    'voltage_penalty_unscaled': 0.0,
                    'opendss_failure_penalty_unscaled': failure_penalty,
                    'base_reward_unscaled': base_reward,
                    'reward_mode': self.reward_mode,
                    'opendss_failed': True,
                }

                if self.reward_mode == "station_operator":
                    reward_unscaled = self._calculate_station_operator_reward(
                        physical_actions=physical_actions,
                        price_now=self.price[self.current_step],
                        info=info
                    )
                else:
                    reward_unscaled = base_reward
        else:
            optimization_results = self._solve_optimal_flow_for_step(physical_actions)
            cost = optimization_results.get('total_cost', 1e7)
            slack_penalty = optimization_results.get('slack_penalty', 0.0)
            sop_slack_penalty = optimization_results.get('sop_slack_penalty', 0.0)
            nop_slack_penalty = optimization_results.get('nop_slack_penalty', 0.0)
            distflow_penalty = slack_penalty + sop_slack_penalty + nop_slack_penalty
            total_objective_cost = cost + distflow_penalty
            voltages = optimization_results.get('voltages', {})
            # In single stage, the basic reward is also used as "unscaled reward"
            voltage_penalty = self._get_voltage_penalty(voltages)
            base_reward = -cost * self.reward_weights["cost_penalty_factor"] + voltage_penalty

            info = {
                'grid_purchase_cost': optimization_results.get('grid_purchase_cost', 0),
                'generation_cost': optimization_results.get('generation_cost', 0),
                'ess_discharge_cost': optimization_results.get('ess_discharge_cost', 0),
                'sop_loss_cost': optimization_results.get('sop_loss_cost', 0),
                'total_cost': cost,
                'total_objective_cost': total_objective_cost,
                'slack_penalty': slack_penalty,
                'sop_slack_penalty': sop_slack_penalty,
                'nop_slack_penalty': nop_slack_penalty,
                'distflow_penalty': distflow_penalty,
                'voltage_penalty_unscaled': voltage_penalty,
                'opendss_failure_penalty_unscaled': 0.0,
                'base_reward_unscaled': base_reward,
                'reward_mode': self.reward_mode,
                'ev_power_kw': physical_actions.get('ev_power', []),
                'opendss_failed': False,
            }

            if self.reward_mode == "station_operator":
                reward_unscaled = self._calculate_station_operator_reward(
                    physical_actions=physical_actions,
                    price_now=self.price[self.current_step],
                    info=info
                )
            else:
                reward_unscaled = base_reward

        info['reconfiguration_plan'] = getattr(self.grid, 'reconfiguration_plan', {}) or {}

        # --------- EV SOC & charging penalty part, still accumulated on the "unscaled reward" ----------
        urgency_penalty = self._calculate_ev_urgency_penalty(physical_actions)
        info['ev_urgency_penalty'] = urgency_penalty
        
        self._update_bocs(physical_actions)

        departure_penalty = self._check_departures_and_get_reward()
        info['ev_shortage_penalty_unscaled'] = departure_penalty

        if self.reward_mode == "station_operator":
            if self.station_operator_cfg.get("include_penalty_cost", True):
                reward_unscaled += departure_penalty
                reward_unscaled += urgency_penalty
        else:
            reward_unscaled += departure_penalty
            reward_unscaled += urgency_penalty

        info["step_ev_shortage_penalty_cost"] = max(0.0, -departure_penalty)

        reward = reward_unscaled * self.reward_scale
        info["reward_unscaled"] = reward_unscaled

        step_diagnostics = self._build_step_diagnostics(info, physical_actions)
        info.update(step_diagnostics)

        self._update_episode_metrics(info)

        will_terminate = (self.current_step + 1) >= self.total_timesteps
        if will_terminate:
            info.update(self._finalize_episode_metrics())

        self.current_step += 1
        terminated = self.current_step >= self.total_timesteps
        truncated = False
        observation = self._get_observation() if not terminated else np.zeros(self.observation_space.shape)

        return observation, reward, terminated, truncated, info


    def _update_grid_for_opendss(self, physical_actions, distflow_results):
        """
        Helper function: only responsible for updating the output of all power side devices.
        The update on the load side will be completed directly in the step function through temporary replacement.
        """
        # 1. Update conventional generator output (optimization results from the first stage)
        gen_powers_pu = distflow_results.get('generation_powers', {})
        gen_q_powers_pu = distflow_results.get('generation_q_powers', {})
        for gen in self.grid.Gens:
            if gen.ID in gen_powers_pu:
                gen._p = gen_powers_pu[gen.ID]
                gen._q = gen_q_powers_pu.get(gen.ID, 0.0)

        # 2. Update photovoltaic/wind power output
        pvw_powers_pu = physical_actions.get('pvw_power', [])
        for i, pvw in enumerate(self.pvw_list):
            pvw._pr = pvw_powers_pu[i]
            tan_phi = (1 - pvw.PF**2)**0.5 / pvw.PF if pvw.PF != 0 else 0
            pvw._qr = pvw._pr * tan_phi

        # 3. Update energy storage output
        ess_powers_pu = physical_actions.get('ess_power', [])
        for i, ess in enumerate(self.ess_list):
            ess.P = ess_powers_pu[i]

    def _get_voltage_penalty(self, voltages):
        standards = EVALUATION_CONFIG["standards"]
        violations = sum(
            1 for v in voltages.values()
            if not (standards["voltage_min_pu"] <= v <= standards["voltage_max_pu"])
        )
        return violations * self.reward_weights["voltage_violation_penalty"]

    def _calculate_base_reward(self, cost, voltages):
        """Power Grid Operator Mode: System Total Cost + Voltage Penalty"""
        return -cost * self.reward_weights["cost_penalty_factor"] + self._get_voltage_penalty(voltages)

    def _calculate_station_operator_reward(self, physical_actions, price_now, info):
        """
        Charging station operator model:
        r_ops,t = Σ[(π_uc - π_t) * P_charge * η_c + (π_t - π_ud) * P_discharge / η_d] * Δt - C_p
        """
        cfg = self.station_operator_cfg
        step_h = self.params['step_minutes'] / 60.0

        ev_powers_kw = np.asarray(physical_actions.get('ev_power', []), dtype=float)
        charge_kw = np.clip(ev_powers_kw, 0.0, None)
        discharge_kw = np.clip(-ev_powers_kw, 0.0, None)

        pi_uc = float(cfg.get("charge_service_price", 1.20))
        pi_ud = float(cfg.get("v2g_subsidy_price", 0.80))
        eta_c = max(self.ev_params.charge_efficiency, 1e-6)
        eta_d = max(self.ev_params.discharge_efficiency, 1e-6)

        charge_profit = float(np.sum((pi_uc - price_now) * charge_kw * eta_c * step_h))
        discharge_profit = float(np.sum((price_now - pi_ud) * discharge_kw / eta_d * step_h))
        gross_profit = charge_profit + discharge_profit

        extra_cost = 0.0
        if cfg.get("include_grid_cost", False):
            extra_cost += info.get("grid_purchase_cost", 0.0)
        if cfg.get("include_generation_cost", True):
            extra_cost += info.get("generation_cost", 0.0)
        if cfg.get("include_ess_cost", True):
            extra_cost += info.get("ess_discharge_cost", 0.0)
        if cfg.get("include_sop_loss_cost", True):
            extra_cost += info.get("sop_loss_cost", 0.0)

        penalty_cost = 0.0
        if cfg.get("include_penalty_cost", True):
            penalty_cost += -info.get("voltage_penalty_unscaled", 0.0)
            penalty_cost += -info.get("opendss_failure_penalty_unscaled", 0.0)

        net_profit = gross_profit - extra_cost - penalty_cost

        info["charge_service_profit"] = charge_profit
        info["v2g_spread_profit"] = discharge_profit
        info["station_gross_profit"] = gross_profit
        info["station_extra_cost"] = extra_cost
        info["station_penalty_cost"] = penalty_cost
        info["station_net_profit"] = net_profit

        return net_profit

    def _prepare_ev_simulation_data(self):
        """(Multiple charging station version) Prepare simulation data that aggregates EV information of all charging stations."""
        self.ev_present = np.zeros((self.total_spots, self.total_timesteps), dtype=int)
        self.ev_boc = np.zeros((self.total_spots, self.total_timesteps + 1), dtype=np.float32)
        self.active_session_map = {}

        steps_per_hour = 60 // self.params['step_minutes']
        start_hour = self.params['start_hour']
        end_hour = self.params['end_hour']

        spot_counter = 0
        for station in self.stations_list:
            for session in station.daily_sessions:
                global_spot_id = spot_counter + session.spot_id

                # Calculation start and end
                if not (session.arrival_hour < end_hour and session.departure_hour > start_hour):
                    continue

                effective_arrival_hour = max(session.arrival_hour, start_hour)
                effective_departure_hour = min(session.departure_hour, end_hour)

                start_step = int(round((effective_arrival_hour - start_hour) * steps_per_hour))
                end_step = int(round((effective_departure_hour - start_hour) * steps_per_hour))

                # start_step ∈ [0, T-1]
                start_step = max(0, min(start_step, self.total_timesteps - 1))
                # end_step ∈ [0, T], allowed to be equal to T to mean "departure after the end of simulation"
                end_step = max(0, min(end_step, self.total_timesteps))

                # Skip if there is no valid stay time
                if end_step <= start_step:
                    continue

                # Last present time step (index)
                last_active_step = end_step - 1  # ∈ [0, T-1]

                # During the period [start_step, end_step), the car is on this spot
                for t in range(start_step, end_step):
                    self.active_session_map[(global_spot_id, t)] = {
                        "initial_soc": session.initial_soc,
                        "arrival_step": start_step,
                        "departure_step": last_active_step, # Uniformly use "last active step" as departure_step
                    }

                # mark presence matrix
                self.ev_present[global_spot_id, start_step:end_step] = 1

            spot_counter += station.num_spots

        # Initialize the SOC at t=0: set the initial SOC for those EVs that are already present at the beginning
        self.ev_boc[:, 0] = 0.0
        for (spot, step), info in self.active_session_map.items():
            if step == 0:
                self.ev_boc[spot, 0] = info['initial_soc']

    def _get_effective_ev_soc(self, spot_id: int, step: int, session_info=None) -> float:
        if step >= self.total_timesteps:
            return 0.0
        if session_info is None:
            session_info = self.active_session_map.get((spot_id, step))
        if session_info and step == session_info.get("arrival_step"):
            return float(np.clip(session_info["initial_soc"], 0.0, 1.0))
        return float(np.clip(self.ev_boc[spot_id, step], 0.0, 1.0))

    def _get_observation(self):
        t = self.current_step
        time_in_seconds = t * self.params['step_minutes'] * 60
        obs_list = [t / self.total_timesteps, self._scale_price(self.price[t])]
        obs_list.extend([self._scale_power_kw(b.Pd(time_in_seconds)) for b in self.grid.Buses])
        obs_list.extend([self._scale_power_kw(b.Qd(time_in_seconds)) for b in self.grid.Buses])
        obs_list.extend([self._scale_power_kw(pvw.P(time_in_seconds)) for pvw in self.pvw_list])
        obs_list.extend([ess.SOC for ess in self.ess_list])

        for i in range(self.total_spots):
            if t < self.total_timesteps:
                is_present = self.ev_present[i, t]
                session_info = self.active_session_map.get((i, t))
                if session_info:
                    current_soc = self._get_effective_ev_soc(i, t, session_info)
                    target_soc = EVALUATION_CONFIG["standards"]["ev_charged_soc_threshold"]
                    steps_left = max(1, session_info["departure_step"] - t + 1)
                    time_to_departure = steps_left / self.total_timesteps
                    energy_requested = max(0.0, target_soc - current_soc)
                else:
                    current_soc = 0
                    time_to_departure = 0
                    energy_requested = 0
            else:
                is_present, current_soc, time_to_departure, energy_requested = 0, 0, 0, 0
            obs_list.extend([is_present, current_soc, time_to_departure, energy_requested])

        return np.array(obs_list, dtype=np.float32)

    def _calculate_ev_urgency_penalty(self, physical_actions):
        enable_urgency = bool(
            self.params.get(
                "enable_ev_urgency_penalty",
                CORE_PARAMS.get("enable_ev_urgency_penalty", True)
            )
        )
        if not enable_urgency:
            return 0.0

        dense_gap_penalty = abs(float(
            self.params.get(
                "ev_dense_gap_penalty",
                CORE_PARAMS.get("ev_dense_gap_penalty", 2.0)
            )
        ))
        step_hours = self.params['step_minutes'] / 60.0
        urgency_penalty = 0.0
        target_soc = EVALUATION_CONFIG["standards"]["ev_charged_soc_threshold"]

        for i in range(self.total_spots):
            session_info = self.active_session_map.get((i, self.current_step))
            if self.ev_present[i, self.current_step] and session_info:
                current_soc = self._get_effective_ev_soc(i, self.current_step, session_info)
                if current_soc >= target_soc:
                    continue
                
                soc_deficit = target_soc - current_soc
                deficit_kwh = soc_deficit * self.ev_params.capacity_kwh
                
                departure_step = session_info["departure_step"]
                steps_left = max(1, departure_step - self.current_step + 1)
                
                required_kwh_this_step = deficit_kwh / steps_left
                
                ev_charge_power_kw = physical_actions['ev_power'][i]
                actual_charge_kwh = max(0.0, ev_charge_power_kw) * step_hours * self.ev_params.charge_efficiency
                
                gap_kwh = max(0.0, required_kwh_this_step - actual_charge_kwh)
                urgency_weight = 1.0 + min(1.0, 1.0 / steps_left)
                
                urgency_penalty += -dense_gap_penalty * urgency_weight * gap_kwh
                
        return urgency_penalty

    def _check_departures_and_get_reward(self):
        """
        Check vehicles leaving the site at current_step and apply penalties based on under-charge (kWh).
        """
        reward = 0.0
        visited_spots = set() # Prevent the same vehicle from being processed repeatedly

        for (spot_id, t), session_info in self.active_session_map.items():
            # Only settle once when "last moment present + current step" is satisfied at the same time
            if (
                    session_info["departure_step"] == self.current_step
                    and t == self.current_step
                    and spot_id not in visited_spots
            ):
                # SOC after the end of this step (already plus the charge/discharge of the current step)
                final_soc = self.ev_boc[spot_id, self.current_step + 1]

                ev_target_soc = EVALUATION_CONFIG["standards"]["ev_charged_soc_threshold"]
                if final_soc < ev_target_soc:
                    kwh_shortage = (ev_target_soc - final_soc) * self.ev_params.capacity_kwh
                    penalty = kwh_shortage * self.reward_weights["ev_kwh_shortage_penalty"]
                    reward += penalty
                visited_spots.add(spot_id)

        return reward

    def _update_bocs(self, physical_actions):
        step_hr = self.params['step_minutes'] / 60.0
        next_ev_boc = np.zeros_like(self.ev_boc[:, self.current_step])

        for i in range(self.total_spots):
            session_info = self.active_session_map.get((i, self.current_step))
            if self.ev_present[i, self.current_step]:
                current_soc = self._get_effective_ev_soc(i, self.current_step, session_info)
                power_kw = physical_actions['ev_power'][i]
                effic = self.ev_params.charge_efficiency if power_kw >= 0 else 1 / self.ev_params.discharge_efficiency
                energy_change = (power_kw * step_hr * effic) / self.ev_params.capacity_kwh
                next_ev_boc[i] = current_soc + energy_change

        self.ev_boc[:, self.current_step + 1] = np.clip(next_ev_boc, 0, 1)

        for i, ess in enumerate(self.ess_list):
            power_pu = physical_actions['ess_power'][i]
            effic = ess.EC if power_pu > 0 else 1 / ess.ED
            energy_change_puh = power_pu * step_hr * effic
            ess._elec = np.clip(ess._elec + energy_change_puh, 0, ess.Cap)

    def _build_step_diagnostics(self, info, physical_actions):
        diagnostics = {}

        # Cost item
        diagnostics["step_total_cost"] = float(info.get("total_cost", 0.0))
        diagnostics["step_total_objective_cost"] = float(info.get("total_objective_cost", 0.0))
        diagnostics["step_grid_purchase_cost"] = float(info.get("grid_purchase_cost", 0.0))
        diagnostics["step_generation_cost"] = float(info.get("generation_cost", 0.0))
        diagnostics["step_ess_discharge_cost"] = float(info.get("ess_discharge_cost", 0.0))
        diagnostics["step_sop_loss_cost"] = float(info.get("sop_loss_cost", 0.0))

        # DistFlow Relaxation penalty is originally a positive cost
        diagnostics["step_slack_penalty"] = float(info.get("slack_penalty", 0.0))
        diagnostics["step_sop_slack_penalty"] = float(info.get("sop_slack_penalty", 0.0))
        diagnostics["step_nop_slack_penalty"] = float(info.get("nop_slack_penalty", 0.0))
        diagnostics["step_distflow_penalty"] = float(info.get("distflow_penalty", 0.0))

        # The voltage penalty is a negative number in reward, which is converted into a positive cost here.
        voltage_penalty = float(info.get("voltage_penalty_unscaled", 0.0))
        diagnostics["step_voltage_penalty_cost"] = max(0.0, -voltage_penalty)

        standards = EVALUATION_CONFIG["standards"]
        voltages = info.get("voltages_stage2", info.get("voltages_stage1", {})) or {}
        diagnostics["step_voltage_violation_count"] = float(
            sum(
                1 for v in voltages.values()
                if not (standards["voltage_min_pu"] <= v <= standards["voltage_max_pu"])
            )
        )

        # OpenDSS failure penalty
        opendss_failure_penalty = float(info.get("opendss_failure_penalty_unscaled", 0.0))
        diagnostics["step_opendss_failure_penalty_cost"] = max(0.0, -opendss_failure_penalty)
        diagnostics["step_opendss_failure_count"] = 1.0 if info.get("opendss_failed", False) else 0.0

        # EV charging and discharging capacity
        step_h = self.params["step_minutes"] / 60.0
        ev_powers_kw = physical_actions.get("ev_power", []) or []
        diagnostics["step_ev_charge_kwh"] = float(
            sum(max(0.0, p) * step_h for p in ev_powers_kw)
        )
        diagnostics["step_ev_discharge_kwh"] = float(
            sum(max(0.0, -p) * step_h for p in ev_powers_kw)
        )

        # PV/Wind consumption and abandonment
        time_in_seconds = self.current_step * self.params["step_minutes"] * 60
        pvw_available = []
        for pvw in self.pvw_list:
            try:
                pvw_available.append(float(pvw.P(time_in_seconds)))
            except Exception:
                pvw_available.append(0.0)

        pvw_actual = [float(x) for x in physical_actions.get("pvw_power", []) or []]
        pvw_available_sum = float(sum(pvw_available))
        pvw_actual_sum = float(sum(pvw_actual))
        diagnostics["step_pvw_available_pu"] = pvw_available_sum
        diagnostics["step_pvw_actual_pu"] = pvw_actual_sum
        diagnostics["step_pvw_curtailment_pu"] = max(0.0, pvw_available_sum - pvw_actual_sum)

        # SOP action projection diagnosis, written by _apply_action self.last_action_diagnostics
        action_diag = getattr(self, "last_action_diagnostics", {}) or {}
        diagnostics["step_sop_projection_count"] = float(action_diag.get("sop_projection_count", 0.0))
        diagnostics["step_sop_projection_scale_sum"] = float(action_diag.get("sop_projection_scale_sum", 0.0))
        diagnostics["step_sop_projection_scale_min"] = float(action_diag.get("sop_projection_scale_min", 1.0))
        diagnostics["step_sop_pre_projection_norm_max"] = float(action_diag.get("sop_pre_projection_norm_max", 0.0))

        return diagnostics

    def _update_episode_metrics(self, info):
        m = self.episode_metrics

        m["ep_reward_unscaled"] += float(info.get("reward_unscaled", 0.0))
        m["ep_total_cost"] += float(info.get("step_total_cost", 0.0))
        m["ep_total_objective_cost"] += float(info.get("step_total_objective_cost", 0.0))
        m["ep_grid_purchase_cost"] += float(info.get("step_grid_purchase_cost", 0.0))
        m["ep_generation_cost"] += float(info.get("step_generation_cost", 0.0))
        m["ep_ess_discharge_cost"] += float(info.get("step_ess_discharge_cost", 0.0))
        m["ep_sop_loss_cost"] += float(info.get("step_sop_loss_cost", 0.0))

        m["ep_distflow_penalty"] += float(info.get("step_distflow_penalty", 0.0))
        m["ep_slack_penalty"] += float(info.get("step_slack_penalty", 0.0))
        m["ep_sop_slack_penalty"] += float(info.get("step_sop_slack_penalty", 0.0))
        m["ep_nop_slack_penalty"] += float(info.get("step_nop_slack_penalty", 0.0))

        m["ep_voltage_penalty_cost"] += float(info.get("step_voltage_penalty_cost", 0.0))
        m["ep_voltage_violation_count"] += float(info.get("step_voltage_violation_count", 0.0))
        m["ep_opendss_failure_penalty_cost"] += float(info.get("step_opendss_failure_penalty_cost", 0.0))
        m["ep_opendss_failure_count"] += float(info.get("step_opendss_failure_count", 0.0))

        m["ep_ev_shortage_penalty_cost"] += float(info.get("step_ev_shortage_penalty_cost", 0.0))
        m["ep_ev_urgency_penalty"] += float(info.get("ev_urgency_penalty", 0.0))
        m["ep_ev_charge_kwh"] += float(info.get("step_ev_charge_kwh", 0.0))
        m["ep_ev_discharge_kwh"] += float(info.get("step_ev_discharge_kwh", 0.0))

        m["ep_pvw_available_pu"] += float(info.get("step_pvw_available_pu", 0.0))
        m["ep_pvw_actual_pu"] += float(info.get("step_pvw_actual_pu", 0.0))
        m["ep_pvw_curtailment_pu"] += float(info.get("step_pvw_curtailment_pu", 0.0))

        m["ep_sop_projection_count"] += float(info.get("step_sop_projection_count", 0.0))
        m["ep_sop_projection_scale_sum"] += float(info.get("step_sop_projection_scale_sum", 0.0))
        m["ep_sop_projection_scale_min"] = min(
            float(m.get("ep_sop_projection_scale_min", 1.0)),
            float(info.get("step_sop_projection_scale_min", 1.0)),
        )
        m["ep_sop_pre_projection_norm_max"] = max(
            float(m.get("ep_sop_pre_projection_norm_max", 0.0)),
            float(info.get("step_sop_pre_projection_norm_max", 0.0)),
        )

    def _finalize_episode_metrics(self):
        final_metrics = dict(self.episode_metrics)

        available = final_metrics.get("ep_pvw_available_pu", 0.0)
        actual = final_metrics.get("ep_pvw_actual_pu", 0.0)
        if available > 1e-9:
            final_metrics["ep_pvw_utilization_rate"] = actual / available
        else:
            final_metrics["ep_pvw_utilization_rate"] = 0.0

        projection_count = final_metrics.get("ep_sop_projection_count", 0.0)
        if projection_count > 1e-9:
            final_metrics["ep_sop_projection_scale_avg"] = (
                final_metrics.get("ep_sop_projection_scale_sum", 0.0) / projection_count
            )
        else:
            final_metrics["ep_sop_projection_scale_avg"] = 1.0

        return final_metrics

    def _project_sop_actions(self, p_actions, q1_actions, q2_actions):
        cfg = RL_ENV_CONFIG.get("action_projection", {}) or {}
        enabled = bool(cfg.get("enable_sop_capacity_projection", True))
        margin = float(cfg.get("sop_capacity_projection_margin", 0.98))
        margin = float(np.clip(margin, 0.1, 1.0))

        p_actions = np.asarray(p_actions, dtype=np.float32).copy()
        q1_actions = np.asarray(q1_actions, dtype=np.float32).copy()
        q2_actions = np.asarray(q2_actions, dtype=np.float32).copy()

        diag = {
            "sop_projection_count": 0.0,
            "sop_projection_scale_sum": 0.0,
            "sop_projection_scale_min": 1.0,
            "sop_pre_projection_norm_max": 0.0,
        }

        if not enabled or len(self.sop_list) == 0:
            return p_actions, q1_actions, q2_actions, diag

        for i, sop in enumerate(self.sop_list):
            if not sop.active:
                continue

            p = float(np.clip(p_actions[i], -1.0, 1.0))
            q1 = float(np.clip(q1_actions[i], -1.0, 1.0))
            q2 = float(np.clip(q2_actions[i], -1.0, 1.0))

            norm1 = float(np.sqrt(p * p + q1 * q1))
            norm2 = float(np.sqrt(p * p + q2 * q2))
            norm_max = max(norm1, norm2)

            diag["sop_pre_projection_norm_max"] = max(
                diag["sop_pre_projection_norm_max"],
                norm_max,
            )

            if norm_max > margin and norm_max > 1e-9:
                scale = margin / norm_max
                p *= scale
                q1 *= scale
                q2 *= scale

                diag["sop_projection_count"] += 1.0
                diag["sop_projection_scale_sum"] += scale
                diag["sop_projection_scale_min"] = min(
                    diag["sop_projection_scale_min"],
                    scale,
                )

            p_actions[i] = p
            q1_actions[i] = q1
            q2_actions[i] = q2

        return p_actions, q1_actions, q2_actions, diag

    def _apply_action(self, action: np.ndarray) -> dict:
        self.last_action_diagnostics = {
            "sop_projection_count": 0.0,
            "sop_projection_scale_sum": 0.0,
            "sop_projection_scale_min": 1.0,
            "sop_pre_projection_norm_max": 0.0,
        }
        physical_actions = {}
        action_copy = np.copy(action)
        idx = 0

        # --- Apply mask to EV charging action ---
        ev_action_raw = action_copy[idx : idx + self.total_spots]
        if self.current_step < self.total_timesteps:
            presence_mask = self.ev_present[:, self.current_step]
            ev_action_masked = ev_action_raw * presence_mask
        else:
            ev_action_masked = np.zeros_like(ev_action_raw)
        action_copy[idx : idx + self.total_spots] = ev_action_masked
        idx += self.total_spots

        # --- Action decoding of ESS and PV/Wind ---
        idx += len(self.ess_list)
        idx += len(self.pvw_list)

        # Add action mask to SOP
        # If a certain SOP is set to inactive (active=False) in the configuration, force its P and Q actions at both ends to be 0 to prevent the RL agent from invalidly exploring it.
        sop_p_start_idx = idx
        sop_q1_start_idx = idx + len(self.sop_list)
        sop_q2_start_idx = idx + 2 * len(self.sop_list)
        for i, sop in enumerate(self.sop_list):
            if not sop.active:
                action_copy[sop_p_start_idx + i] = 0.0
                action_copy[sop_q1_start_idx + i] = 0.0
                action_copy[sop_q2_start_idx + i] = 0.0
        # Note: The action of NOP is to determine "whether it is closed", which is the decision itself, so the mask here does not apply.

        # --- Decode all physics actions from the processed action_copy array ---
        # Reset idx to start decoding from scratch
        idx = 0
        ev_action = action_copy[idx: idx + self.total_spots];
        idx += self.total_spots
        ess_action = action_copy[idx: idx + len(self.ess_list)];
        idx += len(self.ess_list)
        pvw_action = action_copy[idx: idx + len(self.pvw_list)];
        idx += len(self.pvw_list)
        sop_p_action = action_copy[idx: idx + len(self.sop_list)];
        idx += len(self.sop_list)
        sop_q1_action = action_copy[idx: idx + len(self.sop_list)];
        idx += len(self.sop_list)
        sop_q2_action = action_copy[idx: idx + len(self.sop_list)];
        idx += len(self.sop_list)
        nop_action = action_copy[idx: idx + len(self.nop_list)]

        ev_power_list = []
        for i, a in enumerate(ev_action):
            power = 0.0
            if self.current_step < self.total_timesteps and self.ev_present[i, self.current_step] == 1:
                current_soc = self._get_effective_ev_soc(i, self.current_step)
                step_hr = self.params['step_minutes'] / 60.0
                max_charge_by_soc = (1.0 - current_soc) * self.ev_params.capacity_kwh / (
                    self.ev_params.charge_efficiency * step_hr
                )
                max_discharge_by_soc = current_soc * self.ev_params.capacity_kwh * (
                    self.ev_params.discharge_efficiency / step_hr
                )
                max_charge_kw = min(self.ev_params.max_charge_kw, max(0.0, max_charge_by_soc))
                max_discharge_kw = min(self.ev_params.max_discharge_kw, max(0.0, max_discharge_by_soc))
                power = a * self.ev_params.max_charge_kw if a > 0 else a * self.ev_params.max_discharge_kw
                power = np.clip(power, -max_discharge_kw, max_charge_kw)
            ev_power_list.append(power)

        physical_actions['ev_power'] = ev_power_list
        ess_power_list = []
        step_hr = self.params['step_minutes'] / 60.0
        for a, ess in zip(ess_action, self.ess_list):
            target_power = a * ess.MaxPc if a > 0 else a * ess.MaxPd
            if step_hr > 0:
                charge_eff = max(float(ess.EC), 1e-9)
                discharge_eff = max(float(ess.ED), 1e-9)
                stored_energy = float(np.clip(ess._elec, 0.0, ess.Cap))
                remaining_capacity = max(0.0, ess.Cap - stored_energy)
                max_charge_by_soc = remaining_capacity / (charge_eff * step_hr)
                max_discharge_by_soc = stored_energy * discharge_eff / step_hr
                max_charge_pu = min(ess.MaxPc, max(0.0, max_charge_by_soc))
                max_discharge_pu = min(ess.MaxPd, max(0.0, max_discharge_by_soc))
                target_power = float(np.clip(target_power, -max_discharge_pu, max_charge_pu))
            ess_power_list.append(target_power)

        physical_actions['ess_power'] = ess_power_list
        physical_actions['pvw_power'] = [(1 - a) * pvw.P(self.current_step * self.params['step_minutes'] * 60) for
                                         a, pvw in zip(pvw_action, self.pvw_list)]
        sop_p_action, sop_q1_action, sop_q2_action, sop_projection_diag = self._project_sop_actions(
            sop_p_action,
            sop_q1_action,
            sop_q2_action,
        )

        self.last_action_diagnostics = sop_projection_diag

        physical_actions["sop_p_power"] = [
            float(a) * sop.PMax
            for a, sop in zip(sop_p_action, self.sop_list)
        ]
        physical_actions["sop_q1_power"] = [
            float(a) * sop.QMax
            for a, sop in zip(sop_q1_action, self.sop_list)
        ]
        physical_actions["sop_q2_power"] = [
            float(a) * sop.QMax
            for a, sop in zip(sop_q2_action, self.sop_list)
        ]
        physical_actions["sop_q_power"] = physical_actions["sop_q1_power"]
        physical_actions['nop_status'] = np.round(np.clip(nop_action, 0, 1)).astype(int)
        return physical_actions

    def _solve_optimal_flow_for_step(self, physical_actions: dict) -> dict:
        """
        In a single time step, solve an optimal power flow problem (Linear DistFlow) for the power grid.
        - Receive and fix the decisions of the RL agent (EV, ESS, PV peak clipping, SOP/NOP, etc.).
        - Economic dispatch of conventional generators to meet the remaining load at the lowest cost.
        - Calculate and return the grid status (voltage, cost, etc.) under this decision.
        - Completely aligned with Baseline's single-step optimization logic.
        """
        t = self.current_step
        grid = self.grid
        model = ConcreteModel(name=f"SingleStepOptimalFlow_t{t}")
        sb_mva = self.grid.SB

        # --- 1. Get the element ID list, used to initialize the Pyomo collection ---
        bus_ids = [b.ID for b in grid.Buses]
        line_ids = [l.ID for l in grid.ActiveLines]
        gen_ids = [g.ID for g in grid.Gens if not str(g.ID).startswith("gen_for_")]
        pvw_ids = [p.ID for p in self.pvw_list]
        ess_ids = [e.ID for e in self.ess_list]
        sop_ids = [s.ID for s in self.sop_list]
        nop_ids = [n.ID for n in self.nop_list]
        spot_ids = list(range(self.total_spots))

        # --- 2. Define Pyomo variables (fully aligned baseline.py) ---
        # Purchased power (inflow from upper-level power grid, non-negative)
        model.grid_inflow_p = Var(domain=NonNegativeReals)

        # Conventional generator power (active pg, reactive qg)
        def pg_bounds_rl(m, g_id):
            gen = grid.Gen(g_id)
            pmin = gen.Pmin(t) if callable(gen.Pmin) else gen.Pmin
            # Similarly, read the real upper limit we set directly from the generator object
            pmax = gen.RealisticPmax if hasattr(gen, 'RealisticPmax') else (
                gen.Pmax(t) if callable(gen.Pmax) else gen.Pmax)
            return (pmin, pmax)

        model.pg = Var(gen_ids, bounds=pg_bounds_rl)
        model.qg = Var(gen_ids, bounds=lambda m, g: (grid.Gen(g).Qmin(t), grid.Gen(g).Qmax(t)))

        # Grid state variables (voltage v, line power flow P, Q)
        model.v = Var(bus_ids, bounds=lambda m, b: (grid.Bus(b).MinV, grid.Bus(b).MaxV))
        model.P = Var(line_ids)
        model.Q = Var(line_ids)

        # Device variables controlled by the agent
        model.pspot = Var(spot_ids) # EV charging pile power (pu)
        model.ess_charge = Var(ess_ids, domain=NonNegativeReals)
        model.ess_discharge = Var(ess_ids, domain=NonNegativeReals)
        model.pvw_p = Var(pvw_ids) # PV/Wind actual output (pu)
        model.pvw_q = Var(pvw_ids)
        model.sop_p1 = Var(sop_ids)
        model.sop_q1 = Var(sop_ids)
        model.sop_q2 = Var(sop_ids)
        model.nop_status = Var(nop_ids, domain=Binary)

        # Auxiliary and slack variables
        model.sop_loss = Var(sop_ids, domain=NonNegativeReals)
        model.nop_p = Var(nop_ids)
        model.nop_q = Var(nop_ids)
        model.sop_capacity_slack = Var(sop_ids, domain=NonNegativeReals)
        model.nop_v_slack_pos = Var(nop_ids, domain=NonNegativeReals)
        model.nop_v_slack_neg = Var(nop_ids, domain=NonNegativeReals)
        model.SlackP = Var(bus_ids, initialize=0.0)
        model.SlackQ = Var(bus_ids, initialize=0.0)

        # --- 3. Decision-making of fixed RL agent ---
        # Convert EV power in kW to pu and fix it into the model
        for i in spot_ids:
            model.pspot[i].fix(physical_actions['ev_power'][i] / (sb_mva * 1000.0))

        # Fixed PV/Wind output (taking into account the agent’s reduction decision)
        for i, pid in enumerate(pvw_ids):
            model.pvw_p[pid].fix(physical_actions['pvw_power'][i])

        # Fixed ESS charging/discharging power
        for i, ess_id in enumerate(ess_ids):
            ess_power_action = physical_actions['ess_power'][i]
            if ess_power_action > 0: # Charging
                model.ess_charge[ess_id].fix(ess_power_action)
                model.ess_discharge[ess_id].fix(0)
            else: # discharge
                model.ess_charge[ess_id].fix(0)
                model.ess_discharge[ess_id].fix(-ess_power_action)

        # Fixed SOP transmission power
        sop_q1_actions = physical_actions.get('sop_q1_power', physical_actions.get('sop_q_power', []))
        sop_q2_actions = physical_actions.get('sop_q2_power', physical_actions.get('sop_q_power', []))
        for i, sid in enumerate(sop_ids):
            model.sop_p1[sid].fix(physical_actions['sop_p_power'][i])
            model.sop_q1[sid].fix(sop_q1_actions[i])
            model.sop_q2[sid].fix(sop_q2_actions[i])

        # Fixed NOP switch state
        for i, nid in enumerate(nop_ids):
            model.nop_status[nid].fix(physical_actions['nop_status'][i])

        # --- 4. Add grid physical constraints (fully aligned baseline.py) ---
        @model.Constraint(pvw_ids)
        def pvw_q_rel_rule(m, pvw_id):
            pvw = next(p for p in self.pvw_list if p.ID == pvw_id)
            tan_phi = np.sqrt(1 - pvw.PF ** 2) / pvw.PF if pvw.PF != 0 else 0
            return m.pvw_q[pvw_id] == m.pvw_p[pvw_id] * tan_phi

        def ess_q_coeff(ess):
            return float(np.sqrt(max(0.0, 1.0 - ess.PF ** 2)))

        # 4.1 Active power balance constraints
        @model.Constraint(bus_ids, doc="Active power balance constraints (multiple charging station version)")
        def p_balance_rule(m, bus_id):
            time_in_seconds = t * self.params['step_minutes'] * 60

            # Injections (Sources of Power)
            power_injections = (
                    sum(m.P[l.ID] for l in grid.LinesOfTBus(bus_id, only_active=True)) +
                    sum(m.pg[g.ID] for g in grid.GensAtBus(bus_id) if g.ID in gen_ids)+
                    sum(m.pvw_p[p.ID] for p in self.pvw_list if p.BusID == bus_id) +
                    sum(m.ess_discharge[e.ID] for e in self.ess_list if e.BusID == bus_id) +
                    sum(m.nop_p[n.ID] for n in self.nop_list if n.Bus2 == bus_id) +
                    sum(m.sop_p1[s.ID] - m.sop_loss[s.ID] for s in self.sop_list if s.Bus2 == bus_id) +
                    sum(-m.pspot[spot_idx] for spot_idx, connected_bus in self.spot_to_bus_map.items()
                        if connected_bus == bus_id and physical_actions['ev_power'][spot_idx] < 0) # V2G discharge
            )

            # Sinks of Power / Loads
            power_ejections = (
                    sum(m.P[l.ID] for l in grid.LinesOfFBus(bus_id, only_active=True)) +
                    grid.Bus(bus_id).Pd(time_in_seconds) +
                    sum(m.ess_charge[e.ID] for e in self.ess_list if e.BusID == bus_id) +
                    sum(m.nop_p[n.ID] for n in self.nop_list if n.Bus1 == bus_id) +
                    sum(m.sop_p1[s.ID] for s in self.sop_list if s.Bus1 == bus_id) +
                    sum(m.pspot[spot_idx] for spot_idx, connected_bus in self.spot_to_bus_map.items()
                        if connected_bus == bus_id and physical_actions['ev_power'][spot_idx] >= 0) # EV charging
            )

            # Balance equation
            if bus_id == self.params['slack_bus']:
                return power_injections + m.grid_inflow_p == power_ejections
            else:
                return power_injections + m.SlackP[bus_id] == power_ejections

        # 4.2 Reactive power balance constraints
        @model.Constraint(bus_ids, doc="Reactive power balance constraints")
        def q_balance_rule(m, bus_id):
            time_in_seconds = t * self.params['step_minutes'] * 60
            q_injections = (
                    sum(m.Q[l.ID] for l in grid.LinesOfTBus(bus_id, only_active=True)) +
                    sum(m.qg[g.ID] for g in grid.GensAtBus(bus_id) if g.ID in gen_ids) +
                    sum(m.pvw_q[p.ID] for p in self.pvw_list if p.BusID == bus_id) +
                    sum(m.ess_discharge[e.ID] * ess_q_coeff(e) for e in self.ess_list if e.BusID == bus_id) +
                    sum(m.sop_q1[s.ID] for s in self.sop_list if s.Bus1 == bus_id) +
                    sum(m.sop_q2[s.ID] for s in self.sop_list if s.Bus2 == bus_id) +
                    sum(m.nop_q[n.ID] for n in self.nop_list if n.Bus2 == bus_id)
            )
            q_ejections = (
                    sum(m.Q[l.ID] for l in grid.LinesOfFBus(bus_id, only_active=True)) +
                    grid.Bus(bus_id).Qd(time_in_seconds) +
                    sum(m.ess_charge[e.ID] * ess_q_coeff(e) for e in self.ess_list if e.BusID == bus_id) +
                    sum(m.nop_q[n.ID] for n in self.nop_list if n.Bus1 == bus_id)
            )
            return q_injections + m.SlackQ[bus_id] == q_ejections

        # 4.3 Voltage drop constraint (DistFlow)
        @model.Constraint(line_ids, doc="Voltage drop constraint")
        def v_update_rule(m, line_id):
            line = grid.Line(line_id)
            return m.v[line.tBus] == m.v[line.fBus] - (line.R * m.P[line_id] + line.X * m.Q[line_id])

        # 4.4 SOP/NOP and other constraints (consistent with baseline.py)
        # SOP capacity constraints
        @model.Constraint(sop_ids)
        def sop_capacity_bus1_rule(m, sop_id):
            sop = self.sop_list[sop_ids.index(sop_id)]
            terms = []
            if sop.PMax > 0:
                terms.append((m.sop_p1[sop_id] / sop.PMax) ** 2)
            if sop.QMax > 0:
                terms.append((m.sop_q1[sop_id] / sop.QMax) ** 2)
            if terms:
                return sum(terms) <= 1 + m.sop_capacity_slack[sop_id]
            return Constraint.Skip

        @model.Constraint(sop_ids)
        def sop_capacity_bus2_rule(m, sop_id):
            sop = self.sop_list[sop_ids.index(sop_id)]
            terms = []
            if sop.PMax > 0:
                terms.append(((m.sop_p1[sop_id] - m.sop_loss[sop_id]) / sop.PMax) ** 2)
            if sop.QMax > 0:
                terms.append((m.sop_q2[sop_id] / sop.QMax) ** 2)
            if terms:
                return sum(terms) <= 1 + m.sop_capacity_slack[sop_id]
            return Constraint.Skip

        # SOP loss constraint
        @model.Constraint(sop_ids)
        def sop_loss_rule(m, sop_id):
            sop = self.sop_list[sop_ids.index(sop_id)]
            if sop.PMax > 0:
                return m.sop_loss[sop_id] >= sop.LossCoeff * (m.sop_p1[sop_id] ** 2) / sop.PMax ** 2
            return Constraint.Skip

        # NOP big-M logical constraints
        M = 1000

        @model.Constraint(nop_ids)
        def nop_p_limit1_rule(m, nop_id):
            return m.nop_p[nop_id] <= M * m.nop_status[nop_id]

        @model.Constraint(nop_ids)
        def nop_p_limit2_rule(m, nop_id):
            return m.nop_p[nop_id] >= -M * m.nop_status[nop_id]

        @model.Constraint(nop_ids)
        def nop_q_limit1_rule(m, nop_id):
            return m.nop_q[nop_id] <= M * m.nop_status[nop_id]

        @model.Constraint(nop_ids)
        def nop_q_limit2_rule(m, nop_id):
            return m.nop_q[nop_id] >= -M * m.nop_status[nop_id]

        def nop_v_residual(m, nop_id):
            nop = self.nop_list[nop_ids.index(nop_id)]
            return m.v[nop.Bus2] - m.v[nop.Bus1] + (
                nop.R * m.nop_p[nop_id] + nop.X * m.nop_q[nop_id]
            )

        @model.Constraint(nop_ids)
        def nop_v_upper_rule(m, nop_id):
            return nop_v_residual(m, nop_id) <= (
                m.nop_v_slack_pos[nop_id] + M * (1 - m.nop_status[nop_id])
            )

        @model.Constraint(nop_ids)
        def nop_v_lower_rule(m, nop_id):
            return nop_v_residual(m, nop_id) >= (
                -m.nop_v_slack_neg[nop_id] - M * (1 - m.nop_status[nop_id])
            )

        # Line capacity constraints
        LINE_CAPACITY_PU = 5.0

        @model.Constraint(line_ids)
        def line_p_upper_limit_rule(m, line_id):
            return m.P[line_id] <= LINE_CAPACITY_PU

        @model.Constraint(line_ids)
        def line_p_lower_limit_rule(m, line_id):
            return m.P[line_id] >= -LINE_CAPACITY_PU

        model.p_balance_constr = Constraint(bus_ids, rule=p_balance_rule)
        model.q_balance_constr = Constraint(bus_ids, rule=q_balance_rule)
        model.v_update_constr = Constraint(line_ids, rule=v_update_rule)
        model.line_p_upper_limit_constr = Constraint(line_ids, rule=line_p_upper_limit_rule)
        model.line_p_lower_limit_constr = Constraint(line_ids, rule=line_p_lower_limit_rule)

        # Fixed the head-end bus (Slack Bus) voltage to 1.0 pu
        # For single-step optimization, we do not need a rule function and can directly define a simple equality constraint.
        # Note: The model.v variable here has no time index because it only represents the voltage of the current time step.
        model.slack_bus_voltage_constraint = Constraint(
            expr=model.v[CORE_PARAMS["slack_bus"]] == 1.0,
            doc="Fix slack bus voltage to 1.0 pu for the current timestep"
        )


        # --- 5. Define the objective function (minimize single-step cost) ---
        price_now = self.price[t]
        step_duration_h = self.params['step_minutes'] / 60.0

        # Grid Purchase Cost
        grid_purchase_cost = price_now * model.grid_inflow_p * sb_mva * 1000 * step_duration_h

        # Generation Cost (quadratic function)
        generation_cost = sum(
            (grid.Gen(g_id).CostA(t) * (model.pg[g_id] * sb_mva) ** 2 +
             grid.Gen(g_id).CostB(t) * (model.pg[g_id] * sb_mva) +
             grid.Gen(g_id).CostC(t)) * step_duration_h
            for g_id in gen_ids
        )

        # Other costs
        ess_discharge_cost = 0.0 # Align with baseline, assuming there is no additional cost for discharge
        sop_loss_cost = price_now * sum(model.sop_loss[s.ID] for s in self.sop_list) * sb_mva * 1000 * step_duration_h

        # Penalty term, uses the same set of configuration coefficients as Baseline.
        penalties = BASELINE_PARAMS["penalty_factors"]
        slack_penalty = penalties["slack_power_penalty"] * sum(
            model.SlackP[b] ** 2 + model.SlackQ[b] ** 2 for b in bus_ids
        )
        nop_slack_penalty = penalties["nop_voltage_penalty"] * sum(
            model.nop_v_slack_pos[nid] + model.nop_v_slack_neg[nid] for nid in nop_ids
        )
        sop_slack_penalty = penalties["sop_capacity_penalty"] * sum(
            model.sop_capacity_slack[sid] for sid in sop_ids
        )

        model.objective = Objective(
            expr=(grid_purchase_cost + generation_cost + ess_discharge_cost + sop_loss_cost +
                  slack_penalty + nop_slack_penalty + sop_slack_penalty),
            sense=minimize
        )

        # --- 6. Solve and return the result ---
        solver = SolverFactory(self.params['solver'])
        try:
            results = solver.solve(model, tee=False)
            if results.solver.termination_condition == 'optimal' or results.solver.status == 'ok':
                total_cost = value(grid_purchase_cost) + value(generation_cost) + value(ess_discharge_cost) + value(
                    sop_loss_cost)
                slack_penalty_val = value(slack_penalty)
                sop_slack_penalty_val = value(sop_slack_penalty)
                nop_slack_penalty_val = value(nop_slack_penalty)
                objective_value = total_cost + slack_penalty_val + sop_slack_penalty_val + nop_slack_penalty_val
                return {
                    "status": "success",
                    "total_cost": total_cost,
                    "objective_value": objective_value,
                    "grid_purchase_cost": value(grid_purchase_cost),
                    "generation_cost": value(generation_cost),
                    "sop_loss_cost": value(sop_loss_cost),
                    "ess_discharge_cost": value(ess_discharge_cost),
                    "slack_penalty": slack_penalty_val,
                    "sop_slack_penalty": sop_slack_penalty_val,
                    "nop_slack_penalty": nop_slack_penalty_val,
                    "voltages": {b: value(model.v[b]) for b in bus_ids},
                    "line_powers": {l: value(model.P[l]) for l in line_ids},
                    "grid_inflow_p": value(model.grid_inflow_p),
                    "generation_powers": {g: value(model.pg[g]) for g in gen_ids},
                    "generation_q_powers": {g: value(model.qg[g]) for g in gen_ids},
                    "sop_losses_pu": {s: value(model.sop_loss[s]) for s in sop_ids},
                    "slack_powers": {b: value(model.SlackP[b]) for b in bus_ids}
                }
            else:
                # The solver reported a non-optimal solution
                raise Exception(f"Solver failed with status: {results.solver.termination_condition}")
        except Exception as e:
            # Any error occurs (including solution failure)
            print(f"!!!!!! Optimization solution failed at time step {t}: {e} !!!!!!")
            # Return a dictionary containing default values to avoid subsequent code crashes
            return {
                "status": "failed", "total_cost": 1e7, "objective_value": 1e7, "generation_cost": 0,
                "grid_purchase_cost": 1e7, "sop_loss_cost": 0, "ess_discharge_cost": 0,
                "slack_penalty": 0, "sop_slack_penalty": 0, "nop_slack_penalty": 0,
                "voltages": {}, "line_powers": {}, "grid_inflow_p": 0,
                "generation_powers": {}, "generation_q_powers": {}, "slack_powers": {}
            }
