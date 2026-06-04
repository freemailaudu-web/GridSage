"""
This module defines the "Baseline" Algorithm of the platform.
The core function is to use Pyomo to build a global optimal power flow model (based on linearized DistFlow),
Calculate at one time the cost of all controllable equipment (generators, energy storage,
charging piles, SOP/NOP, etc.) optimal scheduling strategy to minimize the total operating cost.
This calculation result is used as the "optimal benchmark" to evaluate the performance of reinforcement learning agents.
"""

import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory
from fpowerkit import Grid
from fpowerkit.solbase import GridSolveResult
from config import CORE_PARAMS, BASELINE_PARAMS, EVALUATION_CONFIG
from grid_model import load_station_info # Import to obtain the bus ID of the electric vehicle charging station
from grid_model import load_electricity_price # Import to get electricity price data
from reconfiguration import fixed_nop_status
from pyomo.environ import Binary, NonNegativeReals
from pyomo.opt import SolverStatus, TerminationCondition
from gev_station import GEVStation
from pyomo.environ import value # Make sure you have this import at the top of the file
def create_baseline_model(grid: Grid, stations_list, time_steps: int, gui_params: dict): # <--- receive station object

    """
    Use Pyomo to create and solve a complete 24-hour linearized DistFlow optimal power flow model.
    Control the power distribution of charging piles, assuming that all charging piles, distributed energy resources, SOP and NOP are controllable, and calculate the global optimal solution.
    """
    # Initialize Pyomo model
    model = ConcreteModel(name="LinearDistFlowBaseline")
    sb_mva = grid.SB
    # 1. At the beginning of the function, get the list of all components from the grid object.
    buses = list(grid.Buses)
    lines = list(grid.ActiveLines)
    pvws = list(grid.PVWinds)
    esss = list(grid.ESSs)
    sops = list(getattr(grid, 'SOPs', {}).values())
    # NOP is already materialized as an active/inactive topology line by the
    # selected reconfiguration plan. It is not a separate P/Q controller here.
    nops = []

    # 2. Perform a one-time, decisive filter on the generator list.
    # This enhanced filtering logic will remove all virtual/equivalent generators prepared for the second phase.
    print("--- [BASELINE FIX] Filtering generators for Stage 1 optimization... ---")
    VIRTUAL_GEN_ID = 'gen_for_slack_bus'
    gens_to_optimize = [
        g for g in grid.Gens
        if g.ID != VIRTUAL_GEN_ID and 'gen_for_sop' not in g.ID
    ]
    print(f"  - Original gen count: {len(list(grid.Gens))}")
    print(f"  - Gens after filtering: {len(gens_to_optimize)}")

    # All subsequent operations are based on this clean `gens_to_optimize` list
    gens = gens_to_optimize

    bus_dict = {bus.ID: idx for idx, bus in enumerate(buses)}
    line_dict = {line.ID: idx for idx, line in enumerate(lines)}
    gen_dict = {gen.ID: idx for idx, gen in enumerate(gens)}
    pvw_dict = {pvw.ID: idx for idx, pvw in enumerate(pvws)}
    ess_dict = {ess.ID: idx for idx, ess in enumerate(esss)}
    sop_dict = {sop.ID: idx for idx, sop in enumerate(sops)}
    nop_dict = {nop.ID: idx for idx, nop in enumerate(nops)}

    # 2.【Aggregation of data from all charging stations】

    # Initialize the list used to aggregate data from all stations
    all_present_cars_list = []
    all_boc_initial_list = []
    all_original_envs = []

    # This is the most critical data structure: a dictionary mapping the "global charging pile index" to its "bus ID"
    spot_to_bus_map = {}
    ev_info = {} #Used to store detailed information of all EVs
    ev_count = 0
    total_spots_count = 0

    # Time conversion parameters
    start_hour = gui_params['start_hour']
    end_hour = gui_params['end_hour']
    step_minutes = gui_params['step_minutes']
    steps_per_hour = 60 // step_minutes

    print("Aggregating and processing scene data of multiple charging stations...")
    for station in stations_list:
        original_env = station.get_scenario_for_baseline()
        all_original_envs.append(original_env)

        # --- a. Resample station_present_cars ---
        original_present_cars = original_env.Invalues['present_cars']
        resampled_present_cars_full_day = np.repeat(original_present_cars, steps_per_hour, axis=1)
        start_step = start_hour * steps_per_hour
        end_step = end_hour * steps_per_hour
        station_present_cars = resampled_present_cars_full_day[:, start_step:end_step]
        all_present_cars_list.append(station_present_cars)

        # --- b. Resampling station_boc_initial ---
        original_boc_initial = original_env.Invalues["BOC"]
        original_time_points = np.arange(original_boc_initial.shape[1])
        new_time_points = np.linspace(start_hour, end_hour, time_steps + 1)
        station_boc_initial = np.zeros((station.num_spots, time_steps + 1))
        for i in range(station.num_spots):
            station_boc_initial[i, :] = np.interp(new_time_points, original_time_points, original_boc_initial[i, :])
        all_boc_initial_list.append(station_boc_initial)

        # --- c. Process ev_info and spot_to_bus_map ---
        original_arrival_times = original_env.Invalues['ArrivalT']
        original_departure_times = original_env.Invalues['DepartureT']

        for local_spot_idx in range(station.num_spots):
            global_spot_idx = total_spots_count + local_spot_idx
            spot_to_bus_map[global_spot_idx] = station.bus_id

            # Traverse all charging sessions on the current charging pile
            for i, (t_arr, t_dep) in enumerate(
                    zip(original_arrival_times[local_spot_idx], original_departure_times[local_spot_idx])):
                # Determine whether the session is within our simulation window
                if t_arr < end_hour and t_dep > start_hour:
                    effective_arrival_hour = max(t_arr, start_hour)
                    effective_departure_hour = min(t_dep, end_hour)

                    # Convert to time step index
                    new_arrival_step = int(round((effective_arrival_hour - start_hour) * steps_per_hour))
                    new_departure_step = int(round((effective_departure_hour - start_hour) * steps_per_hour))

                    # Make sure arrival < departure
                    if new_arrival_step >= new_departure_step:
                        continue

                    # Create a globally unique EV ID
                    ev_id = f"EV_ST{station.station_id}_SP{global_spot_idx}_S{i}"

                    # Get the initial battery level when the car arrives
                    initial_soc_at_arrival = station_boc_initial[
                        local_spot_idx, new_arrival_step] if new_arrival_step < time_steps + 1 else 0

                    # Store in ev_info dictionary
                    ev_info[ev_id] = {
                        "spot": global_spot_idx,
                        "arrival": new_arrival_step,
                        "departure": new_departure_step,
                        "initial_boc": initial_soc_at_arrival
                    }
                    ev_count += 1

        total_spots_count += station.num_spots

        # d. Vertically stack the data of all stations into a large Numpy array
    present_cars = np.vstack(all_present_cars_list)
    boc_initial = np.vstack(all_boc_initial_list)
    print(f"Data aggregation completed, total charging stations: {total_spots_count}, total EV sessions: {ev_count}")

    # (Since the EV physical parameters are the same, we only take the first one as the representative)
    ev_capacity = all_original_envs[0].EV_Param['EV_capacity']
    original_env = all_original_envs[0]

    # Load electricity price data from Excel file ($/puh)
    price = load_electricity_price(gui_params=gui_params)

    # Define time steps and index sets
    model.T = RangeSet(0, time_steps - 1) # time step t=0 to t=23
    model.T_plus1 = RangeSet(0, time_steps) # used for SOC update, t=0 to t=24
    model.Buses = Set(initialize=[bus.ID for bus in buses])
    model.Lines = Set(initialize=[line.ID for line in lines])
    model.Gens = Set(initialize=[gen.ID for gen in gens])
    model.PVWs = Set(initialize=[pvw.ID for pvw in pvws])
    model.ESSs = Set(initialize=[ess.ID for ess in esss])
    model.SOPs = Set(initialize=[sop.ID for sop in sops])
    model.NOPs = Set(initialize=[nop.ID for nop in nops])
    # A collection based on ev_info, representing all independent charging events
    model.EVs = Set(initialize=ev_info.keys())
    model.Spots = RangeSet(0, total_spots_count - 1)# Charging pile number

    # Variable definition
    # # Add decision variable: the power flowing into the slack bus from the upper-level power grid (pu)
    # def grid_inflow_p_bounds(model, t):
    # return (-float('inf'), float('inf')) # Assume that electricity purchase is not subject to the upper limit and is allowed to flow in and out.
    # Define the grid interactive power as a non-negative real number, forcing it to only purchase electricity (>=0) and not sell electricity (<0)
    model.grid_inflow_p = Var(model.T, domain=NonNegativeReals, name="grid_inflow_p") # Bus voltage (V)
    model.v = Var(model.Buses, model.T,
                  bounds=lambda model, bus_id, t: (grid.Bus(bus_id).MinV, grid.Bus(bus_id).MaxV))

    # Line active power (P)
    model.P = Var(model.Lines, model.T)

    # Line reactive power (Q)
    model.Q = Var(model.Lines, model.T)

    # Generator active power (pg)
    def pg_bounds(model, gen_id, t):
        gen = next(g for g in gens if g.ID == gen_id)
        pmin = gen.Pmin(t) if callable(gen.Pmin) else gen.Pmin
        pmax = gen.Pmax(t) if callable(gen.Pmax) else gen.Pmax

        # Read the real upper limit we set directly from the generator object
        final_pmax = min(pmax, gen.RealisticPmax) if pmax is not None else gen.RealisticPmax

        return (pmin, final_pmax)

    model.pg = Var(model.Gens, model.T, bounds=pg_bounds)

    # Generator reactive power (qg)
    def qg_bounds(model, gen_id, t):
        gen = next(g for g in gens if g.ID == gen_id)
        qmin = gen.Qmin(t) if callable(gen.Qmin) else (gen.Qmin if gen.Qmin is not None else -float('inf'))
        qmax = gen.Qmax(t) if callable(gen.Qmax) else (gen.Qmax if gen.Qmax is not None else float('inf'))
        return (qmin, qmax)

    model.qg = Var(model.Gens, model.T, bounds=qg_bounds)

    # Charging pile charging/discharging power (pspot), positive value means charging, negative value means discharging (V2G)
    def pspot_bounds(model, spot, t):
        if present_cars[spot, t] == 1:
            return (-original_env.EV_Param['discharging_rate'] / (sb_mva * 1000),
                    original_env.EV_Param['charging_rate'] / (sb_mva * 1000))
        return (0, 0)
    # model.pspot = Var(model.Spots, model.T, bounds=pspot_bounds)
    # model.boc_spot = Var(model.Spots, model.T_plus1, bounds=(0, 1))
    def pev_charge_bounds(model, ev_id, t):
        info = ev_info[ev_id]
        # Charging power is only allowed when the vehicle is present
        if present_cars[info['spot'], t] == 1:
            # Maximum charging power, converted to pu unit
            max_charge_pu = original_env.EV_Param['charging_rate'] / (sb_mva * 1000)
            return (0, max_charge_pu)
        return (0, 0) # When not present, the charging power is 0

    def pev_discharge_bounds(model, ev_id, t):
        info = ev_info[ev_id]
        # Discharge power is only allowed when the vehicle is present
        if present_cars[info['spot'], t] == 1:
            # Maximum discharge power, converted to pu unit
            max_discharge_pu = original_env.EV_Param['discharging_rate'] / (sb_mva * 1000)
            return (0, max_discharge_pu)
        return (0, 0) # When not present, the discharge power is 0

    model.pev_charge = Var(model.EVs, model.T, bounds=pev_charge_bounds)
    model.pev_discharge = Var(model.EVs, model.T, bounds=pev_discharge_bounds)
    model.ev_charge_or_discharge = Var(model.EVs, model.T, domain=Binary)
    model.pev = Var(model.EVs, model.T, bounds=lambda m, ev_id, t: pspot_bounds(m, ev_info[ev_id]['spot'], t))
    model.boc_ev = Var(model.EVs, model.T_plus1, bounds=(0, 1))

    step_seconds = gui_params['step_minutes'] * 60

    def pvw_available_power(pvw, t):
        return pvw.P(t * step_seconds) if callable(pvw.P) else pvw.P

    # Photovoltaic and wind power output (pvw_p), taking into account the maximum output limit
    def pvw_p_bounds(model, pvw_id, t):
        pvw = next(p for p in pvws if p.ID == pvw_id)
        p_max = pvw_available_power(pvw, t)
        return (0, p_max)

    model.pvw_p = Var(model.PVWs, model.T, bounds=pvw_p_bounds)

    def pvw_q_bounds(model, pvw_id, t):
        pvw = next(p for p in pvws if p.ID == pvw_id)
        p_max = pvw_available_power(pvw, t)
        return (-p_max * np.sqrt(1 - pvw.PF ** 2) / pvw.PF if pvw.PF != 0 else -p_max,
                p_max * np.sqrt(1 - pvw.PF ** 2) / pvw.PF if pvw.PF != 0 else p_max)

    model.pvw_q = Var(model.PVWs, model.T, bounds=pvw_q_bounds)

    # Energy storage system charging/discharging power (ess_p), positive value means charging, negative value means discharging
    def ess_p_bounds(model, ess_id, t):
        ess = next(e for e in esss if e.ID == ess_id)
        # Set the boundary only based on the physical maximum charge and discharge power of the energy storage unit itself
        return (-ess.MaxPd, ess.MaxPc)


    model.ess_p = Var(model.ESSs, model.T, bounds=ess_p_bounds)

    def ess_charge_bounds(model, ess_id, t):
        ess = next(e for e in esss if e.ID == ess_id)
        return (0, ess.MaxPc)

    model.ess_charge = Var(model.ESSs, model.T, bounds=ess_charge_bounds)

    def ess_discharge_bounds(model, ess_id, t):
        ess = next(e for e in esss if e.ID == ess_id)
        return (0, ess.MaxPd)

    model.ess_discharge = Var(model.ESSs, model.T, bounds=ess_discharge_bounds)


    def ess_soc_bounds(model, ess_id, t):
        return (0, 1)

    model.ess_soc = Var(model.ESSs, model.T_plus1, bounds=ess_soc_bounds)

    # SOP power flow variable
    def sop_p1_bounds(model, sop_id, t):
        sop = next(s for s in sops if s.ID == sop_id)
        return (-sop.PMax, sop.PMax) if sop.active else (0, 0)

    model.sop_p1 = Var(model.SOPs, model.T, bounds=sop_p1_bounds)

    def sop_q_bounds(model, sop_id, t):
        sop = next(s for s in sops if s.ID == sop_id)
        return (-sop.QMax, sop.QMax) if sop.active else (0, 0)

    model.sop_q1 = Var(model.SOPs, model.T, bounds=sop_q_bounds)
    model.sop_q2 = Var(model.SOPs, model.T, bounds=sop_q_bounds)

    def sop_loss_bounds(model, sop_id, t):
        return (0, float('inf')) if next(s for s in sops if s.ID == sop_id).active else (0, 0)

    model.sop_loss = Var(model.SOPs, model.T, bounds=sop_loss_bounds)

    # NOP state variable (0 means open, 1 means closed) and power flow
    model.nop_status = Var(model.NOPs, model.T, domain=Binary)
    model.nop_p = Var(model.NOPs, model.T)
    model.nop_q = Var(model.NOPs, model.T)

    # Slack variable, used for power flow balancing
    model.SlackP = Var(model.Buses, model.T)
    model.SlackQ = Var(model.Buses, model.T)

    model.unfull_energy_kwh = Var(ev_info.keys(), domain=NonNegativeReals)

    # Add slack variables to SOP capacity constraints
    model.sop_capacity_slack = Var(model.SOPs, model.T, domain=NonNegativeReals)
    # Add slack variables for NOP voltage constraints
    model.nop_v_slack_pos = Var(model.NOPs, model.T, domain=NonNegativeReals)
    model.nop_v_slack_neg = Var(model.NOPs, model.T, domain=NonNegativeReals)
    return (model, ev_info, price, present_cars, boc_initial, original_env,
            ev_capacity, buses, lines, gens, pvws, esss, sops, nops,
            bus_dict, line_dict, gen_dict, pvw_dict, ess_dict, sop_dict,
            nop_dict, grid, spot_to_bus_map)

def add_constraints(model, ev_info, present_cars, boc_initial, original_env, ev_capacity, buses, lines, gens, pvws,
                    esss, sops, nops, bus_dict, line_dict, gen_dict, pvw_dict, ess_dict, sop_dict, nop_dict, grid,
                    time_steps, spot_to_bus_map, gui_params): # <--- add gui_params at the end
    """
    (Multiple charging station version)
    Constraints to the Pyomo model were added, in particular the power flow balance was restructured to support multiple charging stations.
    """
    sb_mva = grid.SB
    # pev power separation constraints
    def pev_split_rule(model, ev_id, t):
        info = ev_info[ev_id]
        if info['arrival'] <= t < info['departure']:
            return model.pev[ev_id, t] == model.pev_charge[ev_id, t] - model.pev_discharge[ev_id, t]
        else:
            # If the vehicle is not present, its individual power is 0
            model.pev[ev_id, t].fix(0)
            return Constraint.Skip

    model.pev_split_constr = Constraint(model.EVs, model.T, rule=pev_split_rule)

    # BOC update constraints (based on EV)
    def boc_ev_update_rule(model, ev_id, t):
        """
        Update EV BOC using correct physics formula, including time step.
        """
        info = ev_info[ev_id]
        step_duration_h = gui_params['step_minutes'] / 60.0 # Get the time step (hours)

        # This rule is only valid during the vehicle’s stay time
        if info['arrival'] <= t < info['departure']:
            # Calculate the SOC change caused by charging and discharging. Be careful to multiply by time.
            charge_change = (model.pev_charge[ev_id, t] * sb_mva * 1000 * original_env.EV_Param[
                'charging_effic'] * step_duration_h) / ev_capacity
            discharge_change = (model.pev_discharge[ev_id, t] * sb_mva * 1000 * step_duration_h) / (
                        ev_capacity * original_env.EV_Param['discharging_effic'])

            # Determine whether it is the first time step when the vehicle arrives
            if t == info['arrival']:
                # At the arrival time, BOC is updated from the initial value
                return model.boc_ev[ev_id, t + 1] == info['initial_boc'] + charge_change - discharge_change
            else:
                # Non-arrival time, BOC inherits and updates from the previous time
                return model.boc_ev[ev_id, t + 1] == model.boc_ev[ev_id, t] + charge_change - discharge_change

        # If the vehicle just leaves at the current time step, we define its current BOC to be equal to the final value of the previous step for easy reading
        # Note: Pyomo's constraints are for t in model.T (0 to 22/23), while boc_ev has T+1
        if t > 0 and t == info['departure']:
            return model.boc_ev[ev_id, t] == model.boc_ev[ev_id, t]

        return Constraint.Skip

    model.boc_ev_update_constr = Constraint(model.EVs, model.T, rule=boc_ev_update_rule)

    # Constraint 1: Only when the switch ev_charge_or_discharge is 1, the charging power pev_charge can be greater than 0
    def ev_charge_limit_rule(model, ev_id, t):
        max_charge_pu = original_env.EV_Param['charging_rate'] / (sb_mva * 1000)
        return model.pev_charge[ev_id, t] <= model.ev_charge_or_discharge[ev_id, t] * max_charge_pu

    model.ev_charge_limit_constr = Constraint(model.EVs, model.T, rule=ev_charge_limit_rule)

    # Constraint 2: Only when the switch ev_charge_or_discharge is 0, the discharge power pev_discharge can be greater than 0
    def ev_discharge_limit_rule(model, ev_id, t):
        max_discharge_pu = original_env.EV_Param['discharging_rate'] / (sb_mva * 1000)
        return model.pev_discharge[ev_id, t] <= (1 - model.ev_charge_or_discharge[ev_id, t]) * max_discharge_pu

    model.ev_discharge_limit_constr = Constraint(model.EVs, model.T, rule=ev_discharge_limit_rule)

    # 3. Energy storage system power update constraints
    def initial_ess_soc_rule(model, ess_id):
        ess = next(e for e in esss if e.ID == ess_id)
        return model.ess_soc[ess_id, 0] == ess.SOC

    model.initial_ess_soc_constr = Constraint(model.ESSs, rule=initial_ess_soc_rule)

    def ess_soc_update_rule(model, ess_id, t):
        """
        Update ESS SOC with correct physical formulas.
        """
        ess = next(e for e in esss if e.ID == ess_id)
        step_duration_h = gui_params['step_minutes'] / 60.0

        charge_change = (model.ess_charge[ess_id, t] * ess.EC * step_duration_h) / ess.Cap
        discharge_change = (model.ess_discharge[ess_id, t] / ess.ED * step_duration_h) / ess.Cap

        return model.ess_soc[ess_id, t + 1] == model.ess_soc[ess_id, t] + charge_change - discharge_change

    model.ess_soc_update_constr = Constraint(model.ESSs, model.T, rule=ess_soc_update_rule)

    def ess_p_split_rule(model, ess_id, t):
        return model.ess_p[ess_id, t] == model.ess_charge[ess_id, t] - model.ess_discharge[ess_id, t]

    model.ess_p_split_constr = Constraint(model.ESSs, model.T, rule=ess_p_split_rule)


    # Prevent ESS from charging and discharging simultaneously
    model.ess_charge_or_discharge = Var(model.ESSs, model.T, domain=Binary)

    def ess_charge_limit_rule(model, ess_id, t):
        ess = next(e for e in esss if e.ID == ess_id)
        return model.ess_charge[ess_id, t] <= model.ess_charge_or_discharge[ess_id, t] * ess.MaxPc

    model.ess_charge_limit_constr = Constraint(model.ESSs, model.T, rule=ess_charge_limit_rule)

    def ess_discharge_limit_rule(model, ess_id, t):
        ess = next(e for e in esss if e.ID == ess_id)
        return model.ess_discharge[ess_id, t] <= (1 - model.ess_charge_or_discharge[ess_id, t]) * ess.MaxPd

    model.ess_discharge_limit_constr = Constraint(model.ESSs, model.T, rule=ess_discharge_limit_rule)

    # 4. Photovoltaic and wind power reactive power output constraints
    def pvw_q_rel_rule(model, pvw_id, t):
        pvw = next(p for p in pvws if p.ID == pvw_id)
        tan_phi = np.sqrt(1 - pvw.PF ** 2) / pvw.PF if pvw.PF != 0 else 0
        return model.pvw_q[pvw_id, t] == model.pvw_p[pvw_id, t] * tan_phi

    model.pvw_q_rel_constr = Constraint(model.PVWs, model.T, rule=pvw_q_rel_rule)

    def ess_q_coeff(ess):
        return float(np.sqrt(max(0.0, 1.0 - ess.PF ** 2)))

    # 5. Power flow balance constraints (active and reactive power)
    @model.Constraint(model.Buses, model.T, doc="Active power balance constraint")
    def p_balance_rule(model, bus_id, t):
        # Calculate the correct number of seconds based on time step index t
        time_in_seconds = t * gui_params['step_minutes'] * 60

        # --- Injections (Sources) ---
        power_injections = (
                sum(model.P[l.ID, t] for l in grid.LinesOfTBus(bus_id, only_active=True)) +
                sum(model.pg[g.ID, t] for g in grid.GensAtBus(bus_id) if g.ID in model.Gens) +
                sum(model.pvw_p[pvw.ID, t] for pvw in pvws if pvw.BusID == bus_id) +
                sum(model.nop_p[nop.ID, t] for nop in nops if nop.Bus2 == bus_id)
        )
        power_injections += sum(model.sop_p1[sop.ID, t] - model.sop_loss[sop.ID, t]
                                for sop in sops if sop.Bus2 == bus_id)

        # --- Ejections ---
        power_ejections = (
                sum(model.P[l.ID, t] for l in grid.LinesOfFBus(bus_id, only_active=True)) +
                grid.Bus(bus_id).Pd(time_in_seconds) + # Use the correct number of seconds to get the load
                sum(model.nop_p[nop.ID, t] for nop in nops if nop.Bus1 == bus_id) +
                sum(model.pev[ev_id, t] for ev_id, info in ev_info.items() if
                    info['spot'] in spot_to_bus_map and spot_to_bus_map[info['spot']] == bus_id) +
                sum(model.ess_p[ess.ID, t] for ess in esss if ess.BusID == bus_id)
        )
        power_ejections += sum(model.sop_p1[sop.ID, t] for sop in sops if sop.Bus1 == bus_id)

        # --- Balanced equation ---
        if bus_id == CORE_PARAMS["slack_bus"]:
            return power_injections + model.grid_inflow_p[t] == power_ejections
        else:
            return power_injections + model.SlackP[bus_id, t] == power_ejections

    @model.Constraint(model.Buses, model.T, doc="Reactive power balance constraint")
    def q_balance_rule(model, bus_id, t):
        # Calculate the correct number of seconds based on time step index t
        time_in_seconds = t * gui_params['step_minutes'] * 60

        q_injections = (
                sum(model.Q[l.ID, t] for l in grid.LinesOfTBus(bus_id, only_active=True)) +
                sum(model.qg[g.ID, t] for g in grid.GensAtBus(bus_id) if g.ID in model.Gens) +
                sum(model.pvw_q[pvw.ID, t] for pvw in pvws if pvw.BusID == bus_id) +
                sum(model.ess_discharge[ess.ID, t] * ess_q_coeff(ess) for ess in esss if ess.BusID == bus_id) +
                sum(model.sop_q1[sop.ID, t] for sop in sops if sop.Bus1 == bus_id) +
                sum(model.sop_q2[sop.ID, t] for sop in sops if sop.Bus2 == bus_id) +
                sum(model.nop_q[nop.ID, t] for nop in nops if nop.Bus2 == bus_id)
        )
        q_ejections = (
                sum(model.Q[l.ID, t] for l in grid.LinesOfFBus(bus_id, only_active=True)) +
                grid.Bus(bus_id).Qd(time_in_seconds) + # Use the correct number of seconds to get the load
                sum(model.ess_charge[ess.ID, t] * ess_q_coeff(ess) for ess in esss if ess.BusID == bus_id) +
                sum(model.nop_q[nop.ID, t] for nop in nops if nop.Bus1 == bus_id)
        )

        if bus_id == CORE_PARAMS["slack_bus"]:
            return q_injections == q_ejections
        else:
            return q_injections + model.SlackQ[bus_id, t] == q_ejections
    # 6. Voltage update constraints (linearized DistFlow model)
    def v_update_rule(model, line_id, t):
        line = next(l for l in lines if l.ID == line_id)
        f_bus = line.fBus
        t_bus = line.tBus
        r_pu = line.R
        x_pu = line.X
        return model.v[t_bus, t] == model.v[f_bus, t] - (r_pu * model.P[line_id, t] + x_pu * model.Q[line_id, t])

    model.v_update_constr = Constraint(model.Lines, model.T, rule=v_update_rule)

    # 7. SOP capacity and loss constraints
    def sop_capacity_bus1_rule(model, sop_id, t):
        sop = next(s for s in sops if s.ID == sop_id)
        if sop.active:
            terms = []
            if sop.PMax > 0:
                terms.append((model.sop_p1[sop_id, t] / sop.PMax) ** 2)
            if sop.QMax > 0:
                terms.append((model.sop_q1[sop_id, t] / sop.QMax) ** 2)
            if terms:
                return sum(terms) <= 1 + model.sop_capacity_slack[sop_id, t]
        return Constraint.Skip

    model.sop_capacity_bus1_constr = Constraint(model.SOPs, model.T, rule=sop_capacity_bus1_rule)

    def sop_capacity_bus2_rule(model, sop_id, t):
        sop = next(s for s in sops if s.ID == sop_id)
        if sop.active:
            terms = []
            if sop.PMax > 0:
                terms.append(((model.sop_p1[sop_id, t] - model.sop_loss[sop_id, t]) / sop.PMax) ** 2)
            if sop.QMax > 0:
                terms.append((model.sop_q2[sop_id, t] / sop.QMax) ** 2)
            if terms:
                return sum(terms) <= 1 + model.sop_capacity_slack[sop_id, t]
        return Constraint.Skip

    model.sop_capacity_bus2_constr = Constraint(model.SOPs, model.T, rule=sop_capacity_bus2_rule)

    def sop_loss_rule(model, sop_id, t):
        sop = next(s for s in sops if s.ID == sop_id)
        if sop.active and sop.PMax > 0:
            # Let the loss only be related to the active power P and remove the influence of Q
            return model.sop_loss[sop_id, t] >= sop.LossCoeff * (
                        model.sop_p1[sop_id, t] ** 2) / sop.PMax ** 2
        return Constraint.Skip

    model.sop_loss_constr = Constraint(model.SOPs, model.T, rule=sop_loss_rule)

    # 8. NOP power flow and voltage update constraints
    M = 1000 # Large M value, used for logical constraints

    def nop_p_limit1_rule(model, nop_id, t):
        return model.nop_p[nop_id, t] <= M * model.nop_status[nop_id, t]

    model.nop_p_limit1_constr = Constraint(model.NOPs, model.T, rule=nop_p_limit1_rule)

    def nop_p_limit2_rule(model, nop_id, t):
        return model.nop_p[nop_id, t] >= -M * model.nop_status[nop_id, t]

    model.nop_p_limit2_constr = Constraint(model.NOPs, model.T, rule=nop_p_limit2_rule)

    def nop_q_limit1_rule(model, nop_id, t):
        return model.nop_q[nop_id, t] <= M * model.nop_status[nop_id, t]

    model.nop_q_limit1_constr = Constraint(model.NOPs, model.T, rule=nop_q_limit1_rule)

    def nop_q_limit2_rule(model, nop_id, t):
        return model.nop_q[nop_id, t] >= -M * model.nop_status[nop_id, t]

    model.nop_q_limit2_constr = Constraint(model.NOPs, model.T, rule=nop_q_limit2_rule)

    def nop_v_residual(model, nop_id, t):
        nop = next(n for n in nops if n.ID == nop_id)
        return model.v[nop.Bus2, t] - model.v[nop.Bus1, t] + (
            nop.R * model.nop_p[nop_id, t] + nop.X * model.nop_q[nop_id, t]
        )

    def nop_v_upper_rule(model, nop_id, t):
        return nop_v_residual(model, nop_id, t) <= (
            model.nop_v_slack_pos[nop_id, t] + M * (1 - model.nop_status[nop_id, t])
        )

    model.nop_v_upper_constr = Constraint(model.NOPs, model.T, rule=nop_v_upper_rule)

    def nop_v_lower_rule(model, nop_id, t):
        return nop_v_residual(model, nop_id, t) >= (
            -model.nop_v_slack_neg[nop_id, t] - M * (1 - model.nop_status[nop_id, t])
        )

    model.nop_v_lower_constr = Constraint(model.NOPs, model.T, rule=nop_v_lower_rule)

    def ev_unfull_energy_rule(model, ev_id):
        info = ev_info[ev_id]
        departure = info["departure"]
        final_soc = model.boc_ev[ev_id, departure]
        return model.unfull_energy_kwh[ev_id] >= (1.00 - final_soc) * ev_capacity

    model.ev_unfull_energy_constr = Constraint(model.EVs, rule=ev_unfull_energy_rule)

    # 10. Line active power upper limit constraint
    def line_p_limit_rule(model, line_id, t):
        max_p = 5.0 # Assume that the maximum active power of each line is 3.0 pu, which can be adjusted according to the actual situation
        return model.P[line_id, t] <= max_p

    model.line_p_limit_constr = Constraint(model.Lines, model.T, rule=line_p_limit_rule)

    def line_p_lower_limit_rule(model, line_id, t):
        min_p = -5.0 # Assume that the minimum active power of each line is -3.0 pu, which can be adjusted according to the actual situation
        return model.P[line_id, t] >= min_p

    model.line_p_lower_limit_constr = Constraint(model.Lines, model.T, rule=line_p_lower_limit_rule)

    def fix_slack_bus_voltage_rule(model, t):
        """This rule fixes the voltage of the slack bus to 1.0 pu at all time steps."""
        return model.v[CORE_PARAMS["slack_bus"], t] == 1.0

    model.slack_bus_voltage_constraint = Constraint(
        model.T,
        rule=fix_slack_bus_voltage_rule,
        doc="Fix slack bus voltage to 1.0 pu for all timesteps"
    )
    return model

def define_objective_and_solve(model, ev_info, price, buses, gens, pvws, esss, sops, nops, grid,
                               time_steps, solver_name, gui_params, spot_to_bus_map=None):
    """
    Define the objective function and solve the model
    """

    # Unified the generator cost model and corrected the units of all cost items.
    def objective_rule(model):
        step_duration_h = gui_params['step_minutes'] / 60.0
        sb_mva = grid.SB
        # Read parameters from config
        penalties = BASELINE_PARAMS["penalty_factors"]
        ess_cost_factor = BASELINE_PARAMS["ess_degradation_cost"]
        ev_penalty_factor = penalties["ev_not_full_penalty"]
        # Grid Purchase Cost (Unit: Yuan)
        grid_purchase_cost = sum(price[t] * model.grid_inflow_p[t] * sb_mva * 1000 * step_duration_h for t in model.T)

        # Generation Cost (Unit: Yuan) - Use a quadratic cost model consistent with the RL environment
        generation_cost = sum(
            (grid.Gen(g_id).CostA(t) * (model.pg[g_id, t] * sb_mva) ** 2 +
             grid.Gen(g_id).CostB(t) * (model.pg[g_id, t] * sb_mva) +
             grid.Gen(g_id).CostC(t)) * step_duration_h
            for g_id in model.Gens for t in model.T
        )

        # SOP Loss Cost (Unit: Yuan)
        sop_loss_cost = sum(price[t] * model.sop_loss[sop_id, t] * sb_mva * 1000 * step_duration_h
                            for sop_id in model.SOPs for t in model.T)

        # ESS Discharge Cost (Unit: Yuan)
        ess_discharge_cost = sum(ess_cost_factor * model.ess_discharge[ess_id, t] * sb_mva * 1000 * step_duration_h
                                 for ess_id in model.ESSs for t in model.T)

        # Various punishment items
        slack_penalty = penalties["slack_power_penalty"] * sum(model.SlackP[b, t] ** 2 + model.SlackQ[b, t] ** 2 for b in model.Buses for t in model.T)
        ev_not_full_penalty = ev_penalty_factor * sum(model.unfull_energy_kwh.values())
        sop_slack_penalty = penalties["sop_capacity_penalty"] * sum(model.sop_capacity_slack.values())
        nop_slack_penalty = penalties["nop_voltage_penalty"] * sum(model.nop_v_slack_pos.values()) + \
                            penalties["nop_voltage_penalty"] * sum(model.nop_v_slack_neg.values())

        # Ultimate goal: Minimize the sum of all costs and penalties
        obj = (grid_purchase_cost + generation_cost + sop_loss_cost + ess_discharge_cost +
               slack_penalty + ev_not_full_penalty + sop_slack_penalty + nop_slack_penalty)
        return obj

    model.objective = Objective(rule=objective_rule, sense=minimize)

    # Solve the model
    solver = SolverFactory(solver_name)
    solver_time_limit = gui_params.get(
        "baseline_solver_time_limit",
        BASELINE_PARAMS.get("solver_time_limit", 300),
    )
    solver.options['TimeLimit'] = solver_time_limit
    solver.options['MIPGap'] = gui_params.get(
        "baseline_mip_gap",
        BASELINE_PARAMS.get("mip_gap", 1e-3),
    )
    if "gurobi" in str(solver_name).lower():
        solver.options['MIPFocus'] = gui_params.get(
            "baseline_mip_focus",
            BASELINE_PARAMS.get("mip_focus", 1),
        )
        solver.options['Heuristics'] = gui_params.get(
            "baseline_heuristics",
            BASELINE_PARAMS.get("heuristics", 0.5),
        )
        norel_time = gui_params.get(
            "baseline_norel_heur_time",
            BASELINE_PARAMS.get("norel_heur_time", 30),
        )
        solver.options['NoRelHeurTime'] = min(float(norel_time), max(0.0, float(solver_time_limit) * 0.5))
    solver.options['OutputFlag'] = 1
    results = solver.solve(model, tee=True, load_solutions=False)
    solver_note = ""

    def _solution_count(solver_results):
        try:
            return len(solver_results.solution)
        except Exception:
            return 0

    termination = results.solver.termination_condition
    status = results.solver.status
    accepted_termination = {
        TerminationCondition.optimal,
        TerminationCondition.globallyOptimal,
        TerminationCondition.feasible,
    }
    has_incumbent = _solution_count(results) > 0
    accepted_timeout = termination == TerminationCondition.maxTimeLimit and has_incumbent

    allow_relaxed_fallback = bool(gui_params.get(
        "baseline_allow_relaxed_fallback",
        BASELINE_PARAMS.get("allow_relaxed_fallback", False),
    ))
    if termination == TerminationCondition.maxTimeLimit and not has_incumbent and allow_relaxed_fallback:
        print("The solution has reached the time limit and there is no integer feasible solution; try relaxing binary variables for emergency solution.")
        TransformationFactory('core.relax_integer_vars').apply_to(model)
        relaxed_time_limit = min(float(solver_time_limit), 120.0)
        solver.options['TimeLimit'] = relaxed_time_limit
        results = solver.solve(model, tee=True, load_solutions=False)
        solver_note = "Binary variables were relaxed after the integer model hit the time limit without an incumbent."
        termination = results.solver.termination_condition
        status = results.solver.status
        has_incumbent = _solution_count(results) > 0
        accepted_timeout = termination == TerminationCondition.maxTimeLimit and has_incumbent
    elif termination == TerminationCondition.maxTimeLimit and not has_incumbent:
        print("The solution reached the time limit and there is no integer feasible solution; integer relaxation fallback is not enabled, Baseline Stage 1 will be treated as failed.")

    # Check the solution results
    if ((status in [SolverStatus.ok, SolverStatus.warning] and termination in accepted_termination) or
            accepted_timeout):
        if accepted_timeout:
            print("The solution has reached the time limit, but a feasible solution has been found; the current incumbent will be used to continue extracting results.")
            solver_note = solver_note or "Integer solve hit the time limit; using the best incumbent found by Gurobi."
        model.solutions.load_from(results)
        solve_result = GridSolveResult.OK
        print("Solving successfully! Extracting results...")

        # Recalculate directly from the model to ensure that it is completely consistent with the optimization goal
        step_duration_h = gui_params['step_minutes'] / 60.0
        sb_mva = grid.SB
        ess_cost_factor = 0.0

        grid_purchase_cost_val = sum(
            price[t] * value(model.grid_inflow_p[t]) * sb_mva * 1000 * step_duration_h for t in model.T)

        generation_cost_val = sum(
            (value(grid.Gen(g_id).CostA(t)) * (value(model.pg[g_id, t]) * sb_mva) ** 2 +
             value(grid.Gen(g_id).CostB(t)) * (value(model.pg[g_id, t]) * sb_mva) +
             value(grid.Gen(g_id).CostC(t))) * step_duration_h
            for g_id in model.Gens for t in model.T
        )

        sop_loss_cost_val = sum(price[t] * value(model.sop_loss[sop_id, t]) * sb_mva * 1000 * step_duration_h
                                for sop_id in model.SOPs for t in model.T)

        ess_discharge_cost_val = sum(
            ess_cost_factor * value(model.ess_discharge[ess_id, t]) * sb_mva * 1000 * step_duration_h
            for ess_id in model.ESSs for t in model.T)

        slack_penalty_val = 1e7 * sum(value(model.SlackP[b, t]) ** 2 + value(model.SlackQ[b, t]) ** 2
                                      for b in model.Buses for t in model.T)

        # Uniformly use the penalty factor in the configuration file for result verification
        penalties = BASELINE_PARAMS["penalty_factors"]
        ev_not_full_penalty_val = penalties["ev_not_full_penalty"] * sum(
            value(v) for v in model.unfull_energy_kwh.values())
        sop_slack_penalty_val = penalties["sop_capacity_penalty"] * sum(
            value(v) for v in model.sop_capacity_slack.values())
        nop_slack_penalty_val = penalties["nop_voltage_penalty"] * sum(
            value(v) for v in model.nop_v_slack_pos.values()) + \
                                penalties["nop_voltage_penalty"] * sum(value(v) for v in model.nop_v_slack_neg.values())

        baseline_data = {
            "objective_value": value(model.objective),
            "solver_note": solver_note,
            "grid_purchase_cost": grid_purchase_cost_val,
            "generation_cost": generation_cost_val,
            "sop_loss_cost": sop_loss_cost_val,
            "ess_discharge_cost": ess_discharge_cost_val,
            "slack_penalty": slack_penalty_val,
            "ev_not_full_penalty": ev_not_full_penalty_val,
            "sop_slack_penalty": sop_slack_penalty_val,
            "nop_slack_penalty": nop_slack_penalty_val,
            "sop_slacks": {sop.ID: [value(model.sop_capacity_slack[sop.ID, t]) for t in model.T] for sop in sops},
            "grid_inflow_p": [value(model.grid_inflow_p[t]) for t in model.T],
            "gen_powers": {gen_id: [value(model.pg[gen_id, t]) for t in model.T] for gen_id in model.Gens},
            "gen_q_powers": {gen_id: [value(model.qg[gen_id, t]) for t in model.T] for gen_id in model.Gens},
            "slack_powers": {b: [value(model.SlackP[b, t]) for t in model.T] for b in model.Buses},
            "bus_voltages": {},
            "ev_info": {},
            "total_ev_count": len(ev_info),
            "charged_ev_count": 0,
            "pvw_powers": {},
            "pvw_available_powers": {},
            "ess_powers": {},
            "ess_soc": {},
            "sop_flows": {},
            "nop_status": {},
            "nop_flows": {},
            "reconfiguration_plan": getattr(grid, "reconfiguration_plan", {}) or {},
            "available_reconfiguration_plans": getattr(grid, "available_reconfiguration_plans", []) or [],
            #Initialize spot_powers and ev_powers
            "spot_powers": {},
            "ev_powers": {},
            "spot_to_bus_map": dict(spot_to_bus_map or {})
        }

        # Extract physical quantities
        time_steps = len(list(model.T))
        baseline_data["nop_status"] = fixed_nop_status(grid, time_steps)

        for bus_id in model.Buses:
            baseline_data["bus_voltages"][bus_id] = [value(model.v[bus_id, t]) for t in model.T]

        baseline_data["line_powers"] = {}
        for line_id in model.Lines:
            baseline_data["line_powers"][line_id] = [value(model.P[line_id, t]) for t in model.T]

        # 1. Instead of reading pspot directly, first extract the new pev (vehicle power)
        for ev_id in model.EVs:
            baseline_data["ev_powers"][ev_id] = [value(model.pev[ev_id, t]) for t in model.T]

        # 2. Aggregate vehicle power (pev) into charging pile power (spot_powers) to be compatible with downstream analysis
        num_spots = len(list(model.Spots))
        baseline_data["spot_powers"] = {spot_idx: [0.0] * time_steps for spot_idx in range(num_spots)}
        for ev_id, info in ev_info.items():
            spot_id = info['spot']
            ev_powers_over_time = baseline_data["ev_powers"][ev_id]
            for t in range(time_steps):
                baseline_data["spot_powers"][spot_id][t] += ev_powers_over_time[t]

        # 3. Extract BOC from the new boc_ev variable instead of the old boc_spot
        charged_count = 0
        for ev_id, info in ev_info.items():
            departure = info["departure"]
            # Make sure the departure time is within the index range
            final_soc = value(model.boc_ev[ev_id, departure]) if departure <= time_steps else 0
            standards = EVALUATION_CONFIG["standards"] # Ensure accessibility within scope
            is_charged = final_soc >= standards["ev_charged_soc_threshold"]
            if is_charged:
                charged_count += 1
            baseline_data["ev_info"][ev_id] = {"initial_boc": info["initial_boc"], "final_boc": final_soc,
                                               "charged": is_charged}
        baseline_data["charged_ev_count"] = charged_count

        for pvw_id in model.PVWs:
            baseline_data["pvw_powers"][pvw_id] = [value(model.pvw_p[pvw_id, t]) for t in model.T]
            pvw = next(p for p in pvws if p.ID == pvw_id)
            step_seconds = gui_params['step_minutes'] * 60
            baseline_data["pvw_available_powers"][pvw_id] = [
                pvw.P(t * step_seconds) if callable(pvw.P) else pvw.P
                for t in model.T
            ]

        for ess_id in model.ESSs:
            baseline_data["ess_powers"][ess_id] = [value(model.ess_p[ess_id, t]) for t in model.T]
            baseline_data["ess_soc"][ess_id] = [value(model.ess_soc[ess_id, t]) for t in model.T_plus1]

        for sop_id in model.SOPs:
            if next(s for s in sops if s.ID == sop_id).active:
                baseline_data["sop_flows"][sop_id] = {"P1": [value(model.sop_p1[sop_id, t]) for t in model.T],
                                                      "Q1": [value(model.sop_q1[sop_id, t]) for t in model.T],
                                                      "Q2": [value(model.sop_q2[sop_id, t]) for t in model.T],
                                                      "Loss": [value(model.sop_loss[sop_id, t]) for t in model.T]}

        for nop_id in model.NOPs:
            baseline_data["nop_status"][nop_id] = [value(model.nop_status[nop_id, t]) for t in model.T]
            baseline_data["nop_flows"][nop_id] = {"P": [value(model.nop_p[nop_id, t]) for t in model.T],
                                                  "Q": [value(model.nop_q[nop_id, t]) for t in model.T]}

        print("Result extraction completed.")
        return solve_result, baseline_data, model

    else:
        solve_result = GridSolveResult.Failed
        print("!!!!!!!! Solution failed!!!!!!!!")
        print(f"Solver status: {results.solver.status}")
        print(f"Termination condition: {results.solver.termination_condition}")
        return solve_result, {}, None # Return None on failure


# file: baseline.py

def solve_baseline(grid: Grid, stations_list, gui_params: dict):
    """
    (Multiple charging station version)
    Integrate model creation, constraint addition, goal definition and solution process.
    """
    # Extract time configuration from parameter dictionary
    time_steps = int((gui_params['end_hour'] - gui_params['start_hour']) * (60 // gui_params['step_minutes']))


    # Create model and variables, now 23 values will be unpacked
    model_data = create_baseline_model(grid, stations_list, time_steps, gui_params)
    (model, ev_info, price, present_cars, boc_initial, original_env,
     ev_capacity, buses, lines, gens, pvws, esss, sops, nops,
     bus_dict, line_dict, gen_dict, pvw_dict, ess_dict, sop_dict,
     nop_dict, grid, spot_to_bus_map) = model_data

    # Add constraints
    model = add_constraints(model, ev_info, present_cars, boc_initial, original_env, ev_capacity, buses, lines, gens,
                            pvws, esss, sops, nops, bus_dict, line_dict, gen_dict, pvw_dict, ess_dict, sop_dict,
                            nop_dict, grid, time_steps, spot_to_bus_map, gui_params)


    # Define the objective function and solve it, receiving all return values
    result, baseline_data, solved_model = define_objective_and_solve(model, ev_info, price, buses, gens, pvws, esss,
                                                                     sops, nops, grid,
                                                                     time_steps, gui_params['solver'], gui_params,
                                                                     spot_to_bus_map)

    # Return all results (including model objects) to the caller
    return result, baseline_data, solved_model
