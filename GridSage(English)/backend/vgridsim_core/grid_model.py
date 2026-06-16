"""
This module is responsible for creating and configuring the power grid (Grid) objects required for simulation.

Core functions:
1. Provide a main function `create_grid` as a unified entrance to generate a complete power grid model including all components.
2. Load standard IEEE test cases (such as IEEE 33, IEEE 69) from the fpowerkit library as the base topology.
3. Read detailed, customized component parameters from the external data file `data/grid_parameters.xlsx`,
    Including load curve, electricity price, generator, distributed energy resources (photovoltaic, wind power, energy storage), and soft switching (SOP/NOP).
4. Load the parameters read from the data file into the basic power grid model, and overwrite or add the corresponding components.
5. Contains parameter standardization logic to ensure that the generated power grid objects behave consistently in different environments.
"""

import os
import pandas as pd
import sys
import numpy as np
from fpowerkit import Grid, Bus, Line, Generator, PVWind, ESS
from fpowerkit.cases import PDNCases
from sop_nop import SOP, NOP
from config import CORE_PARAMS, PATHS
from reconfiguration import apply_reconfiguration_plan
from feasytools.tfunc import SegFunc, ConstFunc

# Define the critical path within the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#GRID_PARAMS_FILE = os.path.join(BASE_DIR, 'data', 'grid_parameters.xlsx')


def _seg_func_from_series(time_points_sec, values):
    """Build SegFunc across feasytools versions with different constructor behavior."""
    times = [int(t) for t in time_points_sec]
    data = [float(v) for v in values]
    try:
        return SegFunc(times, data)
    except TypeError:
        return SegFunc(list(zip(times, data)), data)


def generate_stochastic_power_profile(predicted_profile, error_level=0.08):
    """
    Generate a random actual power curve based on the predicted power curve and error level.
    This is to model the forecast uncertainty of renewable energy generation.

    Parameters:
    - predicted_profile (list or np.array): List of predicted power values (from Excel after interpolation).
    - error_level (float): Uncertainty level, used as the ratio of the standard deviation to the mean.

    Return:
    - np.array: Actual power curve with random perturbations.
    """
    stochastic_profile = []
    for p_predicted in predicted_profile:
        # Apply perturbation only when the predicted value is greater than 0 to avoid abnormal power at night (when there is no light/wind)
        if p_predicted > 0:
            # The mean (mu) is the predicted value
            mu = p_predicted
            # Standard deviation (sigma) is a percentage of the predicted value
            sigma = p_predicted * error_level

            # Sampling a random value from the normal (Gaussian) distribution as the actual output
            p_actual = np.random.normal(loc=mu, scale=sigma)

            # Constrain the result to ensure it conforms to physical reality:
            # 1. The actual output cannot be negative.
            # 2. To simplify the model, it is assumed that the actual output will not exceed the current theoretical prediction value.
            p_actual = np.clip(p_actual, 0, p_predicted)

            stochastic_profile.append(p_actual)
        else:
            # If the predicted value is 0, the actual value is also 0
            stochastic_profile.append(0)

    return np.array(stochastic_profile)


def _profile_lookup(gui_params, key, hour, default):
    profile = (gui_params or {}).get("time_profiles") or {}
    values = profile.get(key) or {}
    if isinstance(values, list):
        return float(values[hour]) if hour < len(values) else default
    if isinstance(values, dict):
        for candidate in (hour, str(hour), f"{hour:02d}"):
            if candidate in values:
                return float(values[candidate])
        for range_key, value in values.items():
            text = str(range_key)
            if "-" not in text:
                continue
            try:
                start, end = text.split("-", 1)
                start_hour = int(float(start))
                end_hour = int(float(end))
            except (TypeError, ValueError):
                continue
            if start_hour <= int(hour) <= end_hour:
                return float(value)
    return default


def _multiplier_vector(gui_params, key, time_axis, default):
    return np.array([
        _profile_lookup(gui_params, key, int(float(hour)) % 24, float(default))
        for hour in time_axis
    ])


def _normalize_bus_id(value):
    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower().startswith("b"):
        try:
            return f"b{int(float(text[1:]))}"
        except (TypeError, ValueError):
            return text
    try:
        return f"b{int(float(text))}"
    except (TypeError, ValueError):
        return text


def _is_device_disabled(gui_params, bus_id, device_type, device_id):
    disabled = (gui_params or {}).get("disabled_devices") or CORE_PARAMS.get("disabled_devices", {}) or {}
    node_devices = disabled.get(_normalize_bus_id(bus_id), {}) or {}
    ids = node_devices.get(device_type, []) or []
    return "*" in ids or str(device_id) in ids


def _to_number(value, default=0.0):
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _node_override_items(gui_params):
    overrides = (gui_params or {}).get("node_overrides") or {}
    for node_name, attrs in overrides.items():
        attrs_dict = vars(attrs) if hasattr(attrs, "__dict__") else (attrs or {})
        yield _normalize_bus_id(node_name), attrs_dict


def _unique_id(existing_ids, preferred_id):
    if preferred_id not in existing_ids:
        return preferred_id
    suffix = 2
    while f"{preferred_id}_{suffix}" in existing_ids:
        suffix += 1
    return f"{preferred_id}_{suffix}"


def apply_node_overrides_to_grid(grid, gui_params):
    """Materialize node-level scenario additions as grid devices where possible."""
    if not gui_params:
        return grid

    sb_mva = getattr(grid, "SB", getattr(grid, "Sb", 1.0))
    for node_name, attrs_dict in _node_override_items(gui_params):
        if node_name not in grid.BusNames:
            print(f"[WARNING] Node {node_name} not found in grid, skipping node override.")
            continue

        bus = grid.Bus(node_name)

        add_load_kw = _to_number(attrs_dict.get("add_load_kw", 0), 0.0)
        if add_load_kw > 0:
            val_pu = add_load_kw / (sb_mva * 1000.0)
            original_pd = bus.Pd
            bus.Pd = lambda t, orig=original_pd, v=val_pu: orig(t) + v
            print(f"[Scenario] Node {node_name}: +{add_load_kw} kW load injected.")

        for device_type, key in (("pv", "add_pv_kw"), ("wind", "add_wind_kw")):
            add_kw = _to_number(attrs_dict.get(key, 0), 0.0)
            if add_kw <= 0:
                continue
            val_pu = add_kw / (sb_mva * 1000.0)
            pvw_id = _unique_id(
                set(grid.PVWindNames),
                f"scenario_{device_type}_{node_name}",
            )
            pvw = PVWind(
                pvw_id,
                node_name,
                p=ConstFunc(val_pu),
                pf=0.95,
                cc=1.5,
                tag=device_type,
            )
            grid.AddPVWind(pvw)
            print(f"[Scenario] Node {node_name}: +{add_kw} kW {device_type.upper()} added as PVWind {pvw_id}.")

        add_ess_kwh = _to_number(attrs_dict.get("add_ess_kwh", 0), 0.0)
        if add_ess_kwh > 0:
            cap_puh = add_ess_kwh / (sb_mva * 1000.0)
            ess_id = _unique_id(
                {ess.ID for ess in grid.ESSs},
                f"scenario_ess_{node_name}",
            )
            explicit_power_kw = _to_number(attrs_dict.get("add_ess_power_kw"), None)
            default_c_rate = _to_number(
                (gui_params or {}).get(
                    "default_ess_c_rate",
                    CORE_PARAMS.get("scenario_defaults", {}).get("ess_c_rate", 0.2),
                ),
                0.2,
            )
            c_rate = _to_number(attrs_dict.get("add_ess_c_rate"), default_c_rate)
            if explicit_power_kw is not None and explicit_power_kw > 0:
                max_power_pu = explicit_power_kw / (sb_mva * 1000.0)
                power_desc = f"{explicit_power_kw} kW"
            else:
                max_power_pu = cap_puh * max(c_rate, 0.0)
                power_desc = f"{max(c_rate, 0.0)}C"
            max_power_pu = max(max_power_pu, 1e-6)
            ess = ESS(
                ess_id,
                node_name,
                cap_puh=cap_puh,
                ec=0.9,
                ed=0.9,
                pc_max=max_power_pu,
                pd_max=max_power_pu,
                pf=0.95,
                policy=None,
                cprice=None,
                dprice=None,
                init_elec_puh=cap_puh * 0.5,
            )
            grid.AddESS(ess)
            print(f"[Scenario] Node {node_name}: +{add_ess_kwh} kWh ESS added as {ess_id} ({power_desc}).")

    return grid


def load_generators_from_excel(grid, gui_params=None):
    """
    Load the configuration of a general generator (Gen) from the 'Generators' worksheet of the Excel file.
    A key feature is that if the data is successfully read, it will first clear the default generator that comes with the case.
    Then add the generator defined in Excel to achieve a completely customized configuration.
    """
    try:
        df_gens = pd.read_excel(PATHS["grid_params_excel"], sheet_name="Generators", engine="openpyxl")

        # As long as the data can be successfully read, clear the default generator that comes with the case.
        if not df_gens.empty:
            #print("--- Diagnostic information: Custom generator data detected, clearing the default generator that comes with the case... ---")
            # Use list comprehensions to safely get and remove all existing generators
            for gen_id in [g.ID for g in grid.Gens]:
                grid.DelGen(gen_id)


        # Iterate through each row in Excel, create and add new generators
        for _, row in df_gens.iterrows():
            # Ensure all required columns are present for data loading
            required_cols = ["ID", "BusID", "Pmax_pu", "Pmin_pu", "Qmax_pu", "Qmin_pu", "CostA", "CostB", "CostC",
                             "RealisticPmax_pu"]
            if not all(col in row for col in required_cols):
                print(f"Warning: Required column missing from Generators worksheet, row skipped: {row.to_dict()}")
                continue


            gen_id = str(row["ID"])
            bus_id = _normalize_bus_id(row["BusID"])
            if _is_device_disabled(gui_params, bus_id, "generator", gen_id):
                print(f"[Scenario] Generator {gen_id} on {bus_id} disabled by current scenario.")
                continue

            gen = Generator(
                id=gen_id,
                busid=bus_id,
                pmax_pu=row["Pmax_pu"],
                pmin_pu=row["Pmin_pu"],
                qmax_pu=row["Qmax_pu"],
                qmin_pu=row["Qmin_pu"],
                costA=row["CostA"],
                costB=row["CostB"],
                costC=row["CostC"]
            )
            # Attach a custom "real physical upper limit" attribute for use in the optimization model
            gen.RealisticPmax = row["RealisticPmax_pu"]

            # Add the generator created from Excel to the grid
            grid.AddGen(gen)


    except Exception as e:
        # If the worksheet does not exist or the read fails, print a warning and continue to use the default generator that comes with the case
        print(f"Warning: Failed to load custom generator - {e}. The default generator that comes with the case will be used.")

    return grid


def load_electricity_price(gui_params):
    """
    Load 24-hour electricity price data from Excel, and based on the simulation parameters (start and end time, step size) passed in from the GUI
    Linearly interpolates the electricity price curve to match the simulation timeline.
    """
    if gui_params is None:
        return [0.05] * 24 # If no parameters are provided, return a default value

    # Get time settings from GUI parameter dictionary
    start_hour = gui_params['start_hour']
    end_hour = gui_params['end_hour']
    step_minutes = gui_params['step_minutes']

    try:
        df = pd.read_excel(PATHS["grid_params_excel"], sheet_name="ElectricityPrice", engine="openpyxl")
        # Read the electricity price at 24 hours from Excel
        hourly_prices = [
            df[f"Price_t{t}"].iloc[0] if f"Price_t{t}" in df.columns and pd.notna(df[f"Price_t{t}"].iloc[0]) else 0.0
            for t in range(24)]

        # Create original time points (0, 1, 2, ..., 23)
        original_time_points = np.arange(24)
        # Create new, higher resolution time points based on GUI settings
        step_duration_hours = step_minutes / 60.0
        new_time_points = np.arange(start_hour, end_hour, step_duration_hours)

        # Use NumPy’s interp function for linear interpolation
        interpolated_prices = np.interp(new_time_points, original_time_points, hourly_prices)

        # If there are invalid values such as -1 in Excel, fill it with the previous valid value
        last_valid_price = 0.05
        for i, price in enumerate(hourly_prices):
            if price != -1.0:
                last_valid_price = price
            else:
                hourly_prices[i] = last_valid_price

        # Use the filled data to interpolate again to ensure the validity of the result
        interpolated_prices = np.interp(new_time_points, original_time_points, hourly_prices, left=hourly_prices[0],
                                        right=hourly_prices[-1])

        print(f"Electricity price data has been resampled, total number of steps: {len(interpolated_prices)}")
        return list(interpolated_prices)
    except Exception as e:
        print(f"Error: Failed to load electricity price - {e}")
        # If loading fails, return a default electricity price list to ensure that the program can continue to run.
        time_steps = int((end_hour - start_hour) * (60 / step_minutes))
        return [0.05] * time_steps


def load_bus_loads(grid, gui_params):
    """
    Load bus loads from Excel. This function can be based on the grid model selected by the GUI (such as 'ieee33')
    Dynamically select the Sheet page to be read (such as 'BusLoads_ieee33') to achieve automatic matching of data and model.
    """
    #print("--- Diagnostic message: Trying to load load from Excel file... ---")

    grid_model_name = gui_params['grid_model']
    # Dynamically construct the name of the Sheet page based on the model name
    sheet_name_to_load = f"BusLoads_{grid_model_name}"

    start_hour = gui_params['start_hour']
    end_hour = gui_params['end_hour']
    step_minutes = gui_params['step_minutes']

    try:
        # Use the dynamically constructed Sheet page name to read data
        df = pd.read_excel(PATHS["grid_params_excel"], sheet_name=sheet_name_to_load, engine="openpyxl")
    except Exception as e:
        print(f"--- Fatal error: Failed to load bus load! Unable to read '{sheet_name_to_load}' Sheet page.---")
        print(f"--- the error message is: {e} ---")
        return grid

    # In order to avoid interpolation timeline misalignment, always generate a complete 24-hour high-resolution timeline first
    full_day_steps = int(24 * (60 / step_minutes))
    full_day_time_axis = np.linspace(0, 24, full_day_steps, endpoint=False)
    original_time_points = np.arange(24)

    # Calculate the start and end index of the simulation window on the complete timeline
    start_step_index = int((start_hour) * (60 / step_minutes))
    end_step_index = int((end_hour) * (60 / step_minutes))

    timestep_seconds = step_minutes * 60
    overwrite_success = False

    # Traverse each row in Excel (representing a bus)
    for _, row in df.iterrows():
        bus_id = str(row["BusID"])
        # If the bus ID in Excel does not exist in the current power grid model, skip
        if bus_id not in grid.BusNames:
            continue

        bus = grid.Bus(bus_id)
        if bus:
            # --- Handle active load Pd ---
            hourly_pd = [row[f"Pd_t{t}"] if f"Pd_t{t}" in row and pd.notna(row[f"Pd_t{t}"]) else 0.0 for t in range(24)]
            load_factor = _multiplier_vector(
                gui_params,
                "load_multiplier_by_hour",
                full_day_time_axis,
                gui_params.get("global_load_multiplier", 1.0),
            )
            full_day_interpolated_pd = np.interp(full_day_time_axis, original_time_points, hourly_pd) * load_factor
            full_day_interpolated_pd = generate_stochastic_power_profile(full_day_interpolated_pd,
                                                                         error_level=0.05) # Assume 5% error
            # Cut out the required part of the simulation window from the complete interpolation result
            final_pd_slice = full_day_interpolated_pd[start_step_index:end_step_index]
            # Use fpowerkit's SegFunc (segmented function) to represent this timing load
            time_points_sec = [int(i * timestep_seconds) for i in range(len(final_pd_slice))]
            bus.Pd = _seg_func_from_series(time_points_sec, final_pd_slice)

            # --- Handle reactive load Qd (logic same as above) ---
            hourly_qd = [row[f"Qd_t{t}"] if f"Qd_t{t}" in row and pd.notna(row[f"Qd_t{t}"]) else 0.0 for t in range(24)]
            full_day_interpolated_qd = np.interp(full_day_time_axis, original_time_points, hourly_qd) * load_factor
            full_day_interpolated_qd = generate_stochastic_power_profile(full_day_interpolated_qd,
                                                                         error_level=0.05) # Assume 5% error
            final_qd_slice = full_day_interpolated_qd[start_step_index:end_step_index]
            time_points_sec_q = [int(i * timestep_seconds) for i in range(len(final_qd_slice))]
            bus.Qd = _seg_func_from_series(time_points_sec_q, final_qd_slice)

            overwrite_success = True

    if overwrite_success:
        #print("--- Diagnostic information: The load data in Excel has been successfully overwritten into the power grid model.---")
        pass

    return grid


def load_distributed_energy(grid, gui_params):
    """
    Load all distributed energy resources (DER) from Excel, including photovoltaic (PV), wind power (Wind) and energy storage (ESS).
    The output curves of photovoltaic and wind power will be interpolated and randomized.
    """
    start_hour = gui_params['start_hour']
    end_hour = gui_params['end_hour']
    step_minutes = gui_params['step_minutes']

    # ------------------- 1. Load photovoltaic (PV) and wind power (Wind) -------------------
    if CORE_PARAMS["distributed_energy"]["pv"] or CORE_PARAMS["distributed_energy"]["wind"]:
        try:
            df_pvw = pd.read_excel(PATHS["grid_params_excel"], sheet_name="PVWind", engine="openpyxl")
            original_time_points = np.arange(24)
            step_duration_hours = step_minutes / 60.0
            new_time_points = np.arange(start_hour, end_hour, step_duration_hours)
            timestep_seconds = step_minutes * 60
            for _, row in df_pvw.iterrows():
                pvw_id = str(row["ID"])
                bus_id = _normalize_bus_id(row["BusID"])
                pvw_type = str(row["Type"]).lower()
                if _is_device_disabled(gui_params, bus_id, pvw_type, pvw_id):
                    print(f"[Scenario] {pvw_type.upper()} {pvw_id} on {bus_id} disabled by current scenario.")
                    continue

                if (pvw_type == "pv" and CORE_PARAMS["distributed_energy"]["pv"]) or \
                        (pvw_type == "wind" and CORE_PARAMS["distributed_energy"]["wind"]):
                    # First check if the bus is in the current grid
                    if hasattr(grid, "_bnames") and bus_id not in grid._bnames:
                        print(f"Warning: Bus {bus_id} of PV/Wind {pvw_id} is not found in the current distribution network model, skip this unit.")
                        continue
                    # Read 24-hour forecast output from Excel
                    p_values = [row[f"P_t{t}"] if f"P_t{t}" in row and pd.notna(row[f"P_t{t}"]) else 0.0 for t in
                                range(24)]

                    # First perform interpolation to obtain a deterministic prediction curve that matches the simulation step size.
                    predicted_p_profile = np.interp(new_time_points, original_time_points, p_values)

                    # Then, call the auxiliary function to add random perturbation to the prediction curve and simulate the prediction error
                    UNCERTAINTY_LEVEL = 0.08
                    pv_factor = _multiplier_vector(
                        gui_params,
                        "pv_multiplier_by_hour",
                        new_time_points,
                        gui_params.get("global_pv_multiplier", 1.0) if pvw_type == "pv" else 1.0,
                    )
                    actual_p_profile = generate_stochastic_power_profile(predicted_p_profile, UNCERTAINTY_LEVEL) * pv_factor

                    # Finally, use this actual output curve with randomness to create fpowerkit’s piecewise function
                    time_points_sec = [int(i * timestep_seconds) for i in range(len(actual_p_profile))]
                    p_func = _seg_func_from_series(time_points_sec, actual_p_profile)

                    pvw = PVWind(
                        pvw_id,
                        bus_id,
                        p=p_func, # Assign the power function with uncertainty to the object
                        pf=row["PF"] if pd.notna(row["PF"]) else 0.95,
                        cc=row["CC"] if pd.notna(row["CC"]) else 1.5,
                        tag=pvw_type
                    )
                    grid.AddPVWind(pvw)
        except ValueError as e:
            print(f"Warning: An error occurred while reading Excel file 'PVWind' worksheet - {e}, skipping photovoltaic/wind power data loading.")

    # ------------------- 2. Load Energy Storage System (ESS) -------------------
    # The parameters of the energy storage system are static, not time series, so they can be read directly.
    if CORE_PARAMS["distributed_energy"]["ess"]:
        try:
            df_ess = pd.read_excel(PATHS["grid_params_excel"], sheet_name="ESS", engine="openpyxl")
            for _, row in df_ess.iterrows():
                bus_id = _normalize_bus_id(row["BusID"])
                ess_id = str(row["ID"])
                if _is_device_disabled(gui_params, bus_id, "ess", ess_id):
                    print(f"[Scenario] ESS {ess_id} on {bus_id} disabled by current scenario.")
                    continue

                # Check if the bus exists
                if hasattr(grid, "_bnames") and bus_id not in grid._bnames:
                    print(f"Warning: Bus {bus_id} of ESS {row['ID']} is not found in the current distribution network model, skip this energy storage.")
                    continue
                # 1. Get the deterministic initial value and capacity
                init_soc_deterministic = row["Init_Elec_puh"] if pd.notna(row["Init_Elec_puh"]) else 0.25
                ess_cap = row["Cap_puh"] if pd.notna(row["Cap_puh"]) else 0.5

                # 2. Generate random disturbance (for example: mean is 0, standard deviation is 0.2, that is, 20% disturbance)
                soc_noise_factor = np.random.normal(loc=1.0, scale=0.2)

                # 3. Apply perturbation and ensure SOC is within [0, capacity] range
                init_soc_stochastic = init_soc_deterministic * soc_noise_factor
                init_soc_stochastic = np.clip(init_soc_stochastic, 0, ess_cap)
                ess = ESS(
                    ess_id,
                    bus_id,
                    cap_puh=row["Cap_puh"] if pd.notna(row["Cap_puh"]) else 0.5,
                    ec=row["EC"] if pd.notna(row["EC"]) else 0.9,
                    ed=row["ED"] if pd.notna(row["ED"]) else 0.9,
                    pc_max=row["Pc_max"] if pd.notna(row["Pc_max"]) else 0.1,
                    pd_max=row["Pd_max"] if pd.notna(row["Pd_max"]) else 0.1,
                    pf=row["PF"] if pd.notna(row["PF"]) else 0.95,
                    # Policy is set to None to indicate that its charging and discharging behavior is determined by an external optimizer (Baseline or RL)
                    policy=None,
                    cprice=None,
                    dprice=None,
                    init_elec_puh=init_soc_stochastic # <-- use randomized values
                )
                grid.AddESS(ess)
        except ValueError as e:
            print(f"Warning: An error occurred while reading Excel file 'ESS' worksheet - {e}, skipping energy storage data loading.")

    return grid


def load_sop_nop(grid, gui_params=None):
    """
    Load the soft switch (SOP) and normally open point (NOP) parameters from an Excel file.
    When loading, it will be verified whether the bus connected to the SOP/NOP exists in the current power grid model.
    """
    # Load SOP
    if CORE_PARAMS["sop_nodes_active"]:
        try:
            df_sop = pd.read_excel(PATHS["grid_params_excel"], sheet_name="SOP", engine="openpyxl")
            if not hasattr(grid, 'SOPs'):
                grid.SOPs = {}
            for _, row in df_sop.iterrows():
                sop_id = str(row["ID"])
                bus1 = _normalize_bus_id(row["Bus1"])
                bus2 = _normalize_bus_id(row["Bus2"])
                if _is_device_disabled(gui_params, bus1, "sop", sop_id) or _is_device_disabled(gui_params, bus2, "sop", sop_id):
                    print(f"[Scenario] SOP {sop_id} disabled by current scenario.")
                    continue
                # Verify whether the bus ID exists in the current distribution network model
                if grid.Bus(bus1) is None or grid.Bus(bus2) is None:
                    print(f"Warning: Bus {bus1} or {bus2} for SOP {sop_id} was not found in the distribution network model, skipping this device.")
                    continue
                sop = SOP(
                    sop_id,
                    bus1,
                    bus2,
                    p_max_pu=row["P_max_pu"] if pd.notna(row["P_max_pu"]) else 0.5,
                    q_max_pu=row["Q_max_pu"] if pd.notna(row["Q_max_pu"]) else 0.3,
                    loss_coeff=row["Loss_Coeff"] if pd.notna(row["Loss_Coeff"]) else 0.05,
                    active=True
                )
                grid.SOPs[sop.ID] = sop
        except ValueError as e:
            print(f"Warning: An error occurred while reading Excel file 'SOP' worksheet - {e}, skipping SOP data loading.")

    # Load NOP
    if CORE_PARAMS["nop_nodes_active"]:
        try:
            df_nop = pd.read_excel(PATHS["grid_params_excel"], sheet_name="NOP", engine="openpyxl")
            if not hasattr(grid, 'NOPs'):
                grid.NOPs = {}
            for _, row in df_nop.iterrows():
                nop_id = str(row["ID"])
                bus1 = _normalize_bus_id(row["Bus1"])
                bus2 = _normalize_bus_id(row["Bus2"])
                if _is_device_disabled(gui_params, bus1, "nop", nop_id) or _is_device_disabled(gui_params, bus2, "nop", nop_id):
                    print(f"[Scenario] NOP {nop_id} disabled by current scenario.")
                    continue
                # Also perform bus ID verification
                if grid.Bus(bus1) is None or grid.Bus(bus2) is None:
                    print(f"Warning: Bus {bus1} or {bus2} for NOP {nop_id} was not found in the distribution network model, skipping this device.")
                    continue
                nop = NOP(
                    nop_id,
                    bus1,
                    bus2,
                    r_pu=row["R_pu"] if pd.notna(row["R_pu"]) else 0.001,
                    x_pu=row["X_pu"] if pd.notna(row["X_pu"]) else 0.01,
                    max_I_kA=row["Max_I_kA"] if pd.notna(row["Max_I_kA"]) else 0.9,
                    active=False  # The status of NOP is determined by the optimization model, here is only for initialization
                )
                grid.NOPs[nop.ID] = nop
        except ValueError as e:
            print(f"Warning: An error occurred while reading Excel file 'NOP' worksheet - {e}, skipping NOP data loading.")
    return grid


def load_station_info(gui_params=None):
    """Load the information of all charging stations from the 'EVStation' worksheet of the Excel file."""
    try:
        df_ev = pd.read_excel(PATHS["grid_params_excel"], sheet_name="EVStation", engine="openpyxl")
        if gui_params:
            rows = []
            for item in df_ev.to_dict("records"):
                bus_id = _normalize_bus_id(item.get("Bus_ID", item.get("BusID", "")))
                station_id = str(item.get("Station_ID", item.get("StationID", item.get("ID", ""))))
                if _is_device_disabled(gui_params, bus_id, "ev_station", station_id):
                    print(f"[Scenario] EV station {station_id} on {bus_id} disabled by current scenario.")
                    continue
                item["Bus_ID"] = bus_id
                rows.append(item)
            print(f"Successfully loaded {len(rows)} charging station information.")
            return rows
        # Convert DataFrame to a list of dictionaries, one dictionary per row, easy to use
        stations_info = df_ev.to_dict('records')
        print(f"Successfully loaded {len(stations_info)} charging station information.")
        return stations_info
    except Exception as e:
        print(f"Warning: An error occurred while reading Excel file 'EVStation' worksheet - {e}, returning an empty list.")
        return []


def create_ieee_grid(model="ieee33"):
    """Create a basic IEEE distribution network case from fpowerkit based on the specified model name (such as 'ieee33')."""
    if model == "ieee33":
        return PDNCases.IEEE33()
    elif model == "ieee69":
        return PDNCases.IEEE69()
    elif model == "ieee123":
        return PDNCases.IEEE123()
    else:
        print(f"Error: Unsupported model {model}, IEEE 33 will be used by default.")
        return PDNCases.IEEE33()


def create_grid(model="ieee33", gui_params=None):
    """
    (Robustness enhanced version) The main function to create a complete distribution network model.

    This is a factory function that performs the following operations in sequence:
    1. Create a basic IEEE power grid topology.
    2. Manually set and correct key grid reference parameters (SB, UB) to ensure compatibility.
    3. Call all other `load_*` functions to load and apply custom data in Excel to the power grid object.
    4. Perform final parameter standardization and sanity check on all components.
    """
    # 1. Create a basic power grid case
    grid = create_ieee_grid(model)

    # In order to cope with the differences that may exist in different versions of the fpowerkit library, manually check and set the SB and UB attributes here.
    # This ensures that the grid object returned from this function must contain the base power and base voltage values ​​we need.
    if model == "ieee33":
        grid.SB = 1.0 # The baseline power of IEEE 33 nodes is 1.0 MVA
        grid.UB = 12.66 # The reference voltage of IEEE 33 node is 12.66 kV
    elif model == "ieee69":
        grid.SB = 10.0 # The baseline power of IEEE 69 nodes is 10.0 MVA
        grid.UB = 12.66 # The reference voltage of IEEE 69 node is 12.66 kV
    elif model == "ieee123":
        grid.SB = 5.0 # The baseline power of IEEE 69 nodes is 5.0 MVA
        grid.UB = 4.16 # The reference voltage of IEEE 69 node is 4.16 kV
    else:
        # Set a default value for other possible models, just in case
        if not hasattr(grid, 'SB'): grid.SB = 1.0
        if not hasattr(grid, 'UB'): grid.UB = 12.66
    print(f"The base power SB={grid.SB} MVA, base voltage UB={grid.UB} kV of grid model '{model}' has been ensured ---")

    # 2. Set a default voltage constraint range for all buses
    for bus in grid.Buses:
        bus.MinV = 0.95
        bus.MaxV = 1.05

    # 3. If GUI parameters are provided (in evaluate_agents.py, GUI parameters have been defined by config.py), call all data loading functions
    if gui_params:
        grid = load_bus_loads(grid, gui_params)
        grid = load_distributed_energy(grid, gui_params)
        grid = apply_node_overrides_to_grid(grid, gui_params)
        grid = load_sop_nop(grid, gui_params)
        grid = load_generators_from_excel(grid, gui_params)

    # 4. Final parameter standardization of all generators
    # Ensure that each generator has the cost function and true upper limit properties required to optimize the model
    REALISTIC_PMAX_PU = 5
    TARGET_COST_A = 0.1
    TARGET_COST_B = 600
    TARGET_COST_C = 10
    for gen in grid.Gens:
        if not hasattr(gen, 'RealisticPmax'):
            gen.RealisticPmax = REALISTIC_PMAX_PU
        if gen.CostA is None: gen.CostA = ConstFunc(TARGET_COST_A)
        if gen.CostB is None: gen.CostB = ConstFunc(TARGET_COST_B)
        if gen.CostC is None: gen.CostC = ConstFunc(TARGET_COST_C)
        print(
            f"Final generator configuration {gen.ID} (on bus {gen.BusID}): RealisticPmax={getattr(gen, 'RealisticPmax', 'N/A')}, CostB={gen.CostB(0)}")

    # 5. Apply the selected radial reconfiguration plan. NOP is a topology
    # action: closing it also opens one base line on the same loop.
    if gui_params:
        plan = apply_reconfiguration_plan(grid, gui_params)
        print(
            f"[Reconfiguration] Active plan {plan.get('plan_id')}: "
            f"close_nop={plan.get('close_nop_id')}, open_line={plan.get('open_line_id')}, "
            f"radial={plan.get('is_radial')}, connected={plan.get('is_connected')}"
        )

    return grid
