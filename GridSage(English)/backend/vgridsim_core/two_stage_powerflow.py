import os
import sys
import gymnasium as gym
import pandas as pd
from grid_model import create_grid
from baseline import solve_baseline, create_baseline_model, add_constraints, define_objective_and_solve
from config import CORE_PARAMS, TIMESTEPS_PER_EPISODE
from fpowerkit.solbase import GridSolveResult
from fpowerkit.soldss import OpenDSSSolver
from pyomo.environ import value
import pickle

# Make sure the output directory exists, set to results/distflow and results/opendss in the project root directory
root_dir = os.getcwd() # Get the current working directory (root directory)
distflow_dir = os.path.join(root_dir, "results", "distflow")
opendss_dir = os.path.join(root_dir, "results", "opendss")
if not os.path.exists(distflow_dir):
    os.makedirs(distflow_dir, exist_ok=True)
if not os.path.exists(opendss_dir):
    os.makedirs(opendss_dir, exist_ok=True)


def save_distflow_results(baseline_data, grid, timestep, result, objective_value, model=None):
    """
    Save the results of Linear DistFlow, using baseline_data or data in model.
    :param baseline_data: Dictionary of results returned from solve_baseline
    :param grid: Distribution grid object
    :param timestep: current time step
    :param result: solution result status (GridSolveResult)
    :param objective_value: target value
    :param model: Pyomo model object, used to directly extract data
    :return: whether saved successfully
    """
    try:
        directory = distflow_dir

        # Save bus voltage results
        bus_data = []
        if model and hasattr(model, 'Buses') and hasattr(model, 'v'):
            for bus_id in model.Buses:
                bus = grid.Bus(bus_id) if hasattr(grid, 'Bus') and callable(getattr(grid, 'Bus')) else None
                voltage = value(model.v[bus_id, timestep]) if timestep in model.T else 'N/A'
                pd_val = bus.Pd(timestep) if bus and hasattr(bus, 'Pd') and callable(getattr(bus, 'Pd', None)) else (getattr(bus, 'Pd', 'N/A') if bus and hasattr(bus, 'Pd') else 'N/A')
                qd_val = bus.Qd(timestep) if bus and hasattr(bus, 'Qd') and callable(getattr(bus, 'Qd', None)) else (getattr(bus, 'Qd', 'N/A') if bus and hasattr(bus, 'Qd') else 'N/A')
                bus_data.append({
                    'BusID': bus_id,
                    'Voltage_pu': voltage,
                    'Pd_pu': pd_val,
                    'Qd_pu': qd_val
                })
        else:
            for bus_id, voltages in baseline_data["bus_voltages"].items():
                bus = grid.Bus(bus_id) if hasattr(grid, 'Bus') and callable(getattr(grid, 'Bus')) else None
                voltage = voltages[timestep] if timestep < len(voltages) else 'N/A'
                pd_val = bus.Pd(timestep) if bus and hasattr(bus, 'Pd') and callable(getattr(bus, 'Pd', None)) else (getattr(bus, 'Pd', 'N/A') if bus and hasattr(bus, 'Pd') else 'N/A')
                qd_val = bus.Qd(timestep) if bus and hasattr(bus, 'Qd') and callable(getattr(bus, 'Qd', None)) else (getattr(bus, 'Qd', 'N/A') if bus and hasattr(bus, 'Qd') else 'N/A')
                bus_data.append({
                    'BusID': bus_id,
                    'Voltage_pu': voltage,
                    'Pd_pu': pd_val,
                    'Qd_pu': qd_val
                })
        bus_df = pd.DataFrame(bus_data)
        bus_file = os.path.join(directory, f"bus_results_t{timestep}.csv")
        bus_df.to_csv(bus_file, index=False)
        print(f"Save {bus_file}, number of data lines: {len(bus_data)}, absolute path: {os.path.abspath(bus_file)}")

        # Save charging pile power allocation results
        spot_data = []
        for spot, powers in baseline_data["spot_powers"].items():
            power = powers[timestep] if timestep < len(powers) else 0.0
            spot_data.append({
                'SpotID': spot,
                'Power_pu': power
            })
        spot_df = pd.DataFrame(spot_data)
        spot_file = os.path.join(directory, f"spot_results_t{timestep}.csv")
        spot_df.to_csv(spot_file, index=False)
        print(f"Save {spot_file}, number of data lines: {len(spot_data)}, absolute path: {os.path.abspath(spot_file)}")

        # Save photovoltaic/wind power output results
        pvw_data = []
        for pvw_id, powers in baseline_data["pvw_powers"].items():
            power = powers[timestep] if timestep < len(powers) else 0.0
            pvw_data.append({
                'PVWID': pvw_id,
                'Power_pu': power
            })
        pvw_df = pd.DataFrame(pvw_data)
        pvw_file = os.path.join(directory, f"pvw_results_t{timestep}.csv")
        pvw_df.to_csv(pvw_file, index=False)
        print(f"Save {pvw_file}, number of data lines: {len(pvw_data)}, absolute path: {os.path.abspath(pvw_file)}")

        # Save energy storage system power and charge results
        ess_data = []
        for ess_id, powers in baseline_data["ess_powers"].items():
            power = powers[timestep] if timestep < len(powers) else 0.0
            socs = baseline_data["ess_soc"].get(ess_id, [])
            soc = socs[timestep] if timestep < len(socs) else 0.0
            ess_data.append({
                'ESSID': ess_id,
                'Power_pu': power,
                'SOC_pu': soc
            })
        ess_df = pd.DataFrame(ess_data)
        ess_file = os.path.join(directory, f"ess_results_t{timestep}.csv")
        ess_df.to_csv(ess_file, index=False)
        print(f"Save {ess_file}, number of data lines: {len(ess_data)}, absolute path: {os.path.abspath(ess_file)}")

        # Save generator power results (if there is a model object)
        gen_data = []
        if model and hasattr(model, 'Gens') and hasattr(model, 'pg') and hasattr(model, 'qg'):
            for gen_id in model.Gens:
                gen = next((g for g in grid.Gens if g.ID == gen_id), None) if hasattr(grid, 'Gens') else None
                p_val = value(model.pg[gen_id, timestep]) if timestep in model.T else 0.0
                q_val = value(model.qg[gen_id, timestep]) if timestep in model.T else 0.0
                gen_data.append({
                    'GenID': gen_id,
                    'BusID': gen.BusID if gen else 'N/A',
                    'P_pu': p_val,
                    'Q_pu': q_val
                })
        gen_df = pd.DataFrame(gen_data)
        gen_file = os.path.join(directory, f"gen_results_t{timestep}.csv")
        gen_df.to_csv(gen_file, index=False)
        print(f"Save {gen_file}, number of data lines: {len(gen_data)}, absolute path: {os.path.abspath(gen_file)}")

        # Save line power results (if there is a model object)
        line_data = []
        if model and hasattr(model, 'Lines') and hasattr(model, 'P') and hasattr(model, 'Q'):
            for line_id in model.Lines:
                line = next((l for l in grid.Lines if l.ID == line_id), None) if hasattr(grid, 'Lines') else None
                p_val = value(model.P[line_id, timestep]) if timestep in model.T else 0.0
                q_val = value(model.Q[line_id, timestep]) if timestep in model.T else 0.0
                line_data.append({
                    'LineID': line_id,
                    'FromBus': line.fBus if line else 'N/A',
                    'ToBus': line.tBus if line else 'N/A',
                    'P_pu': p_val,
                    'Q_pu': q_val,
                    'I_pu': 'N/A'
                })
        line_df = pd.DataFrame(line_data)
        line_file = os.path.join(directory, f"line_results_t{timestep}.csv")
        line_df.to_csv(line_file, index=False)
        print(f"Save {line_file}, number of data lines: {len(line_data)}, absolute path: {os.path.abspath(line_file)}")

        # Save solution status and target value
        summary_file = os.path.join(directory, f"summary_t{timestep}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Solver: Linear DistFlow\n")
            f.write(f"Result: {result}\n")
            f.write(f"Objective Value: {objective_value:.4f}\n")
        print(f"Save {summary_file}, absolute path: {os.path.abspath(summary_file)}")
        return True # indicates successful saving
    except Exception as e:
        print(f"An error occurred while saving distflow results (timestep {timestep}): {e}")
        return False # Indicates save failure


def save_opendss_results(grid, timestep, result, value):
    """
    Save the results of OpenDSSSolver.
    :param grid: Distribution grid object
    :param timestep: current time step
    :param result: solution result status (GridSolveResult)
    :param value: target value
    :return: whether saved successfully
    """
    try:
        directory = opendss_dir
        # Save the results of OpenDSSSolver
        bus_data = []
        bus_count_updated = 0
        for bus in grid.Buses if hasattr(grid, 'Buses') else []:
            voltage = bus.V if bus.V is not None else 'N/A'
            if bus.V is not None:
                bus_count_updated += 1
            pd_val = bus.Pd(timestep) if hasattr(bus, 'Pd') and callable(getattr(bus, 'Pd', None)) else (getattr(bus, 'Pd', 'N/A') if hasattr(bus, 'Pd') else 'N/A')
            qd_val = bus.Qd(timestep) if hasattr(bus, 'Qd') and callable(getattr(bus, 'Qd', None)) else (getattr(bus, 'Qd', 'N/A') if hasattr(bus, 'Qd') else 'N/A')
            bus_data.append({
                'BusID': bus.ID,
                'Voltage_pu': voltage,
                'Pd_pu': pd_val,
                'Qd_pu': qd_val
            })
        bus_df = pd.DataFrame(bus_data)
        bus_file = os.path.join(directory, f"bus_results_t{timestep}.csv")
        bus_df.to_csv(bus_file, index=False)
        print(f"Save {bus_file}, number of data lines: {len(bus_data)}, number of voltage updates: {bus_count_updated}, absolute path: {os.path.abspath(bus_file)}")

        line_data = []
        line_count_updated = 0
        for line in grid.Lines if hasattr(grid, 'Lines') else []:
            p_val = line.P if line.P is not None else 'N/A'
            q_val = line.Q if line.Q is not None else 'N/A'
            i_val = line.I if line.I is not None else 'N/A'
            if line.P is not None:
                line_count_updated += 1
            line_data.append({
                'LineID': line.ID,
                'FromBus': line.fBus,
                'ToBus': line.tBus,
                'P_pu': p_val,
                'Q_pu': q_val,
                'I_pu': i_val
            })
        line_df = pd.DataFrame(line_data)
        line_file = os.path.join(directory, f"line_results_t{timestep}.csv")
        line_df.to_csv(line_file, index=False)
        print(f"Save {line_file}, number of data lines: {len(line_data)}, number of power updates: {line_count_updated}, absolute path: {os.path.abspath(line_file)}")

        gen_data = []
        gen_count_updated = 0
        for gen in grid.Gens if hasattr(grid, 'Gens') else []:
            p = gen.P(timestep) if hasattr(gen, 'P') and callable(getattr(gen, 'P', None)) else (getattr(gen, 'P', None) if hasattr(gen, 'P') else None)
            q = gen.Q(timestep) if hasattr(gen, 'Q') and callable(getattr(gen, 'Q', None)) else (getattr(gen, 'Q', None) if hasattr(gen, 'Q') else None)
            if p is not None:
                gen_count_updated += 1
            gen_data.append({
                'GenID': gen.ID,
                'BusID': gen.BusID,
                'P_pu': p if p is not None else 'N/A',
                'Q_pu': q if q is not None else 'N/A'
            })
        gen_df = pd.DataFrame(gen_data)
        gen_file = os.path.join(directory, f"gen_results_t{timestep}.csv")
        gen_df.to_csv(gen_file, index=False)
        print(f"Save {gen_file}, number of data lines: {len(gen_data)}, number of power updates: {gen_count_updated}, absolute path: {os.path.abspath(gen_file)}")

        pvw_data = []
        pvw_count_updated = 0
        for pvw in grid.PVWinds if hasattr(grid, 'PVWinds') else []:
            p_val = pvw.Pr if pvw.Pr is not None else 0.0
            q_val = pvw.Qr if pvw.Qr is not None else 0.0
            if pvw.Pr is not None:
                pvw_count_updated += 1
            pvw_data.append({
                'PVWID': pvw.ID,
                'BusID': pvw.BusID,
                'P_pu': p_val,
                'Q_pu': q_val,
                'Curtailment_Rate': pvw.CR if pvw.CR is not None else 'N/A'
            })
        pvw_df = pd.DataFrame(pvw_data)
        pvw_file = os.path.join(directory, f"pvw_results_t{timestep}.csv")
        pvw_df.to_csv(pvw_file, index=False)
        print(f"Save {pvw_file}, number of data lines: {len(pvw_data)}, number of power updates: {pvw_count_updated}, absolute path: {os.path.abspath(pvw_file)}")

        ess_data = []
        ess_count_updated = 0
        for ess in grid.ESSs if hasattr(grid, 'ESSs') else []:
            p_val = ess.P if ess.P is not None else 0.0
            q_val = ess.Q if ess.Q is not None else 0.0
            if ess.P is not None:
                ess_count_updated += 1
            ess_data.append({
                'ESSID': ess.ID,
                'BusID': ess.BusID,
                'P_pu': p_val,
                'Q_pu': q_val,
                'SOC': ess.SOC if ess.SOC is not None else 'N/A'
            })
        ess_df = pd.DataFrame(ess_data)
        ess_file = os.path.join(directory, f"ess_results_t{timestep}.csv")
        ess_df.to_csv(ess_file, index=False)
        print(f"Save {ess_file}, number of data lines: {len(ess_data)}, number of power updates: {ess_count_updated}, absolute path: {os.path.abspath(ess_file)}")

        # Save solution status and target value
        summary_file = os.path.join(directory, f"summary_t{timestep}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Solver: OpenDSS\n")
            f.write(f"Result: {result}\n")
            f.write(f"Objective Value: {value:.4f}\n")
        print(f"Save {summary_file}, absolute path: {os.path.abspath(summary_file)}")
        return True # indicates successful saving
    except Exception as e:
        print(f"An error occurred while saving opendss results (timestep {timestep}): {e}")
        return False # Indicates save failure


def fix_bus_voltage_limits(grid):
    """
    Correct the voltage range of the bus to ensure that MinV and MaxV are not inf or -inf.
    :param grid: Distribution grid object
    """
    for bus in grid.Buses if hasattr(grid, 'Buses') else []:
        if bus.MaxV == float('inf'):
            bus.MaxV = 1.5 # Set a reasonable upper limit, such as 1.5 pu
        if bus.MinV == float('-inf'):
            bus.MinV = 0.5 # Set a reasonable lower limit, such as 0.5 pu

def _safe_set_device_power(dev, p_value, names=("P", "_p", "Pr", "_pr", "p", "p_pu")):
    """
    Compatibility writing: Try to write the power of the device to common attribute names to be compatible with different versions of fpowerkit / OpenDSS wrapper.
    - dev: device object (Generator/PVWind/ESS/custom)
    - p_value: value to write (usually in pu units)
    - names: List of attribute names to try to set in sequence (the more commonly used ones are placed first)
    Return: (set_any, set_names)
      - set_any: bool, True if at least one attribute is written
      - set_names: list of successfully written attribute names (may be empty)
    Remarks:
      - This function does not throw exceptions (internal capture), ensuring safe calling.
      - If the device object implements the setter method or the property is read-only, this function will skip and continue trying other names.
    """
    set_any = False
    set_names = []
    # If a dict/list/ndarray is passed in, try to convert it into a scalar (take the value of the current time step):
    try:
        # Only take the first element if p_value is obviously indexable and the first element is a scalar
        if not (isinstance(p_value, (int, float))):
            # Try to convert numpy scalar / 0-d array to python float
            import numpy as _np
            if isinstance(p_value, _np.ndarray) and p_value.shape == ():
                p_value = float(p_value)
    except Exception:
        # Ignore conversion failure and continue using original p_value
        pass

    for attr_name in names:
        try:
            # If the object already has this attribute, try to write it directly
            if hasattr(dev, attr_name):
                try:
                    setattr(dev, attr_name, p_value)
                    set_any = True
                    set_names.append(attr_name)
                    # Without break, let the function write as many fields as possible (to improve compatibility)
                except Exception:
                    # Some attributes may be read-only or have verification, and will throw an exception and skip it.
                    continue
            else:
                # If there is no such attribute, try to create it (most python objects can add attributes dynamically)
                try:
                    setattr(dev, attr_name, p_value)
                    set_any = True
                    set_names.append(attr_name)
                except Exception:
                    # This attribute cannot be created (such as an object using __slots__), skip
                    continue
        except Exception:
            # Ignore any exceptions and continue trying the next attribute name
            continue
    return set_any, set_names

def update_grid_from_model(grid, baseline_data, timestep):
    """
    Update the grid object by extracting data from the baseline_data dictionary and using _safe_set_device_power to ensure compatibility.
    """

    # --- Update Generator ---
    gen_powers = baseline_data.get("gen_powers")
    if gen_powers is None: raise KeyError("Cannot find 'gen_powers' key in baseline_data!")
    gen_q_powers = baseline_data.get("gen_q_powers", {})
    for gen in grid.Gens:
        # If the generator ID currently traversed is not in the first stage of optimization results
        # (This is usually our virtual generator gen_for_slack_bus)
        # Then skip it directly without performing any operation on it.
        if gen.ID not in gen_powers:
            continue
        p_val = gen_powers[gen.ID][timestep]
        q_series = gen_q_powers.get(gen.ID)
        q_val = q_series[timestep] if q_series is not None and timestep < len(q_series) else 0.0
        _safe_set_device_power(gen, p_val, names=("P", "_p")) # Use safe writing
        _safe_set_device_power(gen, q_val, names=("Q", "_q"))

    # --- Update PVWind ---
    pvw_powers = baseline_data.get("pvw_powers")
    if pvw_powers is None: raise KeyError("Cannot find 'pvw_powers' key in baseline_data!")
    for pvw in grid.PVWinds:
        if pvw.ID not in pvw_powers: raise KeyError(f"The data of PV/Wind '{pvw.ID}' cannot be found in pvw_powers!")
        p_val = pvw_powers[pvw.ID][timestep]
        _safe_set_device_power(pvw, p_val, names=("Pr", "_pr", "P")) # Use safe writing
        tan_phi = (1 - pvw.PF ** 2) ** 0.5 / pvw.PF if pvw.PF != 0 else 0
        q_val = p_val * tan_phi
        _safe_set_device_power(pvw, q_val, names=("Qr", "_qr", "Q"))

    # --- Update ESS ---
    ess_powers = baseline_data.get("ess_powers")
    if ess_powers is None: raise KeyError("'ess_powers' key not found in baseline_data!")
    for ess in grid.ESSs:
        if ess.ID not in ess_powers: raise KeyError(f"The data of ESS '{ess.ID}' cannot be found in ess_powers!")
        p_val = ess_powers[ess.ID][timestep]
        _safe_set_device_power(ess, p_val, names=("P", "_p")) # Use safe writing

    # The checking logic of SOP/NOP remains unchanged
    if baseline_data.get("sop_flows") is None: raise KeyError("'sop_flows' key not found in baseline_data!")
    if baseline_data.get("nop_status") is None: raise KeyError("'nop_status' key not found in baseline_data!")



def check_grid_attributes(grid, timestep, stage="before"):
    """
    Check the property values of the grid object and print debugging information.
    :param grid: Distribution grid object
    :param timestep: current time step
    :param stage: check stage ('before' or 'after' OpenDSSSolver)
    """
    print(f"Check grid object property value ({stage} OpenDSSSolver, time step {timestep}):")

    # Check Bus voltage
    bus_updated = 0
    for bus in grid.Buses if hasattr(grid, 'Buses') else []:
        if bus.V is not None:
            bus_updated += 1
    print(f" Number of bus voltage updates: {bus_updated}/{len(grid.Buses if hasattr(grid, 'Buses') else [])}")

    # Check Generator power
    gen_updated = 0
    for gen in grid.Gens if hasattr(grid, 'Gens') else []:
        p = gen.P(timestep) if hasattr(gen, 'P') and callable(getattr(gen, 'P', None)) else (getattr(gen, 'P', None) if hasattr(gen, 'P') else None)
        if p is not None:
            gen_updated += 1
    print(f" Generator power update quantity: {gen_updated}/{len(grid.Gens if hasattr(grid, 'Gens') else [])}")

    # Check PVWind power
    pvw_updated = 0
    for pvw in grid.PVWinds if hasattr(grid, 'PVWinds') else []:
        if pvw.Pr is not None:
            pvw_updated += 1
    print(f" Photovoltaic/wind power updated number: {pvw_updated}/{len(grid.PVWinds if hasattr(grid, 'PVWinds') else [])}")

    # Check ESS power
    ess_updated = 0
    for ess in grid.ESSs if hasattr(grid, 'ESSs') else []:
        if ess.P is not None:
            ess_updated += 1
    print(f" Energy storage system power update quantity: {ess_updated}/{len(grid.ESSs if hasattr(grid, 'ESSs') else [])}")

    # Check Line power
    line_updated = 0
    for line in grid.Lines if hasattr(grid, 'Lines') else []:
        if line.P is not None:
            line_updated += 1
    print(f" Number of line power updates: {line_updated}/{len(grid.Lines if hasattr(grid, 'Lines') else [])}")
