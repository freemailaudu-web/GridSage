import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import pandas as pd
# Set up matplotlib to display Chinese and negative signs correctly
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


import pandas as pd
import numpy as np
import os
import re


def export_simulation_data_to_excel(baseline_data, stations_list, params):
    """
    Export key data (scheduling events, power of each pile, total power) in Baseline mode to an Excel file.

    Args:
        baseline_data (dict): Dictionary of results returned from solve_baseline.
        stations_list (list): List containing all GEVStation objects.
        params (dict): Dictionary containing simulation configuration.
    """
    print("Start exporting Baseline key data to Excel...")

    # Make sure the output directory exists
    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    # Name the new comprehensive file
    excel_path = os.path.join(output_dir, "baseline_simulation_summary.xlsx")

    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

            # Reason: Traverse all charging stations, not just the first one, to support data export in multi-station scenarios.
            # --- Worksheet 1: Charging event schedule ---
            if stations_list:
                all_events_data = []
                # Traverse all charging stations
                for station in stations_list:
                    # Traverse all charging sessions at this station
                    for event in station.daily_sessions:
                        all_events_data.append({
                            "Charging Station ID (Station_ID)": station.station_id,
                            "Charging Pile ID (Spot_ID)": event.spot_id + 1,
                            "Arrival Time (Arrival_Hour)": event.arrival_hour,
                            "Departure Time (Departure_Hour)": event.departure_hour,
                            "Duration (Hours)": event.departure_hour - event.arrival_hour
                        })

                if all_events_data:
                    # Sort by charging station ID, charging pile ID and arrival time
                    df_schedule = pd.DataFrame(all_events_data).sort_values(
                        by=["Charging Station ID (Station_ID)", "Charging Pile ID (Spot_ID)", "Arrival Time (Arrival_Hour)"]
                    )
                    df_schedule.to_excel(writer, sheet_name='Charging_Event_Schedule', index=False)
                    print(f"...{len(all_events_data)} charging events have been written to the worksheet 'Charging_Event_Schedule'")
                else:
                    print("Warning: No charging events at any charging stations.")
            else:
                print("Warning: The charging station list is empty and the scheduling event table cannot be generated.")

            # --- Worksheet 2: Power curve of each charging pile (data source of baseline_ev_spot_powers graph) ---
            if "spot_powers" in baseline_data and baseline_data["spot_powers"]:
                spot_powers_pu = baseline_data["spot_powers"]
                time_steps = len(next(iter(spot_powers_pu.values())))
                start_hour = params.get('start_hour', 0)
                end_hour = params.get('end_hour', 24)
                time_axis = np.linspace(start_hour, end_hour, time_steps)
                base_power_mva = params.get('base_power', 1.0)

                df_individual_data = {'Time (Time_h)': time_axis}
                for spot_id, powers in spot_powers_pu.items():
                    powers_kw = [p * base_power_mva * 1000 for p in powers]
                    df_individual_data[f'charging pile_{spot_id + 1}_power (kW)'] = powers_kw

                df_individual = pd.DataFrame(df_individual_data)
                df_individual.to_excel(writer, sheet_name='Individual_Spot_Powers', index=False)
                print(f"...The power data of each charging pile has been written into the worksheet 'Individual_Spot_Powers'")
            else:
                print("Warning: 'spot_powers' data not found, skipping power curve export.")

            # --- Sheet 2: Total Charge/Discharge Power Split ---
            # Convert dictionary to numpy array for easy calculation
            all_powers_pu = np.array(list(spot_powers_pu.values()))  # shape: (num_spots, time_steps)

            # Calculate the total charging power (only keep positive values)
            charge_powers_pu = all_powers_pu.copy()
            charge_powers_pu[charge_powers_pu < 0] = 0
            total_charge_kw = charge_powers_pu.sum(axis=0) * base_power_mva * 1000

            # Calculate the total discharge power (only keep negative values, and then take the absolute value)
            discharge_powers_pu = all_powers_pu.copy()
            discharge_powers_pu[discharge_powers_pu > 0] = 0
            total_discharge_kw = np.abs(discharge_powers_pu.sum(axis=0)) * base_power_mva * 1000

            df_total_separated = pd.DataFrame({
                'Time (h)': time_axis,
                'Total_Charge_Power (kW)': total_charge_kw,
                'Total_Discharge_Power (kW)': total_discharge_kw
            })
            df_total_separated.to_excel(writer, sheet_name='Total_Load_Separated', index=False)
            print(f"...The total charge and discharge load data has been written to the worksheet 'Total_Load_Separated'")

        print(f"Success! Baseline EV power data has been saved to: {os.path.abspath(excel_path)}")

    except Exception as e:
        print(f"Error: Export to Excel failed - {e}")

def plot_voltage_snapshots(all_ts_data, seed, gui_params):
    """
    [Two-stage comparison version] Generate an independent bus voltage comparison diagram for each time step.
    - solid line: first stage approximation voltage (DistFlow / RL Approx)
    - dashed line: second stage precision voltage (OpenDSS)
    - Each Algorithm uses a fixed color
    """
    print("--- Generating [two-stage comparison version] voltage time-sharing snapshot graph ---")

    # 1. Preparation
    try:
        num_steps = len(next(iter(all_ts_data.values()))['step_costs'])
        if num_steps == 0: raise IndexError
    except (StopIteration, KeyError, IndexError):
        print("Warning: Unable to determine valid number of steps, skipping voltage snapshot plot drawing.")
        return

    # —— Color mapping (family level) + automatic allocation ——
    algorithms = list(all_ts_data.keys())

    def _normalize_algo_name(name: str) -> str:
        s = name.strip()
        # Remove modifications such as Two_Stage / Single_Stage / Seed and unify it as the family name
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

    # Construct the mapping of "Algorithm name → Color" for this chart (unknown Algorithm is automatically assigned)
    unknown_keys = sorted([
        _normalize_algo_name(a) for a in algorithms
        if BASE_COLOR_MAP.get(_normalize_algo_name(a)) is None and _normalize_algo_name(a) != 'BASELINE'
    ])

    algo_colors = {}
    for algo_name in algorithms:
        key = _normalize_algo_name(algo_name)
        if key == 'BASELINE':
            algo_colors[algo_name] = BASE_COLOR_MAP['BASELINE']
            continue
        color = BASE_COLOR_MAP.get(key)
        if color is None:
            idx = unknown_keys.index(key) % len(FALLBACK_PALETTE)
            color = FALLBACK_PALETTE[idx]
        algo_colors[algo_name] = color

    # 2. Create a separate output folder
    output_dir = os.path.join("results_outputs", f"voltage_snapshots_comparison_seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    # 3. Traverse each time step and generate a picture
    for t in range(num_steps):
        fig, ax = plt.subplots(figsize=(20, 10))

        all_bus_ids = set()
        # Collect the voltage data of all algorithms and all stages at this time step to determine the bus ID and Y-axis range
        voltages_in_step_for_ylim = []
        for algo_name, ts_data in all_ts_data.items():
            voltages_s1 = ts_data.get('voltages_data_stage1', {})
            voltages_s2 = ts_data.get('voltages_data_stage2', {})
            if voltages_s1: all_bus_ids.update(voltages_s1.keys())
            if voltages_s2: all_bus_ids.update(voltages_s2.keys())

            if voltages_s1 and t < len(next(iter(voltages_s1.values()))):
                voltages_in_step_for_ylim.extend([v_list[t] for v_list in voltages_s1.values()])
            if voltages_s2 and t < len(next(iter(voltages_s2.values()))):
                voltages_in_step_for_ylim.extend([v_list[t] for v_list in voltages_s2.values()])

        if not all_bus_ids:
            plt.close(fig);
            continue

        # Natural sorting of bus IDs
        sorted_bus_ids = sorted(list(all_bus_ids), key=lambda b: int(''.join(filter(str.isdigit, b)) or 0))

        # 4. Draw two lines for each Algorithm
        for algo_name in algorithms:
            ts_data = all_ts_data[algo_name]
            color = algo_colors.get(algo_name, 'gray') # Get Algorithm color

            # --- Draw the first stage (solid line) ---
            voltages_s1_dict = ts_data.get('voltages_data_stage1', {})
            if voltages_s1_dict:
                voltages_s1 = [
                    voltages_s1_dict.get(bus_id, [])[t] if t < len(voltages_s1_dict.get(bus_id, [])) else np.nan for
                    bus_id in sorted_bus_ids]
                ax.plot(sorted_bus_ids, voltages_s1, color=color, linestyle='-', marker='o', markersize=4,
                        label=f'{algo_name} (Stage 1 - Approx)')

            # --- Draw the second stage (dotted line) ---
            voltages_s2_dict = ts_data.get('voltages_data_stage2', {})
            if voltages_s2_dict:
                voltages_s2 = [
                    voltages_s2_dict.get(bus_id, [])[t] if t < len(voltages_s2_dict.get(bus_id, [])) else np.nan for
                    bus_id in sorted_bus_ids]
                ax.plot(sorted_bus_ids, voltages_s2, color=color, linestyle='--', marker='x', markersize=5,
                        label=f'{algo_name} (Stage 2 - OpenDSS)')

        # 5. Beautification and preservation
        if voltages_in_step_for_ylim:
            min_v = np.nanmin(voltages_in_step_for_ylim)
            max_v = np.nanmax(voltages_in_step_for_ylim)
            padding = (max_v - min_v) * 0.05 if (max_v - min_v) > 0.01 else 0.01
            ax.set_ylim(min_v - padding, max_v + padding)

        ax.set_xlabel('bus ID', fontsize=14)
        ax.set_ylabel('Voltage (pu)', fontsize=14)
        ax.set_title(f'Two-stage voltage comparison (time step: {t}, scene Seed: {seed})', fontsize=18)
        plt.xticks(rotation=90, fontsize=10)
        ax.legend(fontsize=12)
        ax.grid(True, axis='y', linestyle=':')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'voltage_comparison_t{t}.png')
        plt.savefig(save_path)
        plt.close(fig)

    print(f"✅ The two-stage voltage comparison snapshot has been saved to the folder: {os.path.abspath(output_dir)}")

def plot_line_flow_snapshots(all_ts_data, seed, gui_params):
    """
    Generate an independent line power flow comparison diagram (histogram) for each time step and save it in a new folder.
    """
    print("--- Generating time-sharing snapshot comparison chart of line power flow ---")

    # 1. Preparation
    try:
        num_steps = len(next(iter(all_ts_data.values()))['step_costs'])
        algorithms = list(all_ts_data.keys())
        if num_steps == 0: raise IndexError
    except (StopIteration, KeyError, IndexError):
        print("Warning: Unable to determine the valid number of steps, skip drawing the power flow snapshot diagram.")
        return

    # 2. Create a separate output folder
    output_dir = os.path.join("results_outputs", f"line_flow_snapshots_seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    # 3. Traverse each time step and generate a picture
    for t in range(num_steps):
        fig, ax = plt.subplots(figsize=(18, 10))

        all_line_ids = set()
        step_data = {}
        for algo_name, ts_data in all_ts_data.items():
            flows_dict = ts_data.get('line_powers_data', {})
            if not flows_dict: continue

            current_step_flows = {line: p_list[t] for line, p_list in flows_dict.items() if t < len(p_list)}
            step_data[algo_name] = current_step_flows
            all_line_ids.update(current_step_flows.keys())

        if not all_line_ids:
            plt.close(fig)
            continue

        sorted_line_ids = sorted(list(all_line_ids), key=lambda l: int(''.join(filter(str.isdigit, l)) or 0))

        # 4. Draw a histogram for comparison
        num_algorithms = len(algorithms)
        bar_width = 0.8 / num_algorithms
        x_indices = np.arange(len(sorted_line_ids))

        for i, algo_name in enumerate(algorithms):
            flows = [step_data.get(algo_name, {}).get(line_id, np.nan) for line_id in sorted_line_ids]
            offset = i * bar_width - (0.8 - bar_width) / 2
            ax.bar(x_indices + offset, flows, bar_width, label=algo_name)

        # 5. Beautification and preservation
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel('Active Power Flow P (pu)')
        ax.set_title(f'Comparison of active power flow of each Algorithm line (time step: {t}, scene Seed: {seed})', fontsize=16)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(sorted_line_ids, rotation=90)
        ax.legend()
        ax.grid(True, axis='y', linestyle=':')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'line_flow_snapshot_t{t}.png')
        plt.savefig(save_path)
        plt.close(fig)

    print(f"✅ The line flow time-sharing snapshot has been saved to the folder: {os.path.abspath(output_dir)}")

def plot_line_flows(all_ts_data, seed, gui_params):
    """Draw timing diagrams for all line active power flows and display the envelopes of the maximum/minimum power flows.
    Uses the same robust logic as plot_voltage_profiles.
    """
    print("Generating optimized version of line power flow timing diagram...")
    try:
        # Determine the total number of steps
        num_steps = len(next(iter(all_ts_data.values()))['step_costs'])
        if num_steps == 0: raise IndexError
    except (StopIteration, KeyError, IndexError):
        print("Warning: Unable to determine valid number of steps, skip drawing of power flow diagram.")
        return

    start_hour, end_hour = gui_params['start_hour'], gui_params['end_hour']
    time_axis = np.linspace(start_hour, end_hour, num_steps)

    fig, axes = plt.subplots(len(all_ts_data), 1, figsize=(15, 8 * len(all_ts_data)), sharex=True, squeeze=False)
    fig.suptitle(f'Comparison of active power flow distribution of each Algorithm line (scenario Seed: {seed})', fontsize=16)

    for i, (algo_name, ts_data) in enumerate(all_ts_data.items()):
        ax = axes[i, 0]

        # Correctly obtain the unified format "dict of lists"
        line_powers_dict = ts_data.get('line_powers_data', {})

        if not line_powers_dict:
            ax.text(0.5, 0.5, 'No power flow data', horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            continue

        # Convert dictionary values to Numpy arrays for envelope calculation
        # Ensure that all list lengths are consistent with num_steps to avoid drawing errors
        all_flows_matrix = np.array([p for p in line_powers_dict.values() if len(p) == num_steps])

        # Draw the power flow curve of each line (gray background)
        for line_id, p_list in line_powers_dict.items():
            if len(p_list) == num_steps:
                ax.plot(time_axis, p_list, color='gray', alpha=0.3)

        # Draw the envelope of maximum and minimum power flow
        if all_flows_matrix.size > 0:
            min_p = np.nanmin(all_flows_matrix, axis=0)
            max_p = np.nanmax(all_flows_matrix, axis=0)
            ax.plot(time_axis, max_p, color='red', linestyle='--', label=f'maximum power flow ({np.nanmax(max_p):.3f} pu)')
            ax.plot(time_axis, min_p, color='blue', linestyle='--', label=f'minimum power flow ({np.nanmin(min_p):.3f} pu)')
            ax.fill_between(time_axis, min_p, max_p, color='lightgreen', alpha=0.3)
        # ▲▲▲▲▲ [Correction completed] ▲▲▲▲▲

        ax.set_title(f'Algorithm: {algo_name}')
        ax.set_ylabel('Active Power Flow P (pu)')
        ax.legend()
        ax.grid(True, linestyle=':')
        ax.axhline(0, color='black', linewidth=0.5)

    axes[-1, 0].set_xlabel('Time (hours)')
    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'line_flows_seed_{seed}.png')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path)
    plt.close()
    print(f"The line flow diagram of the scenario (Seed: {seed}) has been saved to: {save_path}")
def plot_ev_spot_powers(baseline_data, gui_params):
    """
    Based on the calculation results of Baseline, draw and save the EV charging pile power change graph.

    Parameters:
    - baseline_data (dict): Dictionary of detailed results returned from the solve_baseline function.
    - gui_params (dict): Dictionary of parameters passed in from the GUI, used to obtain time information.
    """
    print("Generating Baseline EV charging pile power visualization diagram...")

    spot_powers_data = baseline_data.get("spot_powers")
    if not spot_powers_data:
        print("Warning: 'spot_powers' data not found in Baseline results, plotting skipped.")
        return

    # Extract data from dictionary
    # The format of spot_powers_data is {spot_id: [p1, p2, ...]}
    # We need to convert it into an array of (steps, spots)
    num_spots = len(spot_powers_data)
    if num_spots == 0:
        print("Warning: The number of charging piles is 0, skip drawing.")
        return

    # Get the number of time steps and unify the data length of all charging piles
    num_steps = 0
    power_lists = []
    # Ensure that some charging piles can be processed even if they have no data
    for i in range(max(spot_powers_data.keys()) + 1):
        power_list = spot_powers_data.get(i, [])
        if len(power_list) > num_steps:
            num_steps = len(power_list)
        power_lists.append(power_list)

    # Fill in data that may be of different lengths
    for i, p_list in enumerate(power_lists):
        if len(p_list) < num_steps:
            power_lists[i] = np.pad(p_list, (0, num_steps - len(p_list)), 'constant')

    powers_array_pu = np.array(power_lists).T # Transpose to (steps, spots)

    # Convert unit per unit (pu) to kilowatt (kW)
    base_power_kw = gui_params.get('base_power', 1.0) * 1000
    powers_array_kw = powers_array_pu * base_power_kw

    # Create timeline
    start_hour = gui_params.get('start_hour', 0)
    end_hour = gui_params.get('end_hour', 24)
    time_axis = np.linspace(start_hour, end_hour, num_steps)

    # --- Start drawing ---
    # Create a subgraph for each charging pile
    fig, axes = plt.subplots(num_spots, 1, figsize=(14, 2 * num_spots), sharex=True, squeeze=False)
    fig.suptitle('Power change curve of each charging pile in Baseline mode', fontsize=18, y=0.95)

    for i in range(num_spots):
        ax = axes[i, 0]
        ax.plot(time_axis, powers_array_kw[:, i], label=f'Charging pile #{i + 1}')
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8) # Zero power reference line
        ax.set_ylabel('Power (kW)')
        ax.legend(loc='upper right')
        ax.grid(True)

    axes[-1, 0].set_xlabel('Time (hours)')

    # Save image
    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'baseline_ev_spot_powers.png')

    plt.tight_layout(rect=[0, 0, 1, 0.93]) # Adjust the layout to fit the main title
    plt.savefig(save_path)
    plt.close()

    print(f"The visualization has been successfully saved to: {save_path}")


def generate_baseline_reports(baseline_data, gui_params):
    """
    Based on the results of the Baseline, generate voltage/power flow diagrams by time period,
    Draw the total EV load curve and save detailed data to Excel.
    """
    print("Generating detailed visual reports and data files of Baseline...")

    # --- Prepare data and directory ---
    voltages_data = baseline_data.get("bus_voltages")
    line_flows_data = baseline_data.get("line_powers")
    spot_powers_data = baseline_data.get("spot_powers")
    output_dir_base = "results_outputs"
    os.makedirs(output_dir_base, exist_ok=True)

    if not voltages_data or not line_flows_data or not spot_powers_data:
        print("Warning: The result data is incomplete and a detailed report cannot be generated.")
        return

    num_steps = len(next(iter(voltages_data.values())))
    time_axis = np.linspace(
        gui_params.get('start_hour', 0),
        gui_params.get('end_hour', 24),
        num_steps
    )
    base_power_kw = gui_params.get('base_power', 1.0) * 1000


    # --- 1. Draw and save the voltage distribution diagram of each time step ---
    voltage_plot_dir = os.path.join(output_dir_base, "baseline_voltage_profiles")
    os.makedirs(voltage_plot_dir, exist_ok=True)

    # Preprocessing and sorting node ID
    def get_bus_sort_key(bus_id):
        try:
            return int(''.join(filter(str.isdigit, bus_id)))
        except:
            return bus_id

    sorted_buses = sorted(voltages_data.keys(), key=get_bus_sort_key)

    for t in range(num_steps):
        fig, ax = plt.subplots(figsize=(16, 9))
        voltages_at_t = [voltages_data[bus][t] for bus in sorted_buses]

        ax.plot(sorted_buses, voltages_at_t, marker='o', linestyle='-', color='dodgerblue')
        ax.set_title(f'Baseline voltage distribution (time step t={t})', fontproperties="SimHei", size=18)
        ax.set_xlabel('Node ID (after sorting)', fontproperties="SimHei", size=12)
        ax.set_ylabel('voltage amplitude (pu)', fontproperties="SimHei", size=12)
        ax.grid(True)
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()
        plt.savefig(os.path.join(voltage_plot_dir, f"voltage_profile_t_{t}.png"))
        plt.close(fig)
    print(f"The voltage distribution timing diagram has been saved to: {voltage_plot_dir}")

    # --- 2. Draw and save the line flow diagram at each time step ---
    flow_plot_dir = os.path.join(output_dir_base, "baseline_line_flows")
    os.makedirs(flow_plot_dir, exist_ok=True)

    def get_line_sort_key(line_id):
        try:
            return int(''.join(filter(str.isdigit, line_id)))
        except:
            return line_id

    sorted_lines = sorted(line_flows_data.keys(), key=get_line_sort_key)

    for t in range(num_steps):
        fig, ax = plt.subplots(figsize=(16, 9))
        flows_at_t = [line_flows_data[line][t] for line in sorted_lines]

        ax.bar(sorted_lines, flows_at_t, color='mediumseagreen')
        ax.axhline(0, color='grey', linewidth=0.8)
        ax.set_title(f'Baseline line active power flow (time step t={t})', fontproperties="SimHei", size=18)
        ax.set_xlabel('Branch ID (after sorting)', fontproperties="SimHei", size=12)
        ax.set_ylabel('Active power flow (pu)', fontproperties="SimHei", size=12)
        ax.grid(axis='y')
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()
        plt.savefig(os.path.join(flow_plot_dir, f"line_flow_t_{t}.png"))
        plt.close(fig)
    print(f"Line power flow timing diagram has been saved to: {flow_plot_dir}")

    # --- 3. Save detailed data to Excel file ---
    excel_path = os.path.join(output_dir_base, 'baseline_detailed_data.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Save voltage data
        volt_df = pd.DataFrame(voltages_data)
        volt_df = volt_df[sorted_buses] # Arrange each column according to the sorted node ID
        volt_df.index.name = "Time_Step"
        volt_df.to_excel(writer, sheet_name='Bus_Voltages')

        # Save power flow data
        flow_df = pd.DataFrame(line_flows_data)
        flow_df = flow_df[sorted_lines] # Arrange each column according to the sorted branch ID
        flow_df.index.name = "Time_Step"
        flow_df.to_excel(writer, sheet_name='Line_Flows')

    print(f"Detailed voltage and power flow data have been saved to Excel file: {excel_path}")
    #Draw and save the total charging load curve
    print("...generating the total charging load curve graph")
    try:
        if spot_powers_data:
            # a. Convert spot_powers_data dictionary to Numpy array of (timesteps, spots)
            num_spots = max(spot_powers_data.keys()) + 1 if spot_powers_data else 0
            powers_array_pu = np.zeros((num_steps, num_spots))
            for spot_idx, power_list in spot_powers_data.items():
                if len(power_list) == num_steps:
                    powers_array_pu[:, spot_idx] = power_list

            # b. Calculate the total charging power (only positive values are added) and the total discharge power (only negative values are added)
            total_charge_power_pu = np.sum(np.where(powers_array_pu > 0, powers_array_pu, 0), axis=1)
            total_discharge_power_pu = np.sum(np.where(powers_array_pu < 0, powers_array_pu, 0), axis=1)

            # c. Convert to kW
            total_charge_power_kw = total_charge_power_pu * base_power_kw
            total_discharge_power_kw = total_discharge_power_pu * base_power_kw

            # d. Start drawing
            fig, ax = plt.subplots(figsize=(14, 7))

            # e. Draw the charging curve and filled area in the upper part
            ax.plot(time_axis, total_charge_power_kw, label='total charging power', color='darkviolet', linewidth=2)
            ax.fill_between(time_axis, total_charge_power_kw, 0, color='darkviolet', alpha=0.3)

            # f. Draw the discharge curve and filled area in the lower half
            ax.plot(time_axis, total_discharge_power_kw, label='Total V2G discharge power', color='green', linewidth=2)
            ax.fill_between(time_axis, total_discharge_power_kw, 0, color='green', alpha=0.3)

            ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
            ax.set_title('Baseline total EV load change curve (charge/discharge separation)', fontproperties="SimHei", size=18)
            ax.set_xlabel('Time (hours)', fontproperties="SimHei", size=12)
            ax.set_ylabel('Total power (kW)', fontproperties="SimHei", size=12)
            ax.legend()
            ax.grid(True)
            fig.tight_layout()

            # g. Save image
            save_path = os.path.join(output_dir_base, 'baseline_total_ev_load_separated.png')
            plt.savefig(save_path)
            plt.close(fig)
            print(f"The separated total charging load curve has been saved to: {save_path}")

    except Exception as e:
        print(f"Error: Failed to generate total charging load curve - {e}")
import matplotlib.pyplot as plt
import os


def plot_spot_schedule_gantt(stations_list, target_spot_id=0):
    """
    Generate and save a Gantt chart of charging events successfully assigned in the simulation for a single charging station.

    Args:
        stations_list (list): List containing all GEVStation objects.
        target_spot_id (int): ID of the target charging pile (index starts from 0).
    """
    print(f"Generating dispatch distribution map for charging pile # {target_spot_id + 1}...")

    # In a multi-charging station scenario, we assume that the first charging station is analyzed
    if not stations_list:
        print("Warning: The charging station list is empty and the dispatch map cannot be generated.")
        return
    station = stations_list[0]

    # 1. Filter out all charging events at the target charging pile
    spot_events = [s for s in station.daily_sessions if s.spot_id == target_spot_id]

    if not spot_events:
        print(f"Analysis completed: Charging pile # {target_spot_id + 1} does not have any charging events in this simulation.")
        return

    # Sort events by arrival time
    spot_events.sort(key=lambda x: x.arrival_hour)

    # 2. Create images and axes
    fig, ax = plt.subplots(figsize=(16, 9))

    # 3. Draw each event as a horizontal bar
    for i, event in enumerate(spot_events):
        start = event.arrival_hour
        duration = event.departure_hour - event.arrival_hour
        # Use hsv color mapping to make each bar a different color and clearer
        color = plt.cm.viridis(i / len(spot_events))
        ax.barh(y=f"Event #{i + 1}", width=duration, left=start, height=0.5, color=color, edgecolor="black")
        # Display time inside the bar
        ax.text(start + duration / 2, f"Event #{i + 1}", f"{start:g}h - {event.departure_hour:g}h",
                ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    # 4. Beautify charts
    ax.set_xlabel("Time of the day (hours)", fontsize=12)
    ax.set_ylabel(f"Charging event assigned to charging pile # {target_spot_id + 1}", fontsize=12)
    ax.set_title(f"Charging pile #{target_spot_id + 1} all-day task scheduling Gantt chart", fontsize=16, fontweight='bold')
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 1))
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # 5. Save the image to the `results_outputs` folder
    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f'spot_{target_spot_id + 1}_schedule.png')

    try:
        plt.savefig(output_filename)
        print(f"Success! The scheduling map has been saved as: {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"Error: Failed to save image - {e}")

    plt.close(fig) # Close the image to prevent it from staying in the background

def plot_ess_soc(baseline_data, params):
    """
    Draw SOC timing diagrams of all energy storage systems (ESS).
    """
    if "ess_soc" not in baseline_data or not baseline_data["ess_soc"]:
        print("Message: 'ess_soc' data not found in baseline_data, skipping ESS image generation.")
        return

    print("Generating ESS SOC timing diagram...")
    ess_soc_data = baseline_data["ess_soc"]

    # SOC data contains T+1 time points
    time_steps = len(next(iter(ess_soc_data.values())))
    start_hour = params.get('start_hour', 0)
    end_hour = params.get('end_hour', 24)
    # Create an X-axis matching T+1 points
    time_axis = np.linspace(start_hour, end_hour, time_steps)

    plt.figure(figsize=(12, 6))
    for ess_id, soc_list in ess_soc_data.items():
        # Convert SOC from pu to percentage
        soc_percent = [s * 100 for s in soc_list]
        plt.plot(time_axis, soc_percent, marker='.', linestyle='-', label=f"ESS #{ess_id}")

    plt.title("Energy Storage System (ESS) SOC Timing Diagram", fontsize=16)
    plt.xlabel("Time (hours)")
    plt.ylabel("State of Charge (SOC %)")
    plt.xlim(start_hour, end_hour)
    plt.ylim(0, 105)
    plt.grid(True)
    plt.legend()

    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, 'ess_soc_timeseries.png')
    plt.savefig(output_filename)
    plt.close()
    print(f"Success! ESS SOC image has been saved to: {os.path.abspath(output_filename)}")


def plot_sop_flows(baseline_data, params):
    """
    Draw power flow timing diagrams for all soft switching (SOP).
    """
    if "sop_flows" not in baseline_data or not baseline_data["sop_flows"]:
        print("Message: 'sop_flows' data not found in baseline_data, skipping SOP image generation.")
        return

    print("Generating SOP power timing diagram...")
    sop_flow_data = baseline_data["sop_flows"]
    num_sops = len(sop_flow_data)

    time_steps = len(next(iter(sop_flow_data.values()))['P1'])
    start_hour = params.get('start_hour', 0)
    end_hour = params.get('end_hour', 24)
    time_axis = np.linspace(start_hour, end_hour, time_steps)

    fig, axes = plt.subplots(num_sops, 1, figsize=(12, 4 * num_sops), sharex=True, squeeze=False)
    fig.suptitle("Soft switching (SOP) power timing diagram", fontsize=16, y=0.95)

    for i, (sop_id, flows) in enumerate(sop_flow_data.items()):
        ax = axes[i, 0]
        ax.plot(time_axis, flows['P1'], marker='o', linestyle='-', label='Active power P (pu)')
        ax.plot(time_axis, flows['Q1'], marker='x', linestyle='--', label='Reactive power Q1 (pu)')
        if 'Q2' in flows:
            ax.plot(time_axis, flows['Q2'], marker='s', linestyle=':', label='Reactive power Q2 (pu)')
        ax.set_title(f"SOP #{sop_id}")
        ax.set_ylabel("Power (pu)")
        ax.grid(True)
        ax.legend()

    plt.xlabel("Time (hours)")

    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, 'sop_flows_timeseries.png')
    plt.savefig(output_filename)
    plt.close()
    print(f"Success! The SOP power image has been saved to: {os.path.abspath(output_filename)}")


def plot_nop_status(baseline_data, params):
    """
    Draw the breaking state timing diagram of all normally open points (NOP).
    """
    if "nop_status" not in baseline_data or not baseline_data["nop_status"]:
        print("Message: 'nop_status' data not found in baseline_data, skipping NOP image generation.")
        return

    print("Generating NOP breaking state timing diagram...")
    nop_status_data = baseline_data["nop_status"]

    time_steps = len(next(iter(nop_status_data.values())))
    start_hour = params.get('start_hour', 0)
    end_hour = params.get('end_hour', 24)
    time_axis = np.linspace(start_hour, end_hour, time_steps)

    plt.figure(figsize=(12, 6))
    for nop_id, status_list in nop_status_data.items():
        # Use step plot to express state switching more clearly
        plt.step(time_axis, status_list, where='post', label=f"NOP #{nop_id}")

    plt.title("Normally open point (NOP) breaking state timing diagram", fontsize=16)
    plt.xlabel("Time (hours)")
    plt.ylabel("status")
    plt.yticks([0, 1], ["Off", "On"])
    plt.ylim(-0.1, 1.1)
    plt.xlim(start_hour, end_hour)
    plt.grid(True)
    plt.legend()

    output_dir = "results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, 'nop_status_timeseries.png')
    plt.savefig(output_filename)
    plt.close()
    print(f"Success! NOP status image has been saved to: {os.path.abspath(output_filename)}")

def plot_line_flow_snapshots_comparison(all_ts_data, seed, gui_params):
    """
    Generate a phased, multi-Algorithm line power flow comparison diagram for each time step.
    - A picture at each moment, including two sub-pictures, representing the first and second stages respectively.
    - In each subgraph, use grouped histograms to compare the line flow of each Algorithm.
    """
    print("--- Generating phased, multi-Algorithm line power flow comparison snapshots ---")

    # 1. Preparation
    try:
        num_steps = len(next(iter(all_ts_data.values()))['step_costs'])
        algorithms = list(all_ts_data.keys())
        if num_steps == 0: raise IndexError
    except (StopIteration, KeyError, IndexError):
        print("Warning: Unable to determine the valid number of steps, skip drawing the power flow comparison snapshot graph.")
        return

    # 2. Create a separate output folder
    output_dir = os.path.join("results_outputs", f"line_flow_snapshots_comparison_seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    # 3. Traverse each time step
    for t in range(num_steps):
        fig, axes = plt.subplots(2, 1, figsize=(20, 16), sharex=True)
        fig.suptitle(f' Line active power flow comparison (time step: {t}, scene Seed: {seed})', fontsize=20)

        # --- Data collection ---
        stage1_data, stage2_data = {}, {}
        all_line_ids = set()

        for algo_name, ts_data in all_ts_data.items():
            # Extract the first stage data
            s1_flows = ts_data.get('line_powers_data_stage1', {})
            if s1_flows:
                all_line_ids.update(s1_flows.keys())
                stage1_data[algo_name] = {line: p_list[t] for line, p_list in s1_flows.items() if t < len(p_list)}

            # Extract second stage data
            s2_flows = ts_data.get('line_powers_data_stage2', {})
            if s2_flows:
                all_line_ids.update(s2_flows.keys())
                stage2_data[algo_name] = {line: p_list[t] for line, p_list in s2_flows.items() if t < len(p_list)}

        if not all_line_ids:
            plt.close(fig)
            continue

        sorted_line_ids = sorted(list(all_line_ids), key=lambda l: int(''.join(filter(str.isdigit, l)) or 0))

        # --- Drawing parameters ---
        num_algorithms = len(algorithms)
        bar_width = 0.8 / num_algorithms
        x_indices = np.arange(len(sorted_line_ids))

        # --- Draw subgraph ---
        plot_titles = ['Phase 1 (Linear DistFlow Approximation)', 'Phase 2 (OpenDSS Precise Calculation)']
        data_sources = [stage1_data, stage2_data]

        for i, ax in enumerate(axes):
            for j, algo_name in enumerate(algorithms):
                source = data_sources[i]
                flows = [source.get(algo_name, {}).get(line_id, 0) for line_id in sorted_line_ids]
                offset = j * bar_width - (0.8 - bar_width) / 2
                ax.bar(x_indices + offset, flows, bar_width, label=algo_name)

            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_title(plot_titles[i], fontsize=16)
            ax.set_ylabel('Active Power Flow P (pu)', fontsize=14)
            ax.legend()
            ax.grid(True, axis='y', linestyle=':')

        plt.xticks(x_indices, sorted_line_ids, rotation=90)
        plt.xlabel('branch ID', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # --- Save image ---
        save_path = os.path.join(output_dir, f'line_flow_comparison_t{t}.png')
        plt.savefig(save_path)
        plt.close(fig)

    print(f"✅ The phased power flow comparison snapshot has been saved to the folder: {os.path.abspath(output_dir)}")
