import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from stable_baselines3.common.callbacks import BaseCallback
import re
# Import relevant modules in your project
from grid_model import create_grid
from evaluate_agents import evaluate_baseline
from gev_station import average_hour_multiplier
from rl_normalization import normalize_observation

# Set up matplotlib to display Chinese and negative signs correctly
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# Callback class used to collect data during training

class CostCurveCallback(BaseCallback):
    """
    (data collection mode)
    A custom callback function whose main responsibilities are:
    1. At the beginning of training, calculate and save the "Pure Operating Cost" and "Total Objective Value" of the Baseline respectively.
    2. During the training process, periodically evaluate the current RL model and record its Total Cost and total reward.
    3. At the end of training or when it is interrupted, save all collected data to a unique file.
    """

    def __init__(self, eval_env, agent_name: str, eval_freq: int = 5000, save_path: str = "./logs/", seed: int = 0):
        super(CostCurveCallback, self).__init__(verbose=1)
        self.eval_env = eval_env
        self.agent_name = agent_name
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.seed = seed

        # List used to store data
        self.training_steps = []
        self.rl_costs = []
        self.rl_rewards = [] # Scaled rewards
        self.rl_rewards_unscaled = [] # Unscaled rewards

        # Define two independent Baseline benchmark file paths
        self.baseline_cost_file = os.path.join(self.save_path, "baseline_cost.pkl")
        self.baseline_objective_file = os.path.join(self.save_path, "baseline_objective.pkl")

        # Define the data file path of the RL agent
        self.rl_data_file = os.path.join(self.save_path, f"{self.agent_name}_data.pkl")

    def _on_training_start(self) -> None:
        """At the beginning of training, calculate and save the pure cost and Total Objective Value of the Baseline respectively"""
        os.makedirs(self.save_path, exist_ok=True)

        # As long as any baseline file does not exist, re-perform a complete Baseline evaluation
        if not os.path.exists(self.baseline_cost_file) or not os.path.exists(self.baseline_objective_file):
            print("--- CostCurveCallback: Baseline benchmark file not found, recalculating... ---")

            # Create a new, clean power grid instance for Baseline evaluation
            grid_for_baseline = create_grid(
                model=self.eval_env.params['grid_model'],
                gui_params=self.eval_env.params
            )

            # Generate charging scene using the same parameters as the RL environment
            stations_list_for_baseline = self.eval_env.stations_list
            ev_hour_multipliers = (self.eval_env.params.get("time_profiles") or {}).get("ev_multiplier_by_hour")
            ev_profile_average = average_hour_multiplier(ev_hour_multipliers, 1.0)
            global_ev_multiplier = self.eval_env.params.get("global_ev_multiplier", 1.0)
            effective_ev_multiplier = ev_profile_average if ev_hour_multipliers else float(global_ev_multiplier)
            for i, station in enumerate(stations_list_for_baseline):
                base_num_evs = self.eval_env.stations_info[i]['Num_EVs_to_Generate']
                num_evs = int(base_num_evs * effective_ev_multiplier)
                station.generate_daily_scenarios(
                    num_evs_to_generate=num_evs,
                    arrival_hour_multipliers=ev_hour_multipliers,
                )

            # Call the evaluation function
            metrics, _ = evaluate_baseline(
                gui_params=self.eval_env.params,
                seed=self.seed,
                stations_list=stations_list_for_baseline,
                grid=grid_for_baseline,
                use_two_stage=self.eval_env.use_two_stage_flow
            )

            # Extract Total Objective Value (including penalty items)
            baseline_objective_value = metrics.get("Total Objective Value")
            if baseline_objective_value is not None:
                with open(self.baseline_objective_file, 'wb') as f:
                    pickle.dump(baseline_objective_value, f)
                print(f"--- Baseline【Total Objective Value】saved: {baseline_objective_value:.2f} ---")

            # Calculate Pure Operating Cost (excluding penalties)
            baseline_operational_cost = (
                    metrics.get("Grid Purchase Cost", 0) +
                    metrics.get("Generation Cost", 0) +
                    metrics.get("SOP Loss Cost", 0) +
                    metrics.get("ESS Discharge Cost", 0)
            )
            with open(self.baseline_cost_file, 'wb') as f:
                pickle.dump(baseline_operational_cost, f)
            print(f"--- Baseline【Pure Operating Cost】saved: {baseline_operational_cost:.2f} ---")

        else:
            print("--- CostCurveCallback: An existing Baseline benchmark file has been found. ---")

    def _on_step(self) -> bool:
        """Periodically evaluate and record the costs and rewards of the RL model"""
        if self.n_calls > 0 and self.n_calls % self.eval_freq == 0:
            print(f"\n--- CostCurveCallback ({self.agent_name}): Evaluating Training Steps {self.num_timesteps}... ---")

            obs, _ = self.eval_env.reset(seed=self.seed)
            terminated = False
            truncated = False

            episode_total_cost = 0
            episode_total_reward = 0
            episode_total_reward_unscaled = 0

            while not (terminated or truncated):
                action_obs = normalize_observation(obs, self.model.get_vec_normalize_env())
                action, _ = self.model.predict(action_obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)

                step_cost = (
                        info.get('grid_purchase_cost', 0)
                        + info.get('generation_cost', 0)
                        + info.get('sop_loss_cost', 0)
                        + info.get('ess_discharge_cost', 0)
                )
                episode_total_cost += step_cost

                # Scaled reward (environment return)
                episode_total_reward += reward

                # Unscaled reward, taken from info (just added in step())
                episode_total_reward_unscaled += info.get('reward_unscaled', 0.0)

            self.training_steps.append(self.num_timesteps)
            self.rl_costs.append(episode_total_cost)
            self.rl_rewards.append(episode_total_reward)
            self.rl_rewards_unscaled.append(episode_total_reward_unscaled)

            print(f"--- Evaluation completed. Current cost: {episode_total_cost:.2f} yuan | Current reward: {episode_total_reward:.2f} ---")
        return True

    def save_data(self):
        """A method that can be called externally and is specifically used to save data"""
        if not self.training_steps:
            print(f"--- CostCurveCallback ({self.agent_name}): Warning - No evaluation data collected, no data file created. ---")
            print(f"--- (This is usually because the Training Steps are too few and the evaluation frequency has not been reached eval_freq={self.eval_freq}) ---")
            return

        print(f"--- CostCurveCallback ({self.agent_name}): Saving cost and reward data... ---")
        data_to_save = {
            'steps': self.training_steps,
            'costs': self.rl_costs,
            'rewards': self.rl_rewards, # After scaling
            'rewards_unscaled': self.rl_rewards_unscaled # Unscaled
        }
        with open(self.rl_data_file, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"--- {self.agent_name}'s data has been successfully saved to: {os.path.abspath(self.rl_data_file)} ---")

    def _on_training_end(self) -> None:
        """When training ends normally, the method to save data is automatically called"""
        self.save_data()

def plot_curves():
    """
    (drawing mode)
    Load all saved cost/reward data and plot:
    1) Cost/objective function comparison chart of each Algorithm item;
    2) Mean ± standard deviation shadow plot under multiple random seeds (if seed_* structure is detected).
    """
    print("\n" + "=" * 50 + "\n--- Start the final training curve drawing program (including multiple random seed statistics) ---")

    import glob
    import os
    from pathlib import Path
    from config import PATHS

    # ========== Unified "File Finder" ==========
    # Recursively search for files under PATHS["models_dir"] and the project root directory/runs
    def _find_all(filename: str):
        bases = []
        # 1) models directory
        if "models_dir" in PATHS and os.path.isdir(PATHS["models_dir"]):
            bases.append(PATHS["models_dir"])
        # 2) runs directory (overnight script product archive)
        runs_dir = Path(__file__).resolve().parent / "runs"
        if runs_dir.is_dir():
            bases.append(str(runs_dir))

        cands = []
        for base in bases:
            cands.extend(glob.glob(os.path.join(base, "**", filename), recursive=True))
        return cands

    def _find_latest(filename: str) -> str:
        cands = _find_all(filename)
        if not cands:
            return ""
        return max(cands, key=os.path.getmtime)

    # ========== Find single/multiple data.pkl by agent name ==========
    def _find_agent_pkl(agent_name: str) -> str:
        """
        First look for the latest data.pkl of the agent in the runs directory (used for multiple random seeds);
        If there is no value in runs, it will fall back to the latest global value.
        """
        filename = f"{agent_name}_data.pkl"
        all_cands = _find_all(filename)
        if not all_cands:
            return ""
        runs_cands = [p for p in all_cands if os.path.sep + "runs" + os.path.sep in p]
        if runs_cands:
            return max(runs_cands, key=os.path.getmtime)
        return max(all_cands, key=os.path.getmtime)

    def _find_all_agent_pkls(agent_name: str):
        """
        Returns the latest batch of all <agent_name>_data.pkl of the agent under runs/<run_id>/...,
        is used for multi-random seed aggregation.
        If the seed_* structure cannot be found, it will degenerate to only returning the latest file.
        """
        first = _find_agent_pkl(agent_name)
        if not first:
            return []

        p = Path(first).resolve()

        # Try to find the seed_* directory (structure: runs/<run_id>/seed_x/<Algo>/artifacts/...)
        run_dir = None
        for parent in p.parents:
            if parent.name.startswith("seed_"):
                run_dir = parent.parent # The upper level of seed_x is the run_id directory
                break

        if run_dir is None or not run_dir.exists():
            # No seed_* structure, only a single file is used by default
            return [first]

        pattern = str(run_dir / "**" / f"{agent_name}_data.pkl")
        files = glob.glob(pattern, recursive=True)
        files = sorted(set(files))
        return files or [first]

    # The agent to be drawn (consistent with the agent_name of CostCurveCallback during your training)
    agent_names = ["SAC_Two_Stage", "DDPG_Two_Stage", "TD3_Two_Stage", "PPO_Two_Stage"]
    agents_to_plot = {name: _find_agent_pkl(name) for name in agent_names}

    # baseline file (whoever updates it first is fine, take the latest time)
    baseline_cost_file = _find_latest("baseline_cost.pkl")
    baseline_objective_file = _find_latest("baseline_objective.pkl")

    # 1. Load Baseline data
    try:
        with open(baseline_cost_file, 'rb') as f:
            baseline_cost = pickle.load(f)
        print(f"Successfully loaded BaselinePure Operating Cost: {baseline_cost:.2f} @ {baseline_cost_file}")
    except Exception:
        print(f"Warning: Baseline cost file not found: {baseline_cost_file}")
        baseline_cost = None

    try:
        with open(baseline_objective_file, 'rb') as f:
            baseline_objective = pickle.load(f)
        print(f"Successfully loaded BaselineTotal Objective Value: {baseline_objective:.2f} @ {baseline_objective_file}")
    except Exception:
        print(f"Warning: Baseline target value file not found: {baseline_objective_file}")
        baseline_objective = None

    # 2. Load the "single curve" data of each Algorithm (the latest one)
    agent_data = {}
    for agent_name, data_file in agents_to_plot.items():
        if not data_file:
            print(f"Warning: Data file not found for {agent_name}, the model will be skipped.")
            continue
        try:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            # Compatible with old versions: when there is no unscaled, use scaled on top.
            if 'rewards_unscaled' not in data and 'rewards' in data:
                data['rewards_unscaled'] = data['rewards']
            agent_data[agent_name] = data
            print(f"Successfully loaded data of {agent_name}: {data_file}")
        except Exception as e:
            print(f"Warning: Failed to open data file for {agent_name} ({data_file}): {e}")

    if not agent_data:
        print("⚠️ No RL data was loaded. Possible reasons: 1) Training Steps did not reach eval_freq and *_data.pkl was not generated;"
              " 2) Data is still in runs/<run_id>/<Algo>/artifacts, but not retrieved; 3) Permission/path error.")

    multi_seed_data = {}

    for agent_name in agent_names:
        all_pkls = _find_all_agent_pkls(agent_name)
        # If there is only 1 file, there is no need to draw shadow; skip
        unique_pkls = sorted(set(all_pkls))
        if len(unique_pkls) <= 1:
            continue

        runs = []
        for pkl_path in unique_pkls:
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                if 'rewards_unscaled' not in data and 'rewards' in data:
                    data['rewards_unscaled'] = data['rewards']
                runs.append(data)
                print(f"[Multiple seeds] {agent_name}: Loaded {pkl_path}")
            except Exception as e:
                print(f"[Multiple seeds] {agent_name}: Loading {pkl_path} failed: {e}")

        if len(runs) < 2:
            # Some loading failed, and there was only 1 item in the end, so statistics will not be done.
            continue

        # Based on the shortest length, assuming eval_freq is the same
        lengths = [len(r['steps']) for r in runs if len(r.get('steps', [])) > 0]
        if not lengths:
            continue
        min_len = min(lengths)

        steps = np.array(runs[0]['steps'][:min_len], dtype=float)
        cost_stack = np.stack([np.array(r['costs'][:min_len], dtype=float) for r in runs], axis=0)
        neg_reward_stack = np.stack(
            [
                -np.array(r.get('rewards_unscaled', r['rewards'])[:min_len], dtype=float)
                for r in runs
            ],
            axis=0
        )

        multi_seed_data[agent_name] = {
            'steps': steps,
            'cost_mean': cost_stack.mean(axis=0),
            'cost_std': cost_stack.std(axis=0),
            'obj_mean': neg_reward_stack.mean(axis=0),
            'obj_std': neg_reward_stack.std(axis=0),
            'num_runs': len(runs),
        }

    save_dir = "./results_outputs"
    os.makedirs(save_dir, exist_ok=True)

    # ========== Figure 1: [Pure Operating Cost] comparison of a single curve ==========
    fig1, ax1 = plt.subplots(figsize=(14, 9))
    if baseline_cost is not None:
        ax1.axhline(
            y=baseline_cost,
            color='r',
            linestyle='--',
            label=f"Baseline Pure Operating Cost ({baseline_cost:.2f})"
        )

    all_costs = [baseline_cost] if baseline_cost is not None else []
    for name, data in agent_data.items():
        ax1.plot(
            data['steps'],
            data['costs'],
            marker='o',
            linestyle='-',
            label=f"{name} Total Cost (single item)"
        )
        all_costs.extend(data['costs'])

    ax1.set_title("RL Agent training process [Pure Operating Cost] convergence comparison chart (single curve)", fontsize=18)
    ax1.set_xlabel("Training Steps", fontsize=14)
    ax1.set_ylabel("Evaluation round Total Cost (yuan)", fontsize=14)
    if agent_data:
        ax1.legend(fontsize=12)
    ax1.grid(True, linestyle=':')
    if all_costs:
        min_val, max_val = np.nanmin(all_costs), np.nanmax(all_costs)
        padding = (max_val - min_val) * 0.1 if max_val > min_val else 10.0
        ax1.set_ylim(min_val - padding, max_val + padding)

    cost_plot_path = os.path.join(save_dir, "final_training_cost_comparison_single.png")
    fig1.savefig(cost_plot_path)
    print(f"✅ Pure cost comparison chart (single line) has been saved to: {os.path.abspath(cost_plot_path)}")
    plt.close(fig1)

    # ========== Figure 2: [-Total Reward] of a single curve vs Baseline Total Objective Value ==========
    fig2, ax2 = plt.subplots(figsize=(14, 9))
    if baseline_objective is not None:
        ax2.axhline(
            y=baseline_objective,
            color='r',
            linestyle='--',
            label=f"Baseline Total Objective Value ({baseline_objective:.2f})"
        )

    all_obj_vals = [baseline_objective] if baseline_objective is not None else []
    for name, data in agent_data.items():
        negative_rewards_unscaled = -np.array(data['rewards_unscaled'])
        ax2.plot(
            data['steps'],
            negative_rewards_unscaled,
            marker='x',
            linestyle='-',
            label=f"{name} (-total reward_unscaled, single)"
        )
        all_obj_vals.extend(negative_rewards_unscaled)

    ax2.set_title("RL Agent (-Total Reward) and Baseline (Total Objective Value) comparison chart (single curve)", fontsize=18)
    ax2.set_xlabel("Training Steps", fontsize=14)
    ax2.set_ylabel("Objective function value (the lower the better)", fontsize=14)
    if agent_data:
        ax2.legend(fontsize=12)
    ax2.grid(True, linestyle=':')
    if all_obj_vals:
        min_val, max_val = np.nanmin(all_obj_vals), np.nanmax(all_obj_vals)
        padding = (max_val - min_val) * 0.1 if max_val > min_val else 10.0
        ax2.set_ylim(min_val - padding, max_val + padding)

    reward_plot_path = os.path.join(save_dir, "final_training_objective_comparison_single.png")
    fig2.savefig(reward_plot_path)
    print(f"✅ Comprehensive target comparison chart (single) has been saved to: {os.path.abspath(reward_plot_path)}")
    plt.close(fig2)

    # ========== Figure 3: Multiple random seeds [Pure Operating Cost] mean ± standard deviation ==========
    if multi_seed_data:
        fig3, ax3 = plt.subplots(figsize=(14, 9))
        if baseline_cost is not None:
            ax3.axhline(
                y=baseline_cost,
                color='r',
                linestyle='--',
                label=f"Baseline Pure Operating Cost ({baseline_cost:.2f})"
            )

        all_costs_ms = [baseline_cost] if baseline_cost is not None else []
        for name, agg in multi_seed_data.items():
            steps = agg['steps']
            mean = agg['cost_mean']
            std = agg['cost_std']
            num_runs = agg['num_runs']

            (line,) = ax3.plot(
                steps,
                mean,
                linestyle='-',
                label=f"{name} Total Cost Mean (N={num_runs})"
            )
            ax3.fill_between(
                steps,
                mean - std,
                mean + std,
                alpha=0.2,
                color=line.get_color()
            )
            all_costs_ms.extend(mean - std)
            all_costs_ms.extend(mean + std)

        ax3.set_title("RL Agent【Pure Operating Cost】Multiple random seeds mean ± standard deviation", fontsize=18)
        ax3.set_xlabel("Training Steps", fontsize=14)
        ax3.set_ylabel("Evaluation round Total Cost (yuan)", fontsize=14)
        ax3.legend(fontsize=12)
        ax3.grid(True, linestyle=':')
        if all_costs_ms:
            min_val, max_val = np.nanmin(all_costs_ms), np.nanmax(all_costs_ms)
            padding = (max_val - min_val) * 0.1 if max_val > min_val else 10.0
            ax3.set_ylim(min_val - padding, max_val + padding)

        cost_ms_path = os.path.join(save_dir, "final_training_cost_comparison_multiseed.png")
        fig3.savefig(cost_ms_path)
        print(f"✅ Pure cost comparison chart (multiple random seeds) has been saved to: {os.path.abspath(cost_ms_path)}")
        plt.close(fig3)

        # ========== Figure 4: Multiple random seeds [-total reward] mean ± standard deviation ==========
        fig4, ax4 = plt.subplots(figsize=(14, 9))
        if baseline_objective is not None:
            ax4.axhline(
                y=baseline_objective,
                color='r',
                linestyle='--',
                label=f"Baseline Total Objective Value ({baseline_objective:.2f})"
            )

        all_obj_ms = [baseline_objective] if baseline_objective is not None else []
        for name, agg in multi_seed_data.items():
            steps = agg['steps']
            mean = agg['obj_mean']
            std = agg['obj_std']
            num_runs = agg['num_runs']

            (line,) = ax4.plot(
                steps,
                mean,
                linestyle='-',
                label=f"{name} (-total reward) mean (N={num_runs})"
            )
            ax4.fill_between(
                steps,
                mean - std,
                mean + std,
                alpha=0.2,
                color=line.get_color()
            )
            all_obj_ms.extend(mean - std)
            all_obj_ms.extend(mean + std)

        ax4.set_title("RL Agent (-Total Reward) Multiple Random Seed Mean ± Standard Deviation vs Baseline", fontsize=18)
        ax4.set_xlabel("Training Steps", fontsize=14)
        ax4.set_ylabel("Objective function value (the lower the better)", fontsize=14)
        ax4.legend(fontsize=12)
        ax4.grid(True, linestyle=':')
        if all_obj_ms:
            min_val, max_val = np.nanmin(all_obj_ms), np.nanmax(all_obj_ms)
            padding = (max_val - min_val) * 0.1 if max_val > min_val else 10.0
            ax4.set_ylim(min_val - padding, max_val + padding)

        obj_ms_path = os.path.join(save_dir, "final_training_objective_comparison_multiseed.png")
        fig4.savefig(obj_ms_path)
        print(f"✅ Comprehensive target comparison chart (multiple random seeds) has been saved to: {os.path.abspath(obj_ms_path)}")
        plt.close(fig4)
    else:
        print("ℹ️ More than 1 random seed data file was not detected, multiple random seed shadow maps will not be generated.")



if __name__ == '__main__':
    plot_curves()
