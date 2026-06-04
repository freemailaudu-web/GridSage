# file: simulation_runner.py
import sys
import time
import io
import traceback
import os
import json
import glob
import importlib.util
from PySide6.QtCore import QObject, Signal, Slot
import pandas as pd

# Import stable_baselines3 and environment
from stable_baselines3 import PPO, DDPG, SAC, TD3
from stable_baselines3 import PPO, DDPG, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from power_grid_env import PowerGridEnv

# Import core functions in the project
from config import PATHS, CORE_PARAMS, TRAINING_CONFIG, RL_ENV_CONFIG, get_effective_rl_hyperparams
from evaluate_agents import evaluate_baseline, evaluate_rl_agent, plot_and_save_results, \
    plot_accumulated_costs, plot_voltage_snapshots, plot_line_flow_snapshots_comparison, \
    plot_aggregated_ev_power, plot_sop_flows, plot_nop_status
from copy import deepcopy
from config import EVALUATION_CONFIG
from rl_normalization import load_eval_vecnormalize, save_vecnormalize_stats, wrap_training_env

def build_model_kwargs(model_class, rl_algo_name: str, env, gui_params: dict) -> tuple[str, dict]:
    """Construct SB3 model initialization parameters based on GUI settings."""
    algo_key = str(rl_algo_name).upper()
    try:
        if issubclass(model_class, PPO):
            algo_key = "PPO"
        elif issubclass(model_class, SAC):
            algo_key = "SAC"
        elif issubclass(model_class, DDPG):
            algo_key = "DDPG"
        elif issubclass(model_class, TD3):
            algo_key = "TD3"
    except TypeError:
        pass

    merged = get_effective_rl_hyperparams(algo_key, gui_settings=gui_params)
    common = merged.get("common", {})
    specific = merged.get("specific", {})

    kwargs = {
        "verbose": 1,
        "tensorboard_log": PATHS["tensorboard_logs"],
    }

    if algo_key == "PPO":
        kwargs.update({
            "learning_rate": float(common.get("learning_rate", 3e-4)),
            "gamma": float(common.get("gamma", 0.99)),
            "batch_size": int(common.get("batch_size", 256)),
            "clip_range": float(specific.get("clip_range", 0.2)),
            "ent_coef": float(specific.get("ent_coef", 0.0)),
        })
    elif algo_key == "SAC":
        kwargs.update({
            "learning_rate": float(common.get("learning_rate", 3e-4)),
            "gamma": float(common.get("gamma", 0.99)),
            "batch_size": int(common.get("batch_size", 256)),
            "tau": float(specific.get("tau", 0.005)),
            "ent_coef": float(specific.get("ent_coef", 0.1)),
        })
    elif algo_key == "DDPG":
        action_dim = env.action_space.shape[0]
        noise_sigma = float(specific.get("action_noise", 0.1))
        kwargs.update({
            "learning_rate": float(common.get("learning_rate", 1e-3)),
            "gamma": float(common.get("gamma", 0.99)),
            "batch_size": int(common.get("batch_size", 256)),
            "tau": float(specific.get("tau", 0.005)),
            "action_noise": NormalActionNoise(
                mean=np.zeros(action_dim, dtype=np.float32),
                sigma=noise_sigma * np.ones(action_dim, dtype=np.float32),
            ),
        })
    elif algo_key == "TD3":
        kwargs.update({
            "learning_rate": float(common.get("learning_rate", 3e-4)),
            "gamma": float(common.get("gamma", 0.99)),
            "batch_size": int(common.get("batch_size", 256)),
            "policy_delay": int(specific.get("policy_delay", 2)),
            "target_policy_noise": float(specific.get("target_policy_noise", 0.2)),
        })

    return algo_key, kwargs
def discover_rl_algorithms_util():
    """Independent, safe utility function for discovering all available RLAlgorithm classes."""
    model_class_registry = {}
    model_class_registry.update({"PPO": PPO, "DDPG": DDPG, "SAC": SAC, "TD3": TD3})
    plugin_dir = "custom_algorithms"
    if os.path.isdir(plugin_dir):
        plugin_files = glob.glob(os.path.join(plugin_dir, "*.py"))
        for plugin_file in plugin_files:
            module_name = os.path.basename(plugin_file)[:-3]
            try:
                spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, 'register_algorithm'):
                    info = module.register_algorithm()
                    model_class_registry[info['name']] = info['class']
            except Exception as e:
                print(f"Background warning: Loading custom Algorithm plug-in {plugin_file} failed: {e}")
    return model_class_registry


class GuiCallback(BaseCallback):
    def __init__(self, worker, total_timesteps, verbose=0):
        super(GuiCallback, self).__init__(verbose)
        self.worker = worker
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        if self.worker.is_stopped:
            return False
        if self.n_calls % 100 == 0:
            self.worker.progress_update.emit(self.num_timesteps, self.total_timesteps)
        return True


class Stream(QObject):
    new_text = Signal(str)

    def write(self, text):
        self.new_text.emit(str(text))

    def flush(self):
        pass


class SimulationWorker(QObject):
    finished = Signal()
    progress = Signal(str)
    error = Signal(str)
    progress_update = Signal(int, int)

    def __init__(self, task_type, task_params):
        super().__init__()
        self.task_type = task_type
        self.task_params = task_params
        self.is_stopped = False

    @Slot()
    def request_stop(self):
        self.progress.emit("...received termination signal, will stop after the current step is completed...")
        self.is_stopped = True

    @Slot()
    def run(self):
        """
        This function is the entry point of the background thread. All time-consuming operations must be performed here.
        """
        start_time = time.time()
        # Redirecting the output stream must be done inside the thread
        sys.stdout = Stream(new_text=self.progress.emit)
        sys.stderr = Stream(new_text=self.error.emit)

        try:
            # All time-consuming operations are inside the run() function, ensuring that they are executed in the background thread
            self.progress.emit("Background task started, preparing environment...")

            with open(PATHS["gui_settings"], 'r') as f:
                self.gui_params = json.load(f)
            self.update_core_configs()

            if self.task_type == "run_baseline":
                self.run_baseline_task()
            elif self.task_type == "train_rl":
                self.run_training_task()
            elif self.task_type == "evaluate":
                self.run_evaluation_task()
            else:
                raise ValueError(f"Unknown task type: {self.task_type}")

        except Exception as e:
            self.error.emit(f"Task execution error: {e}\n{traceback.format_exc()}")
        finally:
            # Restore standard output
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            self.progress.emit(f"\nTask ended, total time taken: {time.time() - start_time:.2f} seconds.")
            self.finished.emit()

    def update_core_configs(self):
        CORE_PARAMS['grid_model'] = self.gui_params['grid_model']
        CORE_PARAMS['solver'] = self.gui_params['solver']
        CORE_PARAMS['start_hour'] = self.gui_params['start_hour']
        CORE_PARAMS['end_hour'] = self.gui_params['end_hour']
        CORE_PARAMS['step_minutes'] = self.gui_params['step_minutes']
        CORE_PARAMS['distributed_energy']['pv'] = self.gui_params['use_pv']
        CORE_PARAMS['distributed_energy']['wind'] = self.gui_params['use_wind']
        CORE_PARAMS['distributed_energy']['ess'] = self.gui_params['use_ess']
        CORE_PARAMS['sop_nodes_active'] = self.gui_params['use_sop']
        CORE_PARAMS['nop_nodes_active'] = self.gui_params['use_nop']
        CORE_PARAMS['reconfiguration_mode'] = self.gui_params.get('reconfiguration_mode', CORE_PARAMS.get('reconfiguration_mode', 'radial_reconfiguration'))
        CORE_PARAMS['selected_reconfiguration_plan_id'] = self.gui_params.get('selected_reconfiguration_plan_id', CORE_PARAMS.get('selected_reconfiguration_plan_id', 'R0'))
        CORE_PARAMS['available_reconfiguration_plans'] = self.gui_params.get('available_reconfiguration_plans', CORE_PARAMS.get('available_reconfiguration_plans', []))
        CORE_PARAMS['reconfiguration_constraints'] = self.gui_params.get('reconfiguration_constraints', CORE_PARAMS.get('reconfiguration_constraints', {}))

        # Pass EV data source settings to core parameters
        # We get this new setting from self.gui_params
        # Defaults to 'random' if the setting does not exist (e.g. using the old gui_settings.json)
        CORE_PARAMS['ev_data_source'] = self.gui_params.get('ev_data_source', 'random')

        # EV physical parameters
        CORE_PARAMS['ev_params'] = self.gui_params.get(
            'ev_params',
            CORE_PARAMS.get('ev_params', {})
        )

        # Reward mode
        CORE_PARAMS['reward_mode'] = self.gui_params.get('reward_mode', 'grid_operator')

        # Operator parameters
        CORE_PARAMS['station_operator'] = self.gui_params.get(
            'station_operator',
            CORE_PARAMS.get('station_operator', {})
        )

        # Reward weight
        gui_rw = self.gui_params.get('reward_weights', {})
        CORE_PARAMS['reward_weights'] = gui_rw

        RL_ENV_CONFIG['reward_weights']['ev_kwh_shortage_penalty'] = gui_rw.get(
            'ev_kwh_shortage_penalty',
            RL_ENV_CONFIG['reward_weights']['ev_kwh_shortage_penalty']
        )
        RL_ENV_CONFIG['reward_weights']['voltage_violation_penalty'] = gui_rw.get(
            'voltage_violation_penalty',
            RL_ENV_CONFIG['reward_weights']['voltage_violation_penalty']
        )
        RL_ENV_CONFIG['reward_weights']['cost_penalty_factor'] = gui_rw.get(
            'cost_penalty_factor',
            RL_ENV_CONFIG['reward_weights']['cost_penalty_factor']
        )
        RL_ENV_CONFIG['penalties']['opendss_failure_penalty'] = gui_rw.get(
            'opendss_failure_penalty',
            RL_ENV_CONFIG['penalties']['opendss_failure_penalty']
        )

        self.progress.emit("Core configuration has been updated according to GUI settings.")

    def run_training_task(self):
        self.progress.emit("\n--- Start training the reinforcement learning model ---")

        rl_algo_name = self.task_params['rl_algo_name']
        use_two_stage = self.task_params['mode'] == 'two_stage'

        available_algos = discover_rl_algorithms_util()
        model_class = available_algos.get(rl_algo_name)
        if not model_class:
            raise ValueError(f"The Algorithm class named {rl_algo_name} cannot be found.")

        self.progress.emit(f"Select Algorithm: {rl_algo_name}")
        self.progress.emit(f"Running mode: {'two-stage' if use_two_stage else 'single-stage'}")

        # Environment initialization is a time-consuming operation and must be performed in a background thread
        self.progress.emit("Initializing the simulation environment (this may take some time)...")
        env_raw = PowerGridEnv(gui_params=CORE_PARAMS, use_two_stage_flow=use_two_stage)
        rl_hparams = get_effective_rl_hyperparams(rl_algo_name, gui_settings=getattr(self, "gui_params", {}))
        gamma = float(rl_hparams.get("common", {}).get("gamma", 0.99))
        env = wrap_training_env(env_raw, gamma=gamma, training_config=TRAINING_CONFIG)
        self.progress.emit("Environment initialization completed.")

        from training_visualizer import CostCurveCallback

        model = model_class('MlpPolicy', env, verbose=1, tensorboard_log=PATHS["tensorboard_logs"])

        total_timesteps = TRAINING_CONFIG["total_timesteps"]
        gui_callback = GuiCallback(self, total_timesteps)

        # The evaluation environment must be consistent with the training configuration, create a separate one
        eval_env = PowerGridEnv(gui_params=CORE_PARAMS, use_two_stage_flow=use_two_stage)

        # Unify naming to ensure that training_visualizer can find the corresponding directories and files
        mode_suffix = "two_stage" if use_two_stage else "single_stage"
        model_dir_name = f"best_{rl_algo_name.lower()}_{mode_suffix}_{total_timesteps}steps"
        save_path = os.path.join(PATHS["models_dir"], model_dir_name)
        os.makedirs(save_path, exist_ok=True)

        agent_name = f"{rl_algo_name.upper()}_{'Two_Stage' if use_two_stage else 'Single_Stage'}"
        cost_callback = CostCurveCallback(
            eval_env=eval_env,
            agent_name=agent_name,
            save_path=save_path,
            eval_freq=int(TRAINING_CONFIG.get("cost_curve_eval_freq", 500))
        )

        #Hang progress callback and collection callback at the same time
        callbacks = [gui_callback, cost_callback]

        self.progress.emit(f"Total Training Steps: {total_timesteps}")
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)

        # Explicitly save the collected data after training (to prevent early interruption without disk placement)
        try:
            cost_callback.save_data()
        except Exception as _:
            pass

        # Save model
        model.save(os.path.join(save_path, "best_model.zip"))
        try:
            save_vecnormalize_stats(env, save_path)
        except Exception as norm_err:
            self.progress.emit(f"[RL-Normalization] Failed to save VecNormalize stats: {norm_err}")
        self.progress.emit(f"The model has been saved to: {save_path}")

        if self.is_stopped:
            self.progress.emit("\nTraining was manually terminated by the user.")
        else:
            self.progress.emit("\nTraining completed!")
            mode_suffix = "two_stage" if use_two_stage else "single_stage"
            model_save_name = f"best_{rl_algo_name.lower()}_{mode_suffix}_{total_timesteps}steps"
            save_path = os.path.join(PATHS["models_dir"], model_save_name)
            os.makedirs(save_path, exist_ok=True)
            model.save(os.path.join(save_path, "best_model.zip"))
            try:
                save_vecnormalize_stats(env, save_path)
            except Exception as norm_err:
                self.progress.emit(f"[RL-Normalization] Failed to save VecNormalize stats: {norm_err}")
            self.progress.emit(f"The model has been saved to: {save_path}")

    def run_baseline_task(self):
        self.progress.emit("\n--- Start running Baseline (based on solver) ---")
        use_two_stage = self.task_params['mode'] == 'two_stage'
        self.progress.emit("Initializing the simulation environment (this may take some time)...")
        env_for_scene = PowerGridEnv(gui_params=CORE_PARAMS, use_two_stage_flow=use_two_stage)
        self.progress.emit("Environment initialization completed.")
        env_for_scene.reset(seed=0)
        grid_instance = deepcopy(env_for_scene.grid)
        stations_list = env_for_scene.stations_list
        metrics, time_series_data = evaluate_baseline(
            CORE_PARAMS, 0, stations_list, grid_instance, use_two_stage
        )
        if not metrics:
            raise Exception("Baseline solution failed, please check the log output.")
        self.progress.emit("\n--- Baseline running result ---")
        for key, val in metrics.items():
            if isinstance(val, float):
                self.progress.emit(f"  - {key}: {val:.4f}")
            else:
                self.progress.emit(f"  - {key}: {val}")

    def run_evaluation_task(self):
        self.progress.emit("\n--- Start evaluation and comparison task ---")
        use_two_stage = self.task_params['mode'] == 'two_stage'
        selected_algos = self.task_params['selected_algos']
        num_episodes = EVALUATION_CONFIG.get("num_test_episodes", 10)
        self.progress.emit(f"Evaluation mode: {'two-stage' if use_two_stage else 'single-stage'}")
        self.progress.emit(f"Compare Algorithm: {', '.join(selected_algos)}")
        model_name_to_class = discover_rl_algorithms_util()
        all_results_metrics = []
        self.progress.emit("Initializing the simulation environment (this may take some time)...")
        env = PowerGridEnv(gui_params=CORE_PARAMS, use_two_stage_flow=use_two_stage)
        self.progress.emit("Environment initialization completed.")
        for i in range(num_episodes):
            seed = i
            self.progress.emit(f"\n--- Start evaluating scenario {i + 1}/{num_episodes} (seed={seed}) ---")
            ts_log_this_seed = {}
            metrics_log_this_seed = {}
            env.reset(seed=seed)
            grid_instance_for_this_seed = env.grid
            stations_list_for_this_seed = env.stations_list
            for algo_name in selected_algos:
                if self.is_stopped:
                    self.progress.emit("The evaluation task was terminated by the user.")
                    return
                if algo_name == "Baseline":
                    grid_for_baseline = deepcopy(grid_instance_for_this_seed)
                    metrics, ts_data = evaluate_baseline(CORE_PARAMS, seed, stations_list_for_this_seed,
                                                         grid_for_baseline, use_two_stage)
                else:
                    folder_name = "best_" + algo_name.lower()
                    model_type_key = next((key for key in model_name_to_class if key.lower() in algo_name.lower()),
                                          None)
                    if not model_type_key:
                        self.progress.emit(f"Warning: No matching Algorithm class found for {algo_name}, skipping.")
                        continue
                    model_path = os.path.join(PATHS["models_dir"], folder_name, "best_model.zip")
                    if not os.path.exists(model_path):
                        self.progress.emit(f"Warning: Model file {model_path} not found, skipping {algo_name}.")
                        continue
                    model_class = model_name_to_class[model_type_key]
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
                        env,
                        model_dir,
                        training_config=TRAINING_CONFIG,
                        manifest=manifest,
                    )
                    model_env = eval_normalizer if eval_normalizer is not None else env
                    model = model_class.load(model_path, env=model_env)
                    metrics, ts_data = evaluate_rl_agent(
                        model,
                        env,
                        seed,
                        obs_normalizer=eval_normalizer,
                    )
                if metrics:
                    metrics['Algorithm'] = algo_name
                    all_results_metrics.append(metrics)
                    ts_log_this_seed[algo_name] = ts_data
                    metrics_log_this_seed[algo_name] = metrics
                    self.progress.emit(f" - Algorithm '{algo_name}' evaluation completed.")
            if ts_log_this_seed:
                self.progress.emit("\nGenerating a visual report for the current scenario...")
                plot_and_save_results(ts_log_this_seed, seed, CORE_PARAMS)
                # plot_accumulated_costs(metrics_log_this_seed, ts_log_this_seed, seed, CORE_PARAMS,
                #                        grid_instance_for_this_seed)
                plot_voltage_snapshots(ts_log_this_seed, seed, CORE_PARAMS)
                plot_line_flow_snapshots_comparison(ts_log_this_seed, seed, CORE_PARAMS)
                plot_aggregated_ev_power(ts_log_this_seed, seed, CORE_PARAMS)
                plot_sop_flows(ts_log_this_seed, seed, CORE_PARAMS)
                plot_nop_status(ts_log_this_seed, seed, CORE_PARAMS)
                self.progress.emit("Report generated!")
        if all_results_metrics:
            results_df = pd.DataFrame(all_results_metrics)
            summary = results_df.groupby('Algorithm').mean()
            summary_std = results_df.groupby('Algorithm').std()
            self.progress.emit("\n\n" + "=" * 80)
            self.progress.emit("Evaluation results summary (Mean - average)")
            self.progress.emit("=" * 80)
            self.progress.emit(summary.to_string(float_format="%.4f"))

            self.progress.emit("\n\n" + "=" * 80)
            self.progress.emit("Summary of evaluation results (Std Dev - Standard Deviation)")
            self.progress.emit("=" * 80)
            self.progress.emit(summary_std.to_string(float_format="%.4f"))
            self.progress.emit("=" * 80)
