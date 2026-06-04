"""
File: tune_ppo.py
Description: Automated PPO hyperparameter tuning script.
      Traverse several sets of core hyperparameters, train the model and save it to facilitate subsequent comparison of the optimal results.
"""

import os
import traceback
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# Import your existing environment and configuration
from power_grid_env import PowerGridEnv
from config import PATHS, CORE_PARAMS, TRAINING_CONFIG, load_gui_settings


def make_env(use_two_stage: bool):
    """Construct environment: reuse your original environment construction logic"""
    gui_params = load_gui_settings()
    params = {
        "grid_model": gui_params.get("grid_model", CORE_PARAMS.get("grid_model", "ieee33")),
        "solver": gui_params.get("solver", CORE_PARAMS.get("solver", "gurobi")),
        "start_hour": gui_params.get("start_hour", CORE_PARAMS.get("start_hour", 1)),
        "end_hour": gui_params.get("end_hour", CORE_PARAMS.get("end_hour", 24)),
        "step_minutes": gui_params.get("step_minutes", CORE_PARAMS.get("step_minutes", 60)),
        "distributed_energy": CORE_PARAMS.get("distributed_energy", {}),
        "sop_nodes_active": CORE_PARAMS.get("sop_nodes_active", True),
        "nop_nodes_active": CORE_PARAMS.get("nop_nodes_active", True),
        "slack_bus": CORE_PARAMS.get("slack_bus", "b1"),
        "base_power": CORE_PARAMS.get("base_power", 1.0),
        "ev_data_source": gui_params.get("ev_data_source", CORE_PARAMS.get("ev_data_source", "random")),
        "ev_params": gui_params.get("ev_params", CORE_PARAMS.get("ev_params", {})),
        "reward_weights": gui_params.get("reward_weights", CORE_PARAMS.get("reward_weights", {})),
        "reward_mode": gui_params.get("reward_mode", CORE_PARAMS.get("reward_mode", "grid_operator")),
        "station_operator": gui_params.get("station_operator", CORE_PARAMS.get("station_operator", {})),
        "reconfiguration_mode": gui_params.get("reconfiguration_mode", CORE_PARAMS.get("reconfiguration_mode", "radial_reconfiguration")),
        "selected_reconfiguration_plan_id": gui_params.get("selected_reconfiguration_plan_id", CORE_PARAMS.get("selected_reconfiguration_plan_id", "R0")),
        "available_reconfiguration_plans": gui_params.get("available_reconfiguration_plans", CORE_PARAMS.get("available_reconfiguration_plans", [])),
        "reconfiguration_constraints": gui_params.get("reconfiguration_constraints", CORE_PARAMS.get("reconfiguration_constraints", {})),
    }
    return PowerGridEnv(gui_params=params, use_two_stage_flow=use_two_stage)


def run_tuning_experiment():
    """Execute PPO grid parameter adjustment experiment"""
    #Basic configuration
    TOTAL_TIMESTEPS = 100000 # Consistent with 100000 steps in the paper
    EVAL_FREQ = 5000
    RANDOM_SEED = 0

    # ==========================================================
    # Step 1: Define the hyperparameter grid to test
    # Default value reference: learning_rate=3e-3, ent_coef=0.01, batch_size=512
    # ==========================================================
    hyperparams_grid = {
        "PPO_Default": {"learning_rate": 3e-3, "ent_coef": 0.01, "batch_size": 512},
        "PPO_High_Expl": {"learning_rate": 3e-3, "ent_coef": 0.05, "batch_size": 512}, # Increase exploration rate
        "PPO_Low_LR": {"learning_rate": 3e-4, "ent_coef": 0.02, "batch_size": 1024} # Reduce the learning rate and increase the batch to make the update more stable
    }

    print("=" * 50)
    print("Start PPO hyperparameter tuning experiment...")
    print("=" * 50)

    for config_name, params in hyperparams_grid.items():
        print(f"\n--->Training configuration: {config_name}")
        print(f"parameters: {params}")

        # Set a special save path
        log_dir = os.path.join(PATHS["logs_dir"], f"tuning_{config_name}")
        best_model_dir = os.path.join(PATHS["models_dir"], f"tuning_{config_name}")
        tb_log_dir = os.path.join(PATHS["tensorboard_logs"], f"tuning_{config_name}")

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(best_model_dir, exist_ok=True)

        # Step 2: Create a separate environment
        try:
            train_env_raw = make_env(use_two_stage=True)
            train_env = DummyVecEnv([lambda: Monitor(train_env_raw)])

            eval_env_raw = make_env(use_two_stage=True)
            eval_env = Monitor(eval_env_raw)

            np.random.seed(RANDOM_SEED)
        except Exception as e:
            print(f"Environment creation failed: {e}")
            continue

        # Step 3: Instantiate PPO and inject tuning parameters
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=params["learning_rate"],
            ent_coef=params["ent_coef"],
            batch_size=params["batch_size"],
            tensorboard_log=tb_log_dir,
            verbose=0, # Set to 0 to reduce screen spam
            seed=RANDOM_SEED,
        )

        # Set evaluation callback
        eval_cb = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=best_model_dir,
            log_path=log_dir,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=3,
            deterministic=True,
            render=False
        )

        # Step 4: Start training
        try:
            model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_cb)
            print(f"Configuration {config_name} training is completed. The optimal model has been saved to: {best_model_dir}")
        except Exception as e:
            print(f"Training exception: {e}")
            traceback.print_exc()
        finally:
            train_env.close()
            eval_env.close()


if __name__ == "__main__":
    run_tuning_experiment()
