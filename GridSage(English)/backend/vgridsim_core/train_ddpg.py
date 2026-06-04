"""
File: train_ddpg.py
Description: DDPG training script (parameters fully aligned in config.py, no more hard-coded time settings)
Dependencies:
  - stable_baselines3
  - Within the project: power_grid_env.py, config.py, training_visualizer.py
"""

import os
import traceback
import json
import shutil
from datetime import datetime
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise

from power_grid_env import PowerGridEnv
from training_visualizer import CostCurveCallback
from config import PATHS, CORE_PARAMS, TRAINING_CONFIG, load_gui_settings, get_effective_rl_hyperparams
from scenario_fingerprint import build_scenario_fingerprint, build_scenario_signature
from rl_normalization import save_vecnormalize_stats, wrap_callback_eval_env, wrap_training_env


def make_env(use_two_stage: bool):
    """Construction environment: all "time related" parameters come from config.py/GUI settings."""
    gui_params = load_gui_settings()

    params = {
        "grid_model": gui_params.get("grid_model", CORE_PARAMS.get("grid_model", "ieee33")),
        "solver": gui_params.get("solver", CORE_PARAMS.get("solver", "gurobi")),
        "start_hour": gui_params.get("start_hour", CORE_PARAMS.get("start_hour", 1)),
        "end_hour": gui_params.get("end_hour", CORE_PARAMS.get("end_hour", 24)),
        "step_minutes": gui_params.get("step_minutes", CORE_PARAMS.get("step_minutes", 60)),
        "distributed_energy": {
            "pv": gui_params.get("use_pv", CORE_PARAMS.get("distributed_energy", {}).get("pv", True)),
            "wind": gui_params.get("use_wind", CORE_PARAMS.get("distributed_energy", {}).get("wind", True)),
            "ess": gui_params.get("use_ess", CORE_PARAMS.get("distributed_energy", {}).get("ess", True)),
        },
        "sop_nodes_active": gui_params.get("use_sop", CORE_PARAMS.get("sop_nodes_active", True)),
        "nop_nodes_active": gui_params.get("use_nop", CORE_PARAMS.get("nop_nodes_active", True)),
        "slack_bus": CORE_PARAMS.get("slack_bus", "b1"),
        "base_power": CORE_PARAMS.get("base_power", 1.0),
        "ev_data_source": gui_params.get("ev_data_source", CORE_PARAMS.get("ev_data_source", "random")),
        "ev_params": gui_params.get("ev_params", CORE_PARAMS.get("ev_params", {})),
        "reward_weights": gui_params.get("reward_weights", CORE_PARAMS.get("reward_weights", {})),
        "reward_mode": gui_params.get("reward_mode", CORE_PARAMS.get("reward_mode", "grid_operator")),
        "station_operator": gui_params.get("station_operator", CORE_PARAMS.get("station_operator", {})),
        "global_pv_multiplier": gui_params.get("global_pv_multiplier", CORE_PARAMS.get("global_pv_multiplier", 1.0)),
        "global_load_multiplier": gui_params.get("global_load_multiplier", CORE_PARAMS.get("global_load_multiplier", 1.0)),
        "global_ev_multiplier": gui_params.get("global_ev_multiplier", CORE_PARAMS.get("global_ev_multiplier", 1.0)),
        "obs_power_base_kw": gui_params.get("obs_power_base_kw", CORE_PARAMS.get("obs_power_base_kw", 1000.0)),
        "obs_price_base": gui_params.get("obs_price_base", CORE_PARAMS.get("obs_price_base", 1.0)),
        "time_profiles": gui_params.get("time_profiles", CORE_PARAMS.get("time_profiles", {})),
        "node_overrides": gui_params.get("node_overrides", CORE_PARAMS.get("node_overrides", {})),
        "disabled_devices": gui_params.get("disabled_devices", CORE_PARAMS.get("disabled_devices", {})),
        "reconfiguration_mode": gui_params.get("reconfiguration_mode", CORE_PARAMS.get("reconfiguration_mode", "radial_reconfiguration")),
        "selected_reconfiguration_plan_id": gui_params.get("selected_reconfiguration_plan_id", CORE_PARAMS.get("selected_reconfiguration_plan_id", "R0")),
        "available_reconfiguration_plans": gui_params.get("available_reconfiguration_plans", CORE_PARAMS.get("available_reconfiguration_plans", [])),
        "reconfiguration_constraints": gui_params.get("reconfiguration_constraints", CORE_PARAMS.get("reconfiguration_constraints", {})),
    }
    return PowerGridEnv(gui_params=params, use_two_stage_flow=use_two_stage)


def main():
    # path
    project_root = os.path.dirname(os.path.abspath(__file__))
    logs_root = PATHS.get("logs_dir", os.path.join(project_root, "logs"))
    models_root = os.path.join(project_root, "models")
    tb_root = os.path.join(project_root, "tensorboard_logs")
    os.makedirs(logs_root, exist_ok=True)
    os.makedirs(models_root, exist_ok=True)
    os.makedirs(tb_root, exist_ok=True)

    # Configuration
    TWO_STAGE_TRAIN = bool(TRAINING_CONFIG.get("two_stage_training", True))
    TOTAL_TIMESTEPS = int(TRAINING_CONFIG.get("total_timesteps", 300_000))
    EVAL_FREQ = int(TRAINING_CONFIG.get("eval_freq", 5_000))
    CHECKPOINT_FREQ = int(TRAINING_CONFIG.get("checkpoint_freq", 10_000))
    COST_CURVE_EVAL_FREQ = int(TRAINING_CONFIG.get("cost_curve_eval_freq", 500))
    RANDOM_SEED = int(TRAINING_CONFIG.get("seed", 0))

    env_total = os.getenv("TRAINING_TOTAL_TIMESTEPS")
    if env_total is not None:
        try:
            TOTAL_TIMESTEPS = int(env_total)
        except ValueError:
            pass

    env_seed = os.getenv("TRAINING_SEED")
    if env_seed is not None:
        try:
            RANDOM_SEED = int(env_seed)
        except ValueError:
            pass


    norm_hparams = get_effective_rl_hyperparams("DDPG", gui_settings=load_gui_settings())
    norm_common_params = norm_hparams.get("common", {})
    normalization_gamma = float(
        TRAINING_CONFIG.get("ddpg_params", {}).get("gamma", norm_common_params.get("gamma", 0.99))
    )

    algo_tag = "ddpg"
    stage_tag = "2stage" if TWO_STAGE_TRAIN else "1stage"
    # Each training session is saved to an independent version directory to avoid overwriting each other with different steps/batches.
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{algo_tag}_{stage_tag}_{TOTAL_TIMESTEPS}steps_seed{RANDOM_SEED}_{run_stamp}"

    LOG_DIR = os.path.join(logs_root, run_tag)
    BEST_MODEL_DIR = os.path.join(models_root, run_tag)
    FINAL_MODEL_PATH = os.path.join(BEST_MODEL_DIR, "final_model.zip")
    LATEST_MODEL_PATH = os.path.join(models_root, f"{algo_tag}_{stage_tag}_latest.zip")
    TENSORBOARD_LOG_DIR = os.path.join(tb_root, run_tag)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    print("=" * 80)
    print(f"[DDPG] This training version: {run_tag}")
    print(f"[DDPG] Log directory: {LOG_DIR}")
    print(f"[DDPG] Model version directory: {BEST_MODEL_DIR}")
    print(f"[DDPG] TensorBoard: {TENSORBOARD_LOG_DIR}")
    print(f"[DDPG] Training Steps: {TOTAL_TIMESTEPS:,}")
    print(f"[DDPG] Two-stage training: {TWO_STAGE_TRAIN}")
    print("=" * 80)

    try:
        # Assessment environment: fixed two stages to ensure fairness
        eval_env_raw = make_env(use_two_stage=True)
        eval_env = wrap_callback_eval_env(
            eval_env_raw, gamma=normalization_gamma, training_config=TRAINING_CONFIG
        )

        # Training environment
        train_env_raw = make_env(use_two_stage=TWO_STAGE_TRAIN)
        train_env = wrap_training_env(
            train_env_raw, gamma=normalization_gamma, training_config=TRAINING_CONFIG
        )

        #seed
        np.random.seed(RANDOM_SEED)
        for _env in [train_env, eval_env]:
            try: _env.seed(RANDOM_SEED)
            except Exception: pass

    except Exception as e:
        print(f"[DDPG] Failed to create environment: {e}")
        traceback.print_exc()
        return

    gui_hparams = get_effective_rl_hyperparams("DDPG", gui_settings=load_gui_settings())
    common_params = gui_hparams.get("common", {})
    specific_params = gui_hparams.get("specific", {})
    ddpg_params = TRAINING_CONFIG.get("ddpg_params", {})

    learning_rate = float(ddpg_params.get("learning_rate", common_params.get("learning_rate", 1e-3)))
    gamma = float(ddpg_params.get("gamma", common_params.get("gamma", 0.99)))
    batch_size = int(ddpg_params.get("batch_size", common_params.get("batch_size", 256)))
    tau = float(ddpg_params.get("tau", specific_params.get("tau", 0.005)))
    action_noise_sigma = float(ddpg_params.get("action_noise", specific_params.get("action_noise", 0.1)))

    # Action noise (required for DDPG)
    action_dim = train_env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(action_dim, dtype=np.float32),
        sigma=action_noise_sigma * np.ones(action_dim, dtype=np.float32)
    )

    # Model
    model = DDPG(
        policy="MlpPolicy",
        env=train_env,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        verbose=1,
        seed=RANDOM_SEED,
        action_noise=action_noise,
        learning_rate=learning_rate,
        buffer_size=200_000, # Larger experience buffer to improve sample diversity
        learning_starts=2_000, # Save 2k steps before training to avoid early over-fitting noise
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=(1, "step"), # Trigger update after each step of collection
        gradient_steps=2, # Do 2 gradient steps for each update (unable to learn → 2~4, unstable → 1)
        policy_kwargs=dict(net_arch=[256, 256]), # Thicker MLP
    )

    # Callback
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=3,
        deterministic=True,
        render=False
    )
    checkpoint_cb = CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=BEST_MODEL_DIR, name_prefix="ckpt")
    # Define a clear name for plotting
    agent_name_for_plot = f"DDPG_{'Two_Stage' if TWO_STAGE_TRAIN else 'Single_Stage'}"

    cost_curve_cb = CostCurveCallback(
        eval_env=eval_env_raw,
        agent_name=agent_name_for_plot,
        save_path=BEST_MODEL_DIR,
        eval_freq=COST_CURVE_EVAL_FREQ
    )

    # training
    try:
        tb_run_name = f"seed_{RANDOM_SEED}"
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[eval_cb, checkpoint_cb, cost_curve_cb],
            tb_log_name=tb_run_name,
            progress_bar=True,
        )

    except KeyboardInterrupt:
        print("\n[DDPG] Interrupt, save model...")
    except Exception as e:
        print(f"[DDPG] Training exception: {e}")
        traceback.print_exc()
    finally:
        try: cost_curve_cb.save_data()
        except Exception: pass
        print(f"[DDPG] Save the final model to: {os.path.abspath(FINAL_MODEL_PATH)}")
        model.save(FINAL_MODEL_PATH)
        try:
            vecnormalize_rel_path = save_vecnormalize_stats(train_env, BEST_MODEL_DIR)
        except Exception as norm_err:
            vecnormalize_rel_path = None
            print(f"[RL-Normalization] Failed to save VecNormalize stats: {norm_err}")

        gui_settings = load_gui_settings()
        manifest = {
            "algo": "DDPG",
            "stage_tag": stage_tag,
            "total_timesteps": TOTAL_TIMESTEPS,
            "seed": RANDOM_SEED,
            "two_stage_training": TWO_STAGE_TRAIN,
            "created_at": run_stamp,
            "final_model": os.path.abspath(FINAL_MODEL_PATH),
            "best_model": os.path.abspath(os.path.join(BEST_MODEL_DIR, "best_model.zip")),
            "grid_model": gui_settings.get("grid_model", "ieee33"),
            "node_overrides_keys": sorted(gui_settings.get("node_overrides", {}).keys()),
            "scenario_signature": build_scenario_signature(gui_settings),
            "scenario_fingerprint": build_scenario_fingerprint(gui_settings),
            "use_vec_normalize": bool(TRAINING_CONFIG.get("use_vec_normalize", True)),
            "vecnormalize_path": vecnormalize_rel_path,
            "obs_power_base_kw": float(getattr(train_env_raw, "obs_power_base_kw", 1000.0)),
            "obs_price_base": float(getattr(train_env_raw, "obs_price_base", 1.0)),
        }
        with open(os.path.join(BEST_MODEL_DIR, "model_manifest.json"), "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, ensure_ascii=False, indent=2)

        try:
            shutil.copy2(FINAL_MODEL_PATH, LATEST_MODEL_PATH)
            print(f"[DDPG] Quick copy of the latest model: {os.path.abspath(LATEST_MODEL_PATH)}")
        except Exception as copy_err:
            print(f"[DDPG] Failed to write the latest model shortcut copy: {copy_err}")
        try: model.logger.close()
        except Exception: pass
        try:
            train_env.close()
            eval_env.close()
        except Exception:
            pass
        print("[DDPG] training ended.")


if __name__ == "__main__":
    main()
