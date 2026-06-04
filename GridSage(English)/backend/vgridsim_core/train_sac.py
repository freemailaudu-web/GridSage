"""
File: train_sac.py
Description: SAC training script (time/number of steps/two stages and other parameters are all read from config.py/GUI)
Dependencies:
  - stable_baselines3
  - Within the project: power_grid_env.py, config.py, training_visualizer.py
"""

import os
import traceback
import json
import shutil
from datetime import datetime

# Compatibility note: LVGS_* environment keys remain unchanged because existing
# launchers and training automation may depend on them.
def suppress_windows_error_dialog():
    if os.name != "nt" or os.getenv("LVGS_SUPPRESS_WINDOWS_ERROR_DIALOG", "0") != "1":
        return
    try:
        import ctypes

        SEM_FAILCRITICALERRORS = 0x0001
        SEM_NOGPFAULTERRORBOX = 0x0002
        SEM_NOOPENFILEERRORBOX = 0x8000
        ctypes.windll.kernel32.SetErrorMode(
            SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX
        )
    except Exception:
        pass


suppress_windows_error_dialog()

import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from power_grid_env import PowerGridEnv
from training_visualizer import CostCurveCallback
from config import PATHS, CORE_PARAMS, TRAINING_CONFIG, load_gui_settings, get_effective_rl_hyperparams
from scenario_fingerprint import build_scenario_fingerprint, build_scenario_signature
from rl_normalization import save_vecnormalize_stats, wrap_callback_eval_env, wrap_training_env


def make_env(use_two_stage: bool):
    """
    Construction environment:
    - all "time related" parameters (start_hour/end_hour/step_minutes) from GUI settings or CORE_PARAMS
    - Do not hard-code any time parameters in the script
    """
    gui_params = load_gui_settings()

    params = {
        "grid_model": gui_params.get("grid_model", CORE_PARAMS.get("grid_model", "ieee33")),
        "solver": gui_params.get("solver", CORE_PARAMS.get("solver", "gurobi")),
        "start_hour": gui_params.get("start_hour", CORE_PARAMS.get("start_hour", 1)),
        "end_hour": gui_params.get("end_hour", CORE_PARAMS.get("end_hour", 24)),
        "step_minutes": gui_params.get("step_minutes", CORE_PARAMS.get("step_minutes", 60)),
        "distributed_energy": {
            "pv":   gui_params.get("use_pv",   CORE_PARAMS.get("distributed_energy", {}).get("pv", True)),
            "wind": gui_params.get("use_wind", CORE_PARAMS.get("distributed_energy", {}).get("wind", True)),
            "ess":  gui_params.get("use_ess",  CORE_PARAMS.get("distributed_energy", {}).get("ess", True)),
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
    suppress_windows_error_dialog()

    # ============== Path and directory ==============
    project_root = os.path.dirname(os.path.abspath(__file__))
    logs_root = PATHS.get("logs_dir", os.path.join(project_root, "logs"))
    models_root = os.path.join(project_root, "models")
    tb_root = os.path.join(project_root, "tensorboard_logs")
    os.makedirs(logs_root, exist_ok=True)
    os.makedirs(models_root, exist_ok=True)
    os.makedirs(tb_root, exist_ok=True)

    # ============== Unified reading of training configuration ==============
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
    full_training = os.getenv("LVGS_FULL_TRAINING", "0") == "1"
    FAST_TRAINING = (
        not full_training
        and (os.getenv("LVGS_FAST_TRAINING", "1") == "1" or TOTAL_TIMESTEPS <= 10_000)
    )

    # SAC hyperparameters: read GUI first, TRAINING_CONFIG as supplementary coverage
    gui_hparams = get_effective_rl_hyperparams("SAC", gui_settings=load_gui_settings())
    common_params = gui_hparams.get("common", {})
    specific_params = gui_hparams.get("specific", {})

    SAC_PARAMS = TRAINING_CONFIG.get("sac_params", {})
    learning_rate = SAC_PARAMS.get("learning_rate", common_params.get("learning_rate", 3e-4))
    buffer_size = SAC_PARAMS.get("buffer_size", 1_000_000)
    batch_size = SAC_PARAMS.get("batch_size", common_params.get("batch_size", 256))
    tau = SAC_PARAMS.get("tau", specific_params.get("tau", 0.005))
    gamma = SAC_PARAMS.get("gamma", common_params.get("gamma", 0.99))
    ent_coef = SAC_PARAMS.get("ent_coef", specific_params.get("ent_coef", 0.1))
    train_freq = SAC_PARAMS.get("train_freq", 1)
    gradient_steps = SAC_PARAMS.get("gradient_steps", 1)

    algo_tag = "sac"
    stage_tag = "2stage" if TWO_STAGE_TRAIN else "1stage"
    # Save each training session to a separate version directory to prevent the 1000-step model from being overwritten by the subsequent 500 steps of training.
    # The directory name also contains Algorithm, stage, step number, seed, and timestamp, which facilitates accurate matching of the "X-step model" on the evaluation end.
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    print(f"[SAC] This training version: {run_tag}")
    print(f"[SAC] Log directory: {LOG_DIR}")
    print(f"[SAC] model version directory: {BEST_MODEL_DIR}")
    print(f"[SAC] TensorBoard: {TENSORBOARD_LOG_DIR}")
    print(f"[SAC] Training Steps: {TOTAL_TIMESTEPS:,}")
    print(f"[SAC] Two-stage training: {TWO_STAGE_TRAIN}")
    print(f"[SAC] Fast training mode: {FAST_TRAINING}")
    print("=" * 80)

    # ============== Construct training/evaluation environment ==============
    try:
        # Short step training is only responsible for generating model weights, skipping expensive benchmark/evaluation callbacks.
        eval_env_raw = None
        eval_env = None
        if not FAST_TRAINING:
            eval_env_raw = make_env(use_two_stage=True)
            eval_env = wrap_callback_eval_env(eval_env_raw, gamma=float(gamma), training_config=TRAINING_CONFIG)

        # The training environment can choose two-stage or single-stage according to the configuration (the two-stage can be turned off first during the training period to speed up)
        train_env_raw = make_env(use_two_stage=TWO_STAGE_TRAIN)
        train_env = wrap_training_env(train_env_raw, gamma=float(gamma), training_config=TRAINING_CONFIG)

        # Unified random seed
        np.random.seed(RANDOM_SEED)
        try:
            train_env.seed(RANDOM_SEED)
        except Exception:
            pass
        if eval_env is not None:
            try:
                eval_env.seed(RANDOM_SEED)
            except Exception:
                pass

    except Exception as e:
        print(f"[SAC] Failed to create environment: {e}")
        traceback.print_exc()
        return

    # ============== Build SAC model ==============
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        verbose=1,
        seed=RANDOM_SEED,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        ent_coef=ent_coef,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
    )

    # ============== Callback: evaluate/ckpt/costcurve ==============
    callbacks = []
    cost_curve_cb = None
    if not FAST_TRAINING:
        eval_cb = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=BEST_MODEL_DIR,
            log_path=LOG_DIR,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=3,
            deterministic=True,
            render=False
        )
        checkpoint_cb = CheckpointCallback(
            save_freq=CHECKPOINT_FREQ,
            save_path=BEST_MODEL_DIR,
            name_prefix="ckpt"
        )
        agent_name_for_plot = f"SAC_{'Two_Stage' if TWO_STAGE_TRAIN else 'Single_Stage'}"

        cost_curve_cb = CostCurveCallback(
            eval_env=eval_env_raw,
            agent_name=agent_name_for_plot,
            save_path=BEST_MODEL_DIR,
            eval_freq=COST_CURVE_EVAL_FREQ
        )
        callbacks = [eval_cb, checkpoint_cb, cost_curve_cb]

    # ============== Start training ==============
    training_failed = False
    try:
        tb_run_name = f"seed_{RANDOM_SEED}"
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks or None,
            tb_log_name=tb_run_name,
            progress_bar=False,
        )

    except KeyboardInterrupt:
        training_failed = True
        print("\n[SAC] Received interrupt signal, save current model...")
    except Exception as e:
        training_failed = True
        print(f"[SAC] Training exception: {e}")
        traceback.print_exc()
    finally:
        # Save the final model and close the environment
        try:
            if cost_curve_cb is not None:
                cost_curve_cb.save_data()
        except Exception:
            pass
        print(f"[SAC] Save the final model to: {os.path.abspath(FINAL_MODEL_PATH)}")
        model.save(FINAL_MODEL_PATH)
        try:
            vecnormalize_rel_path = save_vecnormalize_stats(train_env, BEST_MODEL_DIR)
        except Exception as norm_err:
            vecnormalize_rel_path = None
            print(f"[RL-Normalization] Failed to save VecNormalize stats: {norm_err}")

        # manifest allows the back-end evaluation end and manual troubleshooting to know how many training steps this directory corresponds to.
        gui_settings = load_gui_settings()
        manifest = {
            "algo": "SAC",
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

        # Keep the "latest model" compatibility entry, but don't use it to represent a specific step version.
        try:
            shutil.copy2(FINAL_MODEL_PATH, LATEST_MODEL_PATH)
            print(f"[SAC] Latest model shortcut copy: {os.path.abspath(LATEST_MODEL_PATH)}")
        except Exception as copy_err:
            print(f"[SAC] Failed to write the latest model shortcut copy: {copy_err}")
        print("[SAC] training ended.")
        try:
            import sys

            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        if os.name == "nt" and not training_failed:
            os._exit(0)
        try:
            model.logger.close()
        except Exception:
            pass
        if os.getenv("LVGS_CLOSE_TRAIN_ENVS", "0") == "1":
            try:
                train_env.close()
                if eval_env is not None:
                    eval_env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
