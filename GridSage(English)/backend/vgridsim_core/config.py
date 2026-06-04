import os
import json

# ==============================================================================
# 1. Basic path configuration
# ------------------------------------------------------------------------------
# Description: Define the file paths that need to be used in all projects. All paths are dynamically generated based on the location of this file,
# Allows the project to be moved anywhere without modifying the path.
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "data": os.path.join(PROJECT_ROOT, "data"),
    "grid_params_excel": os.path.join(PROJECT_ROOT, "data", "grid_parameters.xlsx"),
    "ev_scenarios_csv": os.path.join(PROJECT_ROOT, "data", "my_ev_scenarios.csv"),
    "logs_dir": os.path.join(PROJECT_ROOT, "logs"),
    "models_dir": os.path.join(PROJECT_ROOT, "models"),
    "results_dir": os.path.join(PROJECT_ROOT, "results_outputs"),
    "tensorboard_logs": os.path.join(PROJECT_ROOT, "tensorboard_logs"),
    "gui_settings": os.path.join(PROJECT_ROOT, "gui_settings.json"),
}
# Allow critical paths to be overridden via environment variables
import os as _os
PATHS["models_dir"]        = _os.getenv("MODELS_DIR",        PATHS["models_dir"])
PATHS["grid_params_excel"] = _os.getenv("GRID_PARAMS_XLSX",  PATHS["grid_params_excel"])
PATHS["ev_scenarios_csv"]  = _os.getenv("EV_SCENARIOS_CSV",  PATHS["ev_scenarios_csv"])
PATHS["results_dir"]       = _os.getenv("RESULTS_DIR",       PATHS.get("results_dir", os.path.join(PROJECT_ROOT, "results_outputs")))
PATHS["logs_dir"]          = _os.getenv("LOGS_DIR",          PATHS["logs_dir"])
PATHS["tensorboard_logs"]  = _os.getenv("TB_LOGS_DIR",       PATHS["tensorboard_logs"])

# ==============================================================================
# 2. Simulation and power grid core parameters (as the default value of GUI)
# ------------------------------------------------------------------------------
# Description: This dictionary is now used as the default parameter when starting the GUI. When actually running, the platform will give priority to using
# Parameters saved on the GUI interface.
# ==============================================================================
CORE_PARAMS = {
    # --- Time setting ---
    "start_hour": 0,
    "end_hour": 24,
    "step_minutes": 60,

    # --- Basic settings of distribution network model ---
    "grid_model": "ieee33", # Optional: "ieee33", "ieee69", "ieee123"
    "slack_bus": "b1", # Slack node
    "base_power": 1.0, # unit MVA

    # --- Distribution network component switch ---
    "distributed_energy": {
        "pv": True, # Whether to include photovoltaic
        "wind": True, # Whether to include wind power
        "ess": True # Whether to include energy storage system
    },
    "sop_nodes_active": True, # Whether to include SOP (soft opening point)
    "nop_nodes_active": True, # Whether to include NOP (normally open nodes)

    # --- Solver settings ---
    "solver": "gurobi", # Optional: "gurobi", "glpk", "cbc", "scip"
    "ev_data_source": "random",
    "reward_mode": "grid_operator",
    "ev_dense_gap_penalty": 2.0,
    "enable_ev_urgency_penalty": True,

    "ev_params": {
        "capacity_kwh": 70.0,
        "max_charge_kw": 60.0,
        "max_discharge_kw": 25.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.90
    },

    "reward_weights": {
        "ev_kwh_shortage_penalty": -20.0,
        "voltage_violation_penalty": -100.0,
        "cost_penalty_factor": 1.0,
        "opendss_failure_penalty": -5000.0
    },

    "station_operator": {
        "charge_service_price": 1.20,
        "v2g_subsidy_price": 0.80,
        "include_grid_cost": False,
        "include_generation_cost": True,
        "include_ess_cost": True,
        "include_sop_loss_cost": True,
        "include_penalty_cost": True
    }

}

# ==============================================================================
# 3. Baseline (global optimization) parameters
# ==============================================================================
BASELINE_PARAMS = {
    "penalty_factors": {
        "ev_not_full_penalty": 100, # EV Undercharge Penalty (yuan/kWh)
        "slack_power_penalty": 1e7, # Power imbalance slack variable penalty
        "sop_capacity_penalty": 1e6, # SOP capacity slack variable penalty
        "nop_voltage_penalty": 1e6, # NOP voltage slack variable penalty
    },
    "ess_degradation_cost": 0.0, # ESS discharge depreciation cost (yuan/kWh)
}

# ==============================================================================
# 4. Running mode and evaluation configuration
# ==============================================================================
EVALUATION_CONFIG = {
    "flow_mode": 'two_stage', 
    "num_test_episodes": 1,
    "standards": {
        "voltage_min_pu": 0.95,
        "voltage_max_pu": 1.05,
        "ev_charged_soc_threshold": 0.95,
    }
}

# ==============================================================================
# 5. Reinforcement learning environment and reward function configuration
# ==============================================================================
RL_ENV_CONFIG = {
    "reward_weights": {
        "ev_kwh_shortage_penalty": -20.0,
        "voltage_violation_penalty": -100.0,
        "cost_penalty_factor": 1,
    },
    "penalties": {
            "opendss_failure_penalty": -5000.0
    },
    # scaling factor
    "reward_scale": 0.01,
    "action_projection": {
        "enable_sop_capacity_projection": True,
        "sop_capacity_projection_margin": 0.98,
    },
}

# ==============================================================================
# 6. Training process configuration
# ==============================================================================
TRAINING_CONFIG = {
    "total_timesteps": 100000,
    "eval_freq": 240,
    "checkpoint_freq": 10000,
    "cost_curve_eval_freq": 240
}
# ==============================================================================
# RL Algorithm hyperparameter configuration
# ==============================================================================
RL_HYPERPARAMS = {
    "common": {
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "batch_size": 256
    },
    "algo_specific": {
        "PPO": {"clip_range": 0.2, "ent_coef": 0.0},
        "SAC": {"tau": 0.005, "ent_coef": 0.1}, # SAC's ent_coef first uses a floating point number to represent the initial value.
        "DDPG": {"tau": 0.005, "action_noise": 0.1},
        "TD3": {"policy_delay": 2, "target_policy_noise": 0.2}
    }
}
# ==============================================================================
# 7. Automatically calculated derived parameters
# ==============================================================================
try:
    _total_duration_hours = CORE_PARAMS['end_hour'] - CORE_PARAMS['start_hour']
    if _total_duration_hours <= 0:
        raise ValueError("Simulation end time must be greater than start time")

    if 60 % CORE_PARAMS['step_minutes'] != 0:
        raise ValueError("Simulation step size must be divisible by 60")

    _steps_per_hour = 60 // CORE_PARAMS['step_minutes']
    TIMESTEPS_PER_EPISODE = int(_total_duration_hours * _steps_per_hour)
except (KeyError, ValueError) as e:
    print(f"Error: Automatic calculation of total steps failed - {e}")
    TIMESTEPS_PER_EPISODE = 24


def load_gui_settings():
    """
    Load GUI settings. If the configuration file exists, it is read; otherwise, the default value in config.py is used.
    """
    if os.path.exists(PATHS["gui_settings"]):
        with open(PATHS["gui_settings"], 'r') as f:
            return json.load(f)
    else:
        # If the file does not exist, build a subset of the GUI configuration from CORE_PARAMS
        default_settings = {
            "grid_model": CORE_PARAMS["grid_model"],
            "solver": CORE_PARAMS["solver"],
            "start_hour": CORE_PARAMS["start_hour"],
            "end_hour": CORE_PARAMS["end_hour"],
            "step_minutes": CORE_PARAMS["step_minutes"],
            "use_pv": CORE_PARAMS["distributed_energy"]["pv"],
            "use_wind": CORE_PARAMS["distributed_energy"]["wind"],
            "use_ess": CORE_PARAMS["distributed_energy"]["ess"],
            "use_sop": CORE_PARAMS["sop_nodes_active"],
            "use_nop": CORE_PARAMS["nop_nodes_active"],

            "ev_data_source": CORE_PARAMS["ev_data_source"],
            "ev_params": CORE_PARAMS["ev_params"],
            "reward_weights": CORE_PARAMS["reward_weights"],
            "reward_mode": CORE_PARAMS["reward_mode"],
            "station_operator": CORE_PARAMS["station_operator"],
            "ev_dense_gap_penalty": CORE_PARAMS["ev_dense_gap_penalty"],
            "enable_ev_urgency_penalty": CORE_PARAMS["enable_ev_urgency_penalty"],


            "rl_common": {
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "batch_size": 256
            },
            "rl_specific": {
                "PPO": {"clip_range": 0.2, "ent_coef": 0.0},
                "SAC": {"tau": 0.005, "ent_coef": 0.1},
                "DDPG": {"tau": 0.005, "action_noise": 0.1},
                "TD3": {"policy_delay": 2, "target_policy_noise": 0.2}
            }
        }
        return default_settings

def get_effective_rl_hyperparams(algo_name: str, gui_settings=None) -> dict:
    """
    Unified calculation of the "final effective" RL hyperparameters.

    Priority:
    1) gui_settings.json saved by GUI
    2) RL_HYPERPARAMS default value in config.py

    Return format:
    {
        "common": {...},
        "specific": {...}
    }
    """
    algo_key = str(algo_name).upper()
    settings = gui_settings if isinstance(gui_settings, dict) else load_gui_settings()

    common_defaults = dict(RL_HYPERPARAMS.get("common", {}))
    specific_defaults = dict(RL_HYPERPARAMS.get("algo_specific", {}).get(algo_key, {}))

    gui_common = dict(settings.get("rl_common", {})) if isinstance(settings, dict) else {}
    gui_specific_all = settings.get("rl_specific", {}) if isinstance(settings, dict) else {}
    gui_specific = dict(gui_specific_all.get(algo_key, {})) if isinstance(gui_specific_all, dict) else {}

    return {
        "common": {**common_defaults, **gui_common},
        "specific": {**specific_defaults, **gui_specific},
    }


