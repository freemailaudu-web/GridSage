# ==============================================================================
# Coding Assistant Platform - Custom Algorithm plug-in sample
#
# File name: my_brilliant_algo.py
# Algorithm name: BrilliantAlgo (based on TD3)
# Description: This is an example of a plug-in file that can be run directly, used to test the plug-in architecture of the platform.
# ==============================================================================

# 1. Import your Algorithm implementation and all related dependencies
# You can import any library, including Algorithm classes you write yourself.
import os
import shutil
import numpy as np

# Import the core components of the platform
from power_grid_env import PowerGridEnv
from config import CORE_PARAMS, TRAINING_CONFIG, PATHS

# Import the basic libraries that your Algorithm depends on (here we use stable-baselines3)
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


# --- Your Algorithm core implementation ---
# In this example, we create a new "BrilliantAlgo" class by inheriting TD3.
# You can add custom logic here, such as modifying initialization parameters, rewriting training loops, etc.
# For testing purposes, even a simple rename can verify that the plug-in mechanism works.
class BrilliantAlgo(TD3):
    """
    A custom Algorithm sample, inherited from TD3 of stable_baselines3.
    """

    def __init__(self, policy, env, **kwargs):
        # You can add custom initialization logic here
        print("=" * 50)
        print("Initializing BrilliantAlgo - A custom TD3 variant!")
        print("=" * 50)
        # Call the constructor of the parent class
        super().__init__(policy, env, **kwargs)

    def learn(self, total_timesteps, callback=None, log_interval=4, tb_log_name="BrilliantAlgo",
              reset_num_timesteps=True, progress_bar=False):
        # You can override the learn method to implement a custom training loop
        print(f"\n--- Starting the learning process using BrilliantAlgo's custom learn method! ---")
        # In this example, we simply call the learn method of the parent class
        return super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar)


# ==============================================================================
# 2. [Must be implemented] Algorithm registration function
# This is the key for the platform to automatically discover your Algorithm. The function name must be register_algorithm.
# It returns a dictionary containing the short name of the Algorithm 'name' (uppercase) and the 'class' required when loading the model.
# ==============================================================================
def register_algorithm():
    """Register the Algorithm in this file to the platform."""
    return {
        'name': 'BRILLIANTALGO', # <-- Give your Algorithm a unique, uppercase abbreviation
        'class': BrilliantAlgo # <-- Point to the Python class of your Algorithm
    }


# ==============================================================================
# 3. [Recommended implementation] Training entrance
# This part of the code allows your plug-in file to be run directly to start training.
# ==============================================================================
if __name__ == '__main__':
    # 3.1. Name your model. This will determine the folder name where the logs and final model will be saved.
    # (We use 'BrilliantAlgo' here to correspond to the class name above)
    model_display_name = "BrilliantAlgo_Two_Stage"

    # 3.2. Create environment (consistent with other parts of the platform)
    print(f"Create environment: {model_display_name}")
    use_two_stage = 'Two_Stage' in model_display_name
    env = PowerGridEnv(gui_params=CORE_PARAMS, use_two_stage_flow=use_two_stage)
    eval_env = PowerGridEnv(gui_params=CORE_PARAMS, use_two_stage_flow=use_two_stage)

    # 3.3. Define callback function to evaluate and save the best model during training
    log_path = os.path.join(PATHS["logs_dir"], model_display_name)
    # Evaluation callback
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_path,
                                 log_path=log_path, eval_freq=TRAINING_CONFIG["eval_freq"],
                                 deterministic=True, render=False)
    # Checkpoint callback (optional, used to save the model regularly to prevent training interruption)
    checkpoint_callback = CheckpointCallback(save_freq=TRAINING_CONFIG["checkpoint_freq"],
                                             save_path=log_path,
                                             name_prefix="rl_model")

    # 3.4. Instantiate your model
    # Pass in all the hyperparameters required by your Algorithm here
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = BrilliantAlgo(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        tensorboard_log=PATHS["tensorboard_logs"],
        learning_rate=0.001,
        buffer_size=100000,
        learning_starts=10000,
        batch_size=256,
        train_freq=(1, "episode"),
        gradient_steps=-1
    )

    # 3.5. Start training
    print(f"--- Model training is about to start: {model_display_name} ---")
    model.learn(
        total_timesteps=TRAINING_CONFIG["total_timesteps"],
        callback=[eval_callback, checkpoint_callback], # Multiple callbacks can be passed in
        tb_log_name=model_display_name
    )

    # 3.6. After training, the best model is automatically copied to the main model folder for discovery by the evaluation script.
    best_model_path = os.path.join(log_path, "best_model.zip")
    final_model_dir = os.path.join(PATHS["models_dir"], f"best_{model_display_name.lower()}")
    os.makedirs(final_model_dir, exist_ok=True)
    final_model_path = os.path.join(final_model_dir, "best_model.zip")

    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, final_model_path)
        print(f"\n--- Training completed! The best model has been automatically saved to: {final_model_path} ---")
    else:
        print(f"\n--- Training completed, but the best model file was not found. Please check the training log. ---")