import sys
import os
from config import load_gui_settings
from power_grid_env import PowerGridEnv
import numpy as np

def test():
    gui_params = load_gui_settings()
    gui_params['enable_ev_urgency_penalty'] = True
    gui_params['ev_dense_gap_penalty'] = 2.0
    gui_params['slack_bus'] = 'b1'
    gui_params['base_power'] = 1.0
    env = PowerGridEnv(gui_params=gui_params)
    obs, info = env.reset(seed=42)
    print("Initial Reset done.")
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, step_info = env.step(action)
        print(f"Step {step}, reward: {reward:.4f}, ev_urgency_penalty: {step_info.get('ev_urgency_penalty', 0):.4f}, ev_shortage: {step_info.get('step_ev_shortage_penalty_cost', 0):.4f}")

if __name__ == "__main__":
    test()
