import os
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
CORE_PATH = os.path.join(ROOT, "backend", "vgridsim_core")
BACKEND_PATH = os.path.join(ROOT, "backend")

sys.path.insert(0, CORE_PATH)
sys.path.insert(0, BACKEND_PATH)
os.chdir(CORE_PATH)

from config import CORE_PARAMS
from evaluate_agents import evaluate_baseline
from power_grid_env import PowerGridEnv


def main():
    env = PowerGridEnv(gui_params=CORE_PARAMS, use_two_stage_flow=True)
    env.reset(seed=0)
    metrics, _ = evaluate_baseline(CORE_PARAMS, 0, env.stations_list, env.grid, use_two_stage=True)
    print(metrics)


if __name__ == "__main__":
    main()
