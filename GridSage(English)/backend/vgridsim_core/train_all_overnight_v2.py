# -*- coding: utf-8 -*-
"""
training_all_parallel.py
Start the training of SAC, DDPG, TD3, and PPO in parallel, suitable for "running all night".
- Open a subprocess for each Algorithm (call existing train_*.py)
- Limit the number of CPU threads per process
- Unified archive of models/charts/data of each child process to runs/<run_id>/<Algo>/artifacts
"""
import argparse
import datetime as dt
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    "SAC":  "train_sac.py",
    "DDPG": "train_ddpg.py",
    "TD3":  "train_td3.py",
    "PPO":  "train_ppo.py",
}
DEFAULT_ORDER = ["SAC", "DDPG", "TD3", "PPO"]
DEFAULT_SEEDS = [0, 1, 2]

def _ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _pump_to_console_and_file(prefix: str, pipe, logfile: Path):
    """Write the output of the child process to the console and log file at the same time"""
    with open(logfile, "wb") as lf:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            try:
                sys.stdout.buffer.write(b"[" + prefix.encode("utf-8") + b"] " + chunk)
                sys.stdout.flush()
            except Exception:
                pass
            lf.write(chunk)

def _collect_artifacts(work_dir: Path, artifacts_dir: Path):
    """Pull common products in the working directory of the child process into artifacts"""
    patterns = [
        "*.zip", "*.pth", "*.pt",
        "*.png", "*.jpg",
        "*.pkl", "*.npz", "*.npy", "*.csv", "*.json",
        "events.*",  # TensorBoard
    ]
    _ensure(artifacts_dir)
    for pat in patterns:
        for src in work_dir.rglob(pat):
            # Avoid copying artifacts themselves
            if artifacts_dir in src.parents:
                continue
            dst = artifacts_dir / src.name
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass

def launch_one(algo: str, run_dir: Path, per_threads: int, nice: str, total_steps: int, seed: int, device: str, tag: str):
    script = SCRIPTS[algo]
    script_path = PROJECT_ROOT / script
    if not script_path.exists():
        raise FileNotFoundError(f"{script} does not exist, please confirm the path.")

    work_dir = _ensure(run_dir / algo)
    logs_dir = _ensure(work_dir / "logs")
    artifacts_dir = _ensure(work_dir / "artifacts")
    logfile = logs_dir / "train.log"

    # Process environment
    env = os.environ.copy()
    # Give the "soft parameter channel" to the subscript, the subscript can optionally read it, and it will not be affected if it is not read.
    if total_steps > 0:
        env["TRAINING_TOTAL_TIMESTEPS"] = str(total_steps)
    if seed >= 0:
        env["TRAINING_SEED"] = str(seed)
    if device:
        env["TRAINING_DEVICE"] = device
    env["RUN_TAG"] = tag

    # Limit BLAS/OMP threads per process to prevent mutual preemption
    for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[k] = str(per_threads)

    # Lightweight lower priority (valid for *nix; ignored by Windows)
    pre_cmd = []
    if nice in ("low", "idle") and os.name == "posix":
        level = {"low": "10", "idle": "19"}[nice]
        pre_cmd = ["nice", "-n", level]

    cmd = pre_cmd + [sys.executable, str(script_path)]

    print(f"[LAUNCH] {algo}: {' '.join(cmd)}")
    print(f"[DIR]    {work_dir}")

    # Start child process
    proc = subprocess.Popen(
        cmd,
        cwd=work_dir, # Use an independent working directory for each Algorithm
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0
    )
    t = threading.Thread(target=_pump_to_console_and_file, args=(algo, proc.stdout, logfile), daemon=True)
    t.start()
    return proc, work_dir, artifacts_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER, help="Algorithm to be trained at the same time")
    ap.add_argument("--per-proc-threads", type=int, default=2, help="Number of BLAS/OMP threads per process")
    ap.add_argument("--nice", choices=["normal", "low", "idle"], default="low", help="Process priority (only valid for *nix)")
    ap.add_argument("--total-steps", type=int, default=0, help="Number of overnight steps (passed to subscript through environment variables)")
    ap.add_argument("--seed", type=int, default=-1, help="Training random seeds (passed to subscripts through environment variables)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[], help="Multiple random seeds, for example: --seeds 0 1 2 3 4")
    ap.add_argument("--device", choices=["cpu", "cuda", ""], default="cpu", help="expected device, passed to subscript for reference")
    ap.add_argument("--tag", type=str, default="overnight", help="tag for this run")
    args = ap.parse_args()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{args.tag}"
    base_run_dir = _ensure(PROJECT_ROOT / "runs" / run_id)
    print(f"[RUN] Run ID: {run_id}")
    print(f"[RUN] root directory: {base_run_dir}")

    # 1) Decide which seeds to use
    if args.seeds: # Give priority to --seeds passed in from the command line
        seed_list = args.seeds
    elif args.seed >= 0: # Use a single --seed next
        seed_list = [args.seed]
    else: # If neither is given, use DEFAULT_SEEDS
        seed_list = DEFAULT_SEEDS

    procs = []
    try:
        for sd in seed_list:
            print("\n" + "=" * 80)
            print(f"[SEED] start training seed = {sd}")
            print("=" * 80)

            # Use a subdirectory for each seed: runs/<run_id>/seed_<sd>/
            run_dir = _ensure(base_run_dir / f"seed_{sd}")

            procs.clear()
            for algo in args.algos:
                proc, work_dir, artifacts_dir = launch_one(
                    algo=algo,
                    run_dir=run_dir,
                    per_threads=args.per_proc_threads,
                    nice=args.nice,
                    total_steps=args.total_steps,
                    seed=sd, # ★ Pass the current seed
                    device=args.device,
                    tag=f"{args.tag}_seed{sd}", # ★ RUN_TAG with seed
                )
                procs.append((algo, proc, work_dir, artifacts_dir))

            # Wait for all Algorithms under this seed to end
            exit_codes = {}
            for algo, proc, work_dir, artifacts_dir in procs:
                code = proc.wait()
                exit_codes[algo] = code
                if code == 0:
                    print(f"[{algo}] training is over, exit code={code}, products have been collected to {artifacts_dir}")
                else:
                    print(f"[{algo}] training failed, exit code={code}, please check {work_dir}/logs/train.log")

            failed = [a for a, c in exit_codes.items() if c != 0]
            if failed:
                print(f"[SUMMARY seed={sd}] There are failed processes: ", failed)
            else:
                print(f"[SUMMARY seed={sd}] All training processes ended successfully.")

    except KeyboardInterrupt:
        print("\n[ABORT] Ctrl+C received, trying to terminate all child processes...")
        for _, proc, _, _ in procs:
            with contextlib.suppress(Exception):
                if proc.poll() is None:
                    if os.name == "posix":
                        os.killpg(proc.pid, signal.SIGTERM)
                    else:
                        proc.terminate()
        sys.exit(130)


if __name__ == "__main__":
    import contextlib
    main()
