import os
import subprocess
import sys
import time


# Compatibility note: the legacy start_lvgs.py filename and LVGS_* environment
# keys are retained because external launchers and training scripts may rely on them.
def project_python():
    venv_python = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        return venv_python
    if sys.version_info >= (3, 8):
        return sys.executable
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["py", "-3.11", "-c", "import sys; print(sys.executable)"],
                check=True,
                capture_output=True,
                text=True,
            )
            candidate = result.stdout.strip()
            if candidate and os.path.exists(candidate):
                return candidate
        except (OSError, subprocess.CalledProcessError):
            pass
    raise RuntimeError(
        "GridSage requires Python 3.8 or newer. Create .venv or install Python 3.11."
    )


def main():
    print("====================================")
    print(" Starting GridSage (LLM-GridSage-back)...")
    print("====================================")

    python_exe = project_python()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["LVGS_FAST_TRAINING"] = "1"
    env["LVGS_SUPPRESS_WINDOWS_ERROR_DIALOG"] = "1"

    print(f" [1/2] Starting backend API: http://127.0.0.1:8000")
    print(f"       Python: {python_exe}")
    backend_process = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=os.getcwd(),
        env=env,
    )

    time.sleep(2)

    print(" [2/2] Starting frontend Vite: http://localhost:3000")
    frontend_process = subprocess.Popen(
        "npm run dev",
        cwd=os.path.join(os.getcwd(), "frontend"),
        shell=True,
        env=env,
    )

    print("\nStartup complete. Keep this window open and visit http://localhost:3000")

    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\nInterrupt received. Shutting down GridSage...")
        backend_process.terminate()
        frontend_process.terminate()


if __name__ == "__main__":
    main()
