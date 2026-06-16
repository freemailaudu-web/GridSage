import os
import subprocess
import sys
import time


# Compatibility note: the legacy start_lvgs.py filename and LVGS_* environment
# keys are retained because external launchers and training scripts may rely on them.
def _python_usable(python_exe):
    try:
        result = subprocess.run(
            [
                python_exe,
                "-c",
                (
                    "import sys; "
                    "assert sys.version_info >= (3, 8); "
                    "import fastapi, uvicorn, pydantic"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except OSError:
        return False


def _py_launcher_python(version):
    if os.name != "nt":
        return None
    try:
        result = subprocess.run(
            ["py", f"-{version}", "-c", "import sys; print(sys.executable)"],
            check=True,
            capture_output=True,
            text=True,
        )
        candidate = result.stdout.strip()
        return candidate if candidate and os.path.exists(candidate) else None
    except (OSError, subprocess.CalledProcessError):
        return None


def project_python():
    venv_python = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_python) and _python_usable(venv_python):
        return venv_python
    if sys.version_info >= (3, 8) and _python_usable(sys.executable):
        return sys.executable
    for version in ("3.11", "3.10", "3.9", "3.8"):
        candidate = _py_launcher_python(version)
        if candidate and _python_usable(candidate):
            return candidate
    raise RuntimeError(
        "GridSage requires Python 3.8+ with FastAPI/Uvicorn installed. Create .venv or install backend requirements."
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
