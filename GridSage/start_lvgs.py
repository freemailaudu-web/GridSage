import os
import subprocess
import sys
import time


def project_python():
    venv_python = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
    return venv_python if os.path.exists(venv_python) else sys.executable


def main():
    print("====================================")
    print(" LVGS (LLM-VGridSim) 一键启动中...")
    print("====================================")

    python_exe = project_python()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["LVGS_FAST_TRAINING"] = "1"
    env["LVGS_SUPPRESS_WINDOWS_ERROR_DIALOG"] = "1"

    print(f" [1/2] 启动后端 API: http://127.0.0.1:8000")
    print(f"       Python: {python_exe}")
    backend_process = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=os.getcwd(),
        env=env,
    )

    time.sleep(2)

    print(" [2/2] 启动前端 Vite: http://localhost:3000")
    frontend_process = subprocess.Popen(
        "npm run dev",
        cwd=os.path.join(os.getcwd(), "frontend"),
        shell=True,
        env=env,
    )

    print("\n启动完成。请保持此窗口打开，访问 http://localhost:3000")

    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\n收到中断指令，正在关闭 LVGS...")
        backend_process.terminate()
        frontend_process.terminate()


if __name__ == "__main__":
    main()
