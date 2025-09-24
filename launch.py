import subprocess
import time
import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 sys.path에 추가하여 src 모듈을 임포트할 수 있도록 합니다.
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# src.config에서 FastAPI 호스트 및 포트 정보 임포트
from src.config import FASTAPI_HOST, FASTAPI_PORT

def run_applications():
    # 현재 실행 중인 Python 인터프리터 경로를 가져옵니다.
    python_executable = sys.executable

    # 환경 변수에 PYTHONIOENCODING=utf-8 추가
    # 이는 subprocess가 실행될 때 Python의 기본 인코딩을 UTF-8로 설정합니다.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # 1. FastAPI 백엔드 (Uvicorn) 실행 명령어
    uvicorn_cmd = [
        python_executable, 
        "-m", "uvicorn", 
        "src.api.main:app", 
        "--host", FASTAPI_HOST, 
        "--port", str(FASTAPI_PORT),
        # "--reload" # 개발용: 코드 변경 시 자동 재시작 (안정성을 위해 임시 제거)
    ]
    print(f"🚀 Starting FastAPI backend: {' '.join(uvicorn_cmd)}")
    # Uvicorn 프로세스를 백그라운드에서 실행합니다. (stdout/stderr는 콘솔로 직접 출력)
    uvicorn_process = subprocess.Popen(uvicorn_cmd, cwd=project_root, env=env)

    # 2. FastAPI 서버가 완전히 시작될 때까지 잠시 기다립니다.
    print("⏳ Waiting for FastAPI backend to start (10 seconds)...") # 대기 시간 10초로 증가
    time.sleep(10) 

    # 3. UI 애플리케이션 실행 명령어
    ui_cmd = [python_executable, "ui/app.py"]
    print(f"🖥️ Starting UI application: {' '.join(ui_cmd)}")
    # UI 프로세스를 실행하고, UI가 종료될 때까지 기다립니다.
    ui_process = subprocess.run(ui_cmd, cwd=project_root, env=env)

    # 4. UI 애플리케이션이 종료되면 FastAPI 백엔드도 종료합니다.
    print("👋 UI application closed. Terminating FastAPI backend...")
    uvicorn_process.terminate() # 프로세스 종료 요청
    try:
        uvicorn_process.wait(timeout=5) # 5초 동안 프로세스 종료 대기
        print("✅ FastAPI backend terminated gracefully.")
    except subprocess.TimeoutExpired:
        print("⚠️ FastAPI backend did not terminate gracefully. Forcing kill...")
        uvicorn_process.kill() # 5초 후에도 종료되지 않으면 강제 종료
        uvicorn_process.wait() # 강제 종료 확인
        print("❌ FastAPI backend forcefully killed.")

if __name__ == "__main__":
    run_applications()
