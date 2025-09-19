import subprocess
import os
import time
import webview
import threading
import sys

streamlit_process = None

def run_streamlit():
    """Runs the Streamlit app in a subprocess."""
    global streamlit_process
    project_root = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(project_root, "ui", "app.py")
    command = ["streamlit", "run", app_path, "--server.headless", "true"]
    
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW

    streamlit_process = subprocess.Popen(command, cwd=project_root, creationflags=creationflags)

def kill_streamlit():
    """Terminates the Streamlit server process."""
    global streamlit_process
    if streamlit_process:
        print("Terminating Streamlit server...")
        if sys.platform == "win32":
            # Use taskkill on Windows to forcefully terminate the process tree
            # Redirect output to DEVNULL to suppress success messages
            subprocess.call(
                ['taskkill', '/F', '/T', '/PID', str(streamlit_process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            import signal
            os.killpg(os.getpgid(streamlit_process.pid), signal.SIGTERM)
        streamlit_process = None
        print("Streamlit server terminated.")

if __name__ == '__main__':
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    streamlit_thread.start()

    print("Starting Streamlit server, please wait...")
    time.sleep(8)

    print("Opening application window.")
    webview.create_window(
        'InfoPilot',
        'http://localhost:8501',
        width=1280,
        height=800
    )
    
    try:
        webview.start()
    finally:
        kill_streamlit()
