import os
import sys
import socket
import subprocess
import threading
import time
import webbrowser
import uvicorn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

def kill_process_on_port(port=8000):
    try:
        output = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode()
        for line in output.strip().splitlines():
            if "LISTENING" in line:
                pid = line.strip().split()[-1]
                if pid and pid != "0":
                    subprocess.call(f"taskkill /F /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(0.5)
    except Exception:
        pass

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    kill_process_on_port(8000)

    print("\n" + "=" * 60)
    print("Customer Feedback AI Application is running!")
    print("Open in browser: http://127.0.0.1:8000")
    print("Press CTRL+C to stop the server.")
    print("=" * 60 + "\n")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
