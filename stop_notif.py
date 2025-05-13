import os
import signal

PID_FILE = "notif_server.pid"

if os.path.exists(PID_FILE):
    with open(PID_FILE, "r") as f:
        pid = int(f.read().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"✅ Notification server (PID {pid}) terminated.")
        os.remove(PID_FILE)
    except ProcessLookupError:
        print("⚠️ Process not found. Maybe it’s already closed?")
        os.remove(PID_FILE)
else:
    print("❌ PID file not found. Is the server running?")