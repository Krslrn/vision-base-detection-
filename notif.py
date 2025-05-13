import socket
import tkinter as tk
import screeninfo
import threading
import time
import pystray
from PIL import Image, ImageDraw
import sys

HOST = "0.0.0.0"
PORT = 5005

# Global state
server_thread = None
server_socket = None
server_running = threading.Event()

def show_notification(message="Alert from Raspberry Pi!"):
    def _show():
        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)

        screen = screeninfo.get_monitors()[0]
        screen_width, screen_height = screen.width, screen.height

        width, height = 300, 60
        x = screen_width - width - 10
        y = screen_height - height - 50
        root.geometry(f"{width}x{height}+{x}+{y}")

        # Style based on message content
        if "moving" in message.lower():
            bg_color = "darkred"
            fg_color = "white"
            duration = 5000
            font_style = ("Arial", 13, "bold")
        else:
            bg_color = "black"
            fg_color = "white"
            duration = 3000
            font_style = ("Arial", 12)

        frame = tk.Frame(root, bg=bg_color)
        frame.pack(fill="both", expand=True)

        label = tk.Label(frame, text=message, fg=fg_color, bg=bg_color, font=font_style)
        label.pack(padx=10, pady=10)

        root.after(duration, root.destroy)
        root.mainloop()

    threading.Thread(target=_show, daemon=True).start()

def handle_client(conn, addr):
    try:
        start_time = time.time()
        data = conn.recv(1024).decode()
        if data and server_running.is_set():
            show_notification(data)
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            print(f"\n📨 Received from {addr}: {data}")
            print(f"⚡ Response Time: {response_time_ms:.2f} ms\n")
    except Exception as e:
        print(f"❌ Error handling client {addr}: {e}")
    finally:
        conn.close()

def server_loop():
    global server_socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)
    print(f"✅ Server started. Listening on port {PORT}...\n")

    try:
        while server_running.is_set():
            server_socket.settimeout(1.0)
            try:
                conn, addr = server_socket.accept()
                threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
            except socket.timeout:
                continue
    except Exception as e:
        print(f"❌ Server error: {e}")
    finally:
        server_socket.close()
        server_socket = None
        print("🛑 Server stopped.\n")

def start_server():
    global server_thread
    if not server_running.is_set():
        server_running.set()
        server_thread = threading.Thread(target=server_loop, daemon=True)
        server_thread.start()

def stop_server():
    if server_running.is_set():
        server_running.clear()
        if server_socket:
            try:
                # Dummy connect to unblock accept()
                dummy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                dummy.connect((HOST, PORT))
                dummy.close()
            except:
                pass

def create_image(color: str) -> Image.Image:
    image = Image.new("RGB", (64, 64), color)
    draw = ImageDraw.Draw(image)
    draw.ellipse((16, 16, 48, 48), fill="white")
    return image

def create_tray_icon():
    icon = pystray.Icon("AlertServer")

    def set_online(icon, item):
        icon.icon = create_image("green")
        icon.title = "Status: Online"
        print("🟢 Status set to Online.")
        start_server()

    def set_offline(icon, item):
        icon.icon = create_image("red")
        icon.title = "Status: Offline"
        print("🔴 Status set to Offline.")
        stop_server()

    def on_quit(icon, item):
        print("🚪 Exiting notification server...")
        stop_server()
        icon.stop()
        sys.exit()

    icon.icon = create_image("black")  # Initial tray icon
    icon.menu = pystray.Menu(
        pystray.MenuItem("Green (Online)", set_online),
        pystray.MenuItem("Red (Offline)", set_offline),
        pystray.MenuItem("Exit", on_quit)
    )
    icon.run()

if __name__ == "__main__":
    create_tray_icon()