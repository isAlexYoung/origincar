import os
import time
import threading
import queue
import requests
import signal
import tkinter as tk
import tkinter.font as tkfont

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


PC_URL = "http://192.168.144.30:8000/describe" # 229_WIFI1
# PC_URL = "http://10.122.97.91:8000/describe" # OPPO Reno
# PC_URL = "http://192.168.43.11:8000/describe" # Young_999

WATCH_DIR = "/userdata/lrj_test_mppi/voice/"
VALID_EXT = (".jpg", ".jpeg", ".png")

# 屏幕分辨率（4.3寸常见 800x480）
SCREEN_W = 800
SCREEN_H = 480

# 线程安全队列：所有要显示的内容先丢这里，由主线程取出更新屏幕
display_queue = queue.Queue()
shutdown_flag = threading.Event()

# ============== 图生文部分 ==============

def describe_image(image_path):
    with open(image_path, "rb") as f:
        r = requests.post(PC_URL, files={"file": f}, timeout=30)
    return r.json()["text"]


def wait_until_stable(path, check_interval=0.3, stable_rounds=2):
    last_size = -1
    stable_count = 0
    while stable_count < stable_rounds:
        try:
            size = os.path.getsize(path)
        except FileNotFoundError:
            return
        if size == last_size and size > 0:
            stable_count += 1
        else:
            stable_count = 0
            last_size = size
        time.sleep(check_interval)


def handle_image(image_path):
    print(f"[检测到新图片] {image_path}")
    wait_until_stable(image_path)
    try:
        text = describe_image(image_path)
        print("描述:", text)
        # 丢进队列，让主线程更新屏幕第二行
        display_queue.put({"type": "desc", "text": text})
    except Exception as e:
        print("图生文处理失败:", e)
        display_queue.put({"type": "desc", "text": "图像识别失败"})


class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.lower().endswith(VALID_EXT):
            handle_image(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return
        if event.dest_path.lower().endswith(VALID_EXT):
            handle_image(event.dest_path)


# ============== 二维码部分 ==============

class QRCodeDisplayNode(Node):
    def __init__(self):
        super().__init__('qrcode_display_node')
        self.qr_shown = False  # 只显示一次
        self.sub = self.create_subscription(
            String, '/qrcode_information', self.qr_callback, 10
        )
        self.get_logger().info("二维码显示节点已启动，监听 /qrcode_information")

    def qr_callback(self, msg):
        if self.qr_shown:
            return

        raw = msg.data.strip()
        digits = ''.join(filter(str.isdigit, raw))
        if not digits:
            self.get_logger().warn(f"二维码内容无数字: {raw}")
            return

        num = int(digits)
        direction = "顺时针" if num % 2 else "逆时针"
        text = f"数字：{num}，{direction}"

        self.get_logger().info(f"[二维码] 显示: {text}")
        # 丢进队列，让主线程更新屏幕第一行
        display_queue.put({"type": "qr", "text": text})
        self.qr_shown = True


# ============== 屏幕显示部分 ==============

class DisplayApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RDK Display")
        self.root.configure(bg="black")
        # 全屏显示
        self.root.attributes("-fullscreen", True)
        self.root.geometry(f"{SCREEN_W}x{SCREEN_H}")
        # 按 Esc 退出全屏（调试用）
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
        self.root.bind("q", lambda e: self.root.destroy())

        # 第一行：二维码内容，大字、醒目色
        self.qr_font = tkfont.Font(size=60, weight="bold")
        self.qr_label = tk.Label(
            self.root, text="等待二维码...", font=self.qr_font,
            fg="#00FF00", bg="black", anchor="center"
        )
        self.qr_label.pack(side="top", fill="x", pady=(20, 10))

        # 分隔线
        sep = tk.Frame(self.root, height=2, bg="#444444")
        sep.pack(fill="x", padx=20)

        # 剩余区域：图生文内容，自动换行
        self.desc_font = tkfont.Font(size=44, weight="bold")
        self.desc_label = tk.Label(
            self.root, text="等待图像识别...", font=self.desc_font,
            fg="#0C22E7", bg="black", anchor="n", justify="center",
            wraplength=SCREEN_W - 40  # 超过宽度自动换行
        )
        self.desc_label.pack(side="top", fill="both", expand=True, padx=20, pady=10)

    def poll_queue(self):
    # 先看是不是要退出了
        if shutdown_flag.is_set():
            self.root.quit()      # 让 mainloop 返回
            return
        try:
            while True:
                msg = display_queue.get_nowait()
                if msg["type"] == "qr":
                    self.qr_label.config(text=msg["text"])
                elif msg["type"] == "desc":
                    self.desc_label.config(text=msg["text"])
        except queue.Empty:
            pass
        self.root.after(100, self.poll_queue)

    def run(self):
        self.root.after(100, self.poll_queue)
        self.root.mainloop()


# ============== 主程序 ==============

def ros_spin(node):
    try:
        while rclpy.ok() and not shutdown_flag.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except Exception as e:
        print("ros spin 退出:", e)


def main():
    os.makedirs(WATCH_DIR, exist_ok=True)

    # 注册 Ctrl+C 处理：按下就置退出标志
    def on_sigint(signum, frame):
        print("\n收到退出信号，正在关闭...")
        shutdown_flag.set()
    signal.signal(signal.SIGINT, on_sigint)

    # 启动文件监听
    print(f"[监听中] {WATCH_DIR}")
    observer = Observer()
    observer.schedule(ImageHandler(), WATCH_DIR, recursive=False)
    observer.start()

    # 启动 ROS2，spin 放后台线程
    rclpy.init()
    node = QRCodeDisplayNode()
    spin_thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
    spin_thread.start()

    # GUI 主线程
    app = DisplayApp()
    try:
        app.run()
    finally:
        # 走到这说明 mainloop 已结束，开始收尾
        shutdown_flag.set()
        observer.stop()
        observer.join(timeout=2)        # 加超时，最多等 2 秒
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
        print("已退出")


if __name__ == "__main__":
    main()