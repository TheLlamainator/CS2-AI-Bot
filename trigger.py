import time
import cv2
import numpy as np
from ultralytics import YOLO
import mss
import threading
from pynput import keyboard
import ctypes
import torch
import random
import win32api
import win32con

MODEL_PATH = 'runs/detect/train/weights/best.pt'
MONITOR = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

model = YOLO(MODEL_PATH)

if torch.cuda.is_available():
    model.model.half()
    device = 0
else:
    device = 'cpu'

class_names = model.names
print("Available classes:")
for idx, name in class_names.items():
    print(f"{idx}: {name}")

selected_class_idx = [None]
class_lock = threading.Lock()

def input_thread():
    global selected_class_idx
    while True:
        new_input = input("[Input] Enter new class name or index: ").strip().lower()
        with class_lock:
            if new_input.isdigit():
                idx = int(new_input)
                if idx in class_names:
                    selected_class_idx[0] = idx
                    print(f"[Input] Target changed to: {class_names[idx]}")
                else:
                    print("[Input] Invalid index.")
            else:
                found = False
                for idx, name in class_names.items():
                    if name.lower() == new_input:
                        selected_class_idx[0] = idx
                        print(f"[Input] Target changed to: {name}")
                        found = True
                        break
                if not found:
                    print("[Input] Invalid class name.")

threading.Thread(target=input_thread, daemon=True).start()

def move_mouse_relative(dx, dy):
    ctypes.windll.user32.mouse_event(0x0001, dx, dy, 0, 0)

def fast_human_move(dx, dy):
    speed_factor = 1.2
    jitter = 0.2
    dx = int(dx * speed_factor + random.uniform(-jitter, jitter))
    dy = int(dy * speed_factor + random.uniform(-jitter, jitter))
    move_mouse_relative(dx, dy)

def is_mb5_pressed():
    return win32api.GetAsyncKeyState(0x06) < 0  # MB5 (forward side button)

ctrl_held = [False]

def on_press(key):
    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
        ctrl_held[0] = True

def on_release(key):
    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
        ctrl_held[0] = False

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

class FrameGrabber(threading.Thread):
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        with mss.mss() as sct:
            while self.running:
                img = np.array(sct.grab(self.monitor))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False

def run_recoil_pattern(modifier=1.0):
    """M4A1-S recoil pattern adapted from AHK."""
    pattern = [
        (1, 6), (0, 4), (-4, 14), (4, 18), (-6, 21), (-4, 24),
        (14, 14), (8, 12), (18, 5), (-4, 10), (-14, 5), (-25, -3),
        (-19, 0), (-22, -3), (1, 3), (8, 3), (-9, 1), (-13, -2),
        (3, 2), (1, 1)
    ]

    for dx, dy in pattern:
        if not is_mb5_pressed():
            break
        move_mouse_relative(int(dx * modifier), int(dy * modifier))
        time.sleep(0.088)

grabber = FrameGrabber(MONITOR)
grabber.start()

locked_target_center = None
lost_frames = 0
max_lost_frames = 5

prev_time = time.time()
frame_count = 0
fps = 0

is_shooting = False

try:
    while True:
        frame = grabber.get_frame()
        if frame is None:
            continue

        frame_count += 1
        now = time.time()
        if now - prev_time >= 1.0:
            fps = frame_count / (now - prev_time)
            frame_count = 0
            prev_time = now

        results_gen = model.predict(
            frame,
            conf=0.2,
            device=device,
            imgsz=640,
            stream=True,
            verbose=False
        )

        try:
            results = next(results_gen)
        except StopIteration:
            continue

        detections = results.boxes

        candidates = []
        with class_lock:
            current_idx = selected_class_idx[0]

        for box in detections:
            cls_id = int(box.cls[0].cpu().numpy())
            if current_idx is not None and cls_id == current_idx:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = int((x1 + x2) / 2)
                box_height = y2 - y1
                cy = int((y1 + y2) / 2 - 0.2 * box_height)
                conf = float(box.conf[0].cpu().numpy())
                candidates.append({'center': (cx, cy), 'conf': conf})
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        if candidates:
            if locked_target_center is not None:
                closest = min(candidates, key=lambda c: np.hypot(c['center'][0] - locked_target_center[0],
                                                                  c['center'][1] - locked_target_center[1]))
                dist = np.hypot(closest['center'][0] - locked_target_center[0],
                                closest['center'][1] - locked_target_center[1])

                if dist < 150:
                    locked_target_center = closest['center']
                    lost_frames = 0
                else:
                    best = max(candidates, key=lambda c: c['conf'])
                    locked_target_center = best['center']
                    lost_frames = 0
            else:
                best = max(candidates, key=lambda c: c['conf'])
                locked_target_center = best['center']
                lost_frames = 0
        else:
            lost_frames += 1
            if lost_frames >= max_lost_frames:
                locked_target_center = None
                lost_frames = 0

        if locked_target_center:
            target_cx, target_cy = locked_target_center
            screen_cx = MONITOR['width'] // 2
            screen_cy = MONITOR['height'] // 2

            dx = int(target_cx - screen_cx)
            dy = int(target_cy - screen_cy)

            if ctrl_held[0]:
                deadzone = 3
                if abs(dx) > deadzone or abs(dy) > deadzone:
                    fast_human_move(dx, dy)

            x_threshold = 35
            y_top_threshold = 25
            y_bottom_threshold = 50

            target_in_crosshair = (abs(dx) < x_threshold and (-y_top_threshold <= dy <= y_bottom_threshold))

            if is_mb5_pressed() and target_in_crosshair:
                if not is_shooting:
                    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)  # Left down
                    is_shooting = True
                    threading.Thread(target=run_recoil_pattern, daemon=True).start()
            else:
                if is_shooting:
                    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)  # Left up
                    is_shooting = False
        else:
            if is_shooting:
                ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
                is_shooting = False

        cv2.putText(frame, f'FPS: {int(fps)}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('Fast Aim + M4A1-S Recoil', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    grabber.stop()
    grabber.join()
    cv2.destroyAllWindows()
    listener.stop()
