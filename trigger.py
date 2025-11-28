import time
import cv2
import numpy as np
from ultralytics import YOLO
import mss
import threading
from pynput import keyboard
import ctypes
import torch
import win32api
import win32gui
import win32con

# === [ Resolution Config ] ===
custom_width = input("Enter screen capture width (default 1920): ").strip()
custom_height = input("Enter screen capture height (default 1080): ").strip()

try:
    width = int(custom_width) if custom_width else 1920
    height = int(custom_height) if custom_height else 1080
except ValueError:
    print("Invalid input, using default resolution (1920x1080).")
    width, height = 1920, 1080

MONITOR = {'top': 0, 'left': 0, 'width': width, 'height': height}

# === [ Sensitivity Config ] ===
try:
    user_sens = float(input("Enter your in-game sensitivity (e.g. 0.6): ").strip())
    if user_sens <= 0:
        raise ValueError
except ValueError:
    print("Invalid sensitivity, defaulting to 1.0")
    user_sens = 1.0

sens_multiplier = round(1.0 / user_sens * 1.3, 2)
print(f"[Config] Sensitivity set to {user_sens}, movement multiplier = {sens_multiplier}")

# === [ Model ] ===
MODEL_PATH = 'runs/detect/train/weights/best.pt'
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

def snap_to_target(dx, dy):
    dx = int(dx * sens_multiplier)
    dy = int(dy * sens_multiplier)
    move_mouse_relative(dx, dy)

def is_mb5_pressed():
    return win32api.GetAsyncKeyState(0x06) < 0

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
                img = sct.grab(self.monitor)
                frame = np.asarray(img, dtype=np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False

def run_recoil_pattern(modifier=1.0):
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

# === [ Overlay Window Setup ] ===
def make_overlay_window(window_name, width, height):
    """Create a transparent, click-through overlay window"""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.moveWindow(window_name, 0, 0)
    
    # Get window handle
    hwnd = win32gui.FindWindow(None, window_name)
    
    if hwnd:
        # Make window layered (required for transparency)
        win32gui.SetWindowLong(
            hwnd,
            win32con.GWL_EXSTYLE,
            win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOPMOST
        )
        
        # Set window to be always on top and transparent
        win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0, 0, 0), 0, win32con.LWA_COLORKEY)
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, width, height, 0)
        
        # Remove window borders and make it click-through
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
        style &= ~(win32con.WS_CAPTION | win32con.WS_THICKFRAME | win32con.WS_MINIMIZEBOX | win32con.WS_MAXIMIZEBOX | win32con.WS_SYSMENU)
        win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)
        
        print("[Overlay] Window configured as transparent overlay")
    else:
        print("[Warning] Could not find window handle for overlay")

grabber = FrameGrabber(MONITOR)
grabber.start()

locked_target_center = None
lost_frames = 0
max_lost_frames = 5

prev_time = time.time()
frame_count = 0
fps = 0
fps_update_time = time.time()

is_shooting = False

# Create overlay window
window_name = 'Aimbot Overlay'
make_overlay_window(window_name, width, height)

# Pre-calculate screen center for performance
screen_cx = width >> 1
screen_cy = height >> 1

try:
    while True:
        frame = grabber.get_frame()
        if frame is None:
            continue

        # Update FPS counter
        frame_count += 1
        now = time.time()
        if now - fps_update_time >= 1.0:
            fps = frame_count / (now - fps_update_time)
            frame_count = 0
            fps_update_time = now

        results_gen = model.predict(
            frame,
            conf=0.3,
            device=device,
            imgsz=320,
            stream=True,
            verbose=False
        )

        try:
            results = next(results_gen)
        except StopIteration:
            continue

        detections = results.boxes

        # Create transparent overlay (black background will be made transparent)
        overlay = np.zeros((height, width, 3), dtype=np.uint8)

        candidates = []
        with class_lock:
            current_idx = selected_class_idx[0]

        # Batch extract all box data at once for speed
        if len(detections) > 0:
            boxes_cpu = detections.xyxy.cpu().numpy()
            cls_cpu = detections.cls.cpu().numpy()
            conf_cpu = detections.conf.cpu().numpy()
            
            for i, (box, cls_id, conf) in enumerate(zip(boxes_cpu, cls_cpu, conf_cpu)):
                if current_idx is not None and int(cls_id) == current_idx:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cx = (x1 + x2) >> 1
                    box_height = y2 - y1
                    cy = int((y1 + y2) / 2 - 0.4 * box_height)
                    candidates.append({'center': (cx, cy), 'conf': float(conf)})
                    
                    # Thinner lines for faster rendering
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.circle(overlay, (cx, cy), 3, (0, 0, 255), -1)
                    cv2.putText(overlay, f'{conf:.2f}', (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        if candidates:
            # Find target closest to screen center using squared distance (faster, no sqrt)
            screen_cx = MONITOR['width'] >> 1
            screen_cy = MONITOR['height'] >> 1
            best = min(candidates, key=lambda c: (c['center'][0] - screen_cx)**2 + (c['center'][1] - screen_cy)**2)
            locked_target_center = best['center']
            lost_frames = 0
        else:
            lost_frames += 1
            if lost_frames >= max_lost_frames:
                locked_target_center = None
                lost_frames = 0

        if locked_target_center:
            target_cx, target_cy = locked_target_center

            dx = target_cx - screen_cx
            dy = target_cy - screen_cy

            if ctrl_held[0]:
                snap_to_target(dx, dy)

            x_threshold = 35
            y_top_threshold = 25
            y_bottom_threshold = 50

            target_in_crosshair = (abs(dx) < x_threshold and (-y_top_threshold <= dy <= y_bottom_threshold))

            if is_mb5_pressed() and target_in_crosshair:
                if not is_shooting:
                    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
                    is_shooting = True
                    threading.Thread(target=run_recoil_pattern, daemon=True).start()
            else:
                if is_shooting:
                    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
                    is_shooting = False
        else:
            if is_shooting:
                ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
                is_shooting = False

        # Draw FPS and status on overlay
        cv2.putText(overlay, f'FPS: {int(fps)}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        status_text = "SHOOTING" if is_shooting else "READY"
        status_color = (0, 0, 255) if is_shooting else (0, 255, 0)
        cv2.putText(overlay, status_text, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        
        if current_idx is not None:
            cv2.putText(overlay, f'Target: {class_names[current_idx]}', (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Display transparent overlay with minimal wait time
        cv2.imshow(window_name, overlay)
        
        # Handle ESC key with minimal delay for faster loop
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

finally:
    grabber.stop()
    grabber.join()
    cv2.destroyAllWindows()
    listener.stop()
    if is_shooting:
        ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
    print("[Exit] Cleanup complete")
