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
import math

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
    user_sens = float(input("Enter your in-game sensitivity (e.g. 0.4): ").strip())
    if user_sens <= 0:
        raise ValueError
except ValueError:
    print("Invalid sensitivity, defaulting to 1.0")
    user_sens = 1.0

# Increased multiplier for snappier movement
sens_multiplier = round(1.0 / user_sens * 2.0, 2)
print(f"[Config] Sensitivity set to {user_sens}, movement multiplier = {sens_multiplier}")

# === [ Model ] ===
MODEL_PATH = 'runs/detect/train/weights/best.pt'
model = YOLO(MODEL_PATH)

if torch.cuda.is_available():
    try:
        model.model.half()
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True
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

# === [ Sub-pixel applier to avoid rounding-induced wobble ] ===
class MouseApplier:
    def __init__(self):
        self.rx = 0.0
        self.ry = 0.0
    def move(self, dx_float, dy_float):
        self.rx += dx_float
        self.ry += dy_float
        ix = int(round(self.rx))
        iy = int(round(self.ry))
        if ix or iy:
            move_mouse_relative(ix, iy)
            self.rx -= ix
            self.ry -= iy

mouse_applier = MouseApplier()

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

# === [ Predictor thread: feed YOLO many more frames ] ===
class Predictor(threading.Thread):
    def __init__(self, grabber):
        super().__init__()
        self.grabber = grabber
        self.lock = threading.Lock()
        self.latest_boxes = None
        self.running = True
    def run(self):
        while self.running:
            frame = self.grabber.get_frame()
            if frame is None:
                continue
            res = model.predict(
                frame, conf=0.2, device=device, imgsz=640,
                stream=False, verbose=False
            )[0]
            with self.lock:
                self.latest_boxes = res.boxes
    def get_boxes(self):
        with self.lock:
            return self.latest_boxes
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

# === [ Locking stack: smoother + aggressively tuned controller ] ===
class TargetSmoother:
    """Snappy EMA with tiny look-ahead; bypass when jumpy to avoid lag."""
    def __init__(self, bypass_dist=80, lookahead=0.01):
        self.bypass_dist = bypass_dist
        self.lookahead = lookahead
        self.pos = None
        self.vel = np.zeros(2, dtype=np.float32)
        self.t_prev = time.time()
    def reset(self):
        self.pos = None
        self.vel[:] = 0
        self.t_prev = time.time()
    def update(self, mxy):
        now = time.time()
        dt = max(1e-3, now - self.t_prev)
        self.t_prev = now
        m = np.array(mxy, dtype=np.float32)
        if self.pos is None:
            self.pos = m.copy(); self.vel[:] = 0
        else:
            if np.linalg.norm(m - self.pos) > self.bypass_dist:
                self.pos = m.copy(); self.vel[:] = 0
            else:
                inst_vel = (m - self.pos) / dt
                self.vel = 0.5 * self.vel + 0.5 * inst_vel
                m_pred = m + self.lookahead * self.vel
                self.pos = 0.6 * self.pos + 0.4 * m_pred
        return int(self.pos[0]), int(self.pos[1])

class CriticallyDampedController(threading.Thread):
    """
    Runs at high rate and moves the mouse smoothly toward the smoothed center
    with aggressive gains and caps for super fast lock.
    """
    def __init__(self, get_target_fn, sens_mul, hz=400):
        super().__init__()
        self.get_target = get_target_fn
        self.hz = hz
        self.period = 1.0 / hz
        self.running = True
        # >>> Aggressive tuning <<<
        self.k = 12.0           # proportional gain (hard snap)
        self.kd = 0.18          # lighter damping for speed
        self.vmax_far = 4800.0  # px/s when far
        self.vmax_near = 1200.0 # px/s near target
        self.accel_limit = 22000.0 # px/s^2
        self.capture_px = 1.2   # slightly looser capture to stick sooner
        self.prev_err = np.zeros(2, dtype=np.float32)
        self.prev_vel = np.zeros(2, dtype=np.float32)
        self.t_prev = time.time()
        self.sens_mul = sens_mul
    def reset(self):
        self.prev_err[:] = 0
        self.prev_vel[:] = 0
        self.t_prev = time.time()
    def _speed_limit(self, dist):
        s = 1.0 - math.exp(-dist / 160.0)
        return self.vmax_near + (self.vmax_far - self.vmax_near) * s
    def step_once(self, err_xy):
        now = time.time()
        dt = max(1e-4, now - self.t_prev); self.t_prev = now
        e = np.array(err_xy, dtype=np.float32)
        dist = float(np.linalg.norm(e))
        if dist <= self.capture_px:
            mouse_applier.move(e[0]*self.sens_mul, e[1]*self.sens_mul)
            self.reset()
            return
        vmax = self._speed_limit(dist)
        de_dt = (e - self.prev_err) / dt
        v_des = self.k * e - self.kd * de_dt
        # cap speed
        mag = float(np.linalg.norm(v_des))
        if mag > 1e-6:
            v_des *= (min(mag, vmax) / mag)
        # accel clamp
        a = (v_des - self.prev_vel) / dt
        a_mag = float(np.linalg.norm(a))
        if a_mag > self.accel_limit:
            a *= (self.accel_limit / a_mag)
            v_cmd = self.prev_vel + a * dt
        else:
            v_cmd = v_des
        delta = v_cmd * dt
        # directional clamp (no overshoot)
        if dist > 0:
            e_dir = e / dist
            along = float(np.dot(delta, e_dir))
            if along > dist:
                delta = e_dir * dist
        self.prev_err = e
        self.prev_vel = v_cmd
        mouse_applier.move(delta[0]*self.sens_mul, delta[1]*self.sens_mul)
    def run(self):
        while self.running:
            t0 = time.time()
            target = self.get_target()
            if target is not None and ctrl_held[0]:
                scx = MONITOR['width'] // 2
                scy = MONITOR['height'] // 2
                err_x = target[0] - scx
                err_y = target[1] - scy
                self.step_once((err_x, err_y))
            else:
                self.reset()
            dt = time.time() - t0
            to_sleep = self.period - dt
            if to_sleep > 0:
                time.sleep(to_sleep)
    def stop(self):
        self.running = False

# === [ Start capture + predictor ] ===
grabber = FrameGrabber(MONITOR)
grabber.start()

predictor = Predictor(grabber)
predictor.start()

# === [ State ] ===
locked_target_center = None
lost_frames = 0
max_lost_frames = 5

prev_time = time.time()
frame_count = 0
fps = 0

is_shooting = False

# smoother + controller (aggressive smoother settings)
smoother = TargetSmoother(bypass_dist=80, lookahead=0.01)

# function for controller to read the smoothed center
_controller_center_lock = threading.Lock()
_controller_center = [None]
def set_controller_center(pt):
    with _controller_center_lock:
        _controller_center[0] = pt
def get_controller_center():
    with _controller_center_lock:
        return _controller_center[0]

controller = CriticallyDampedController(
    get_target_fn=get_controller_center,
    sens_mul=sens_multiplier,
    hz=400
)
controller.start()

try:
    while True:
        frame = grabber.get_frame()
        if frame is None:
            continue

        # FPS counter
        frame_count += 1
        now = time.time()
        if now - prev_time >= 1.0:
            fps = frame_count / (now - prev_time)
            frame_count = 0
            prev_time = now

        detections = predictor.get_boxes()
        if detections is None:
            continue

        candidates = []
        with class_lock:
            current_idx = selected_class_idx[0]

        for box in detections:
            cls_id = int(box.cls[0].detach().cpu().numpy())
            if current_idx is not None and cls_id == current_idx:
                x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()
                cx = int((x1 + x2) / 2)
                box_height = y2 - y1
                # slight head bias
                cy = int((y1 + y2) / 2 - 0.12 * box_height)
                conf = float(box.conf[0].detach().cpu().numpy())
                candidates.append({'center': (cx, cy), 'conf': conf})

                # (Optional) draw — comment out for perf
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Selection logic (sticky to closest-of-previous)
        if candidates:
            if locked_target_center is not None:
                closest = min(
                    candidates,
                    key=lambda c: np.hypot(
                        c['center'][0] - locked_target_center[0],
                        c['center'][1] - locked_target_center[1]
                    )
                )
                dist = np.hypot(
                    closest['center'][0] - locked_target_center[0],
                    closest['center'][1] - locked_target_center[1]
                )
                if dist < 150:
                    locked_target_center = closest['center']
                else:
                    locked_target_center = max(candidates, key=lambda c: c['conf'])['center']
            else:
                locked_target_center = max(candidates, key=lambda c: c['conf'])['center']
            lost_frames = 0
        else:
            lost_frames += 1
            if lost_frames >= max_lost_frames:
                locked_target_center = None
                lost_frames = 0
                smoother.reset()
                controller.reset()

        # Feed smoothed center to the controller
        if locked_target_center:
            sm_cx, sm_cy = smoother.update(locked_target_center)
            set_controller_center((sm_cx, sm_cy))
        else:
            set_controller_center(None)

        # Trigger logic (unchanged: MB5 + recoil when inside window)
        if locked_target_center:
            screen_cx = MONITOR['width'] // 2
            screen_cy = MONITOR['height'] // 2
            dx = int(locked_target_center[0] - screen_cx)
            dy = int(locked_target_center[1] - screen_cy)

            x_threshold = 35
            y_top_threshold = 25
            y_bottom_threshold = 50
            target_in_crosshair = (
                abs(dx) < x_threshold and (-y_top_threshold <= dy <= y_bottom_threshold)
            )

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

        cv2.imshow('Fast Aim + Lock Controller (Super Aggressive)', frame)  # comment for perf
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    controller.stop()
    predictor.stop()
    grabber.stop()
    controller.join()
    predictor.join()
    grabber.join()
    cv2.destroyAllWindows()
    listener.stop()
