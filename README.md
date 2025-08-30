# CS2-AI-Bot
This project is a real-time YOLOv8-based aim + triggerbot for CS2 (Counter-Strike 2) with an integrated recoil pattern. It uses computer vision to automatically detect enemies and fires only when your crosshair is aligned, simulating human reaction with configurable aim assist and recoil compensation.

# Features
- Real-time YOLOv8 object detection on-screen
- Triggerbot fires only when the crosshair is aligned
- Critically-damped aim assist with sub-pixel precision (fast lock-on, smooth tracking)
- Multiple aim profiles: Smooth, Balanced, Super Aggressive
- M4A1-S inspired recoil control pattern
- Hold Mouse Button 5 (Forward button) to activate triggerbot
- Hold CTRL for aim assist (locks directly onto target)
- Live class switching through console input (no restart needed)
- Optional randomized aim jitter for humanization
- Works with custom-trained YOLOv8 models

# Requirements
- Windows 10/11
- Python 3.9+
- NVIDIA GPU recommended (CUDA support)

# Installation

Install dependencies:

```bash
pip install -r requirements.txt
````

Use the pre-trained CS2 YOLOv8 model or train your own model and put it in (must be called best.pt):

```bash
runs/detect/train/weights/
```

# Run the bot:

```bash
python triggerbot.py
```

# Controls:

```text
Mouse Button 5 (held) = activate trigger bot
CTRL (held)           = activate aim bot
Change target Class   = type it in console ("terror" for Terrorist, "counter" for Counter-Terrorist)
```

# Disclaimer

This project is licensed under the GNU GPLv3 to ensure it remains open source forever.
Use at your own risk.
The author is not liable for any misuse, bans, or damages caused by this software.
This project is for educational purposes only.
