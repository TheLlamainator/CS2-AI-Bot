# CS2-AI-Bot
This project is a real-time YOLOv8-based triggerbot for CS2 (Counter-Strike 2) with an integrated recoil pattern. It uses computer vision to automatically detect enemies and fires only when your crosshair is aligned, simulating human reaction with configurable aim assist and recoil compensation.

Features
Real-time YOLOv8 object detection on-screen

Triggerbot fires only when the crosshair is aligned

M4A1-S inspired recoil control pattern

Hold Mouse Button 5 (Forward button) to activate triggerbot

Hold CTRL for human-like fast aim assist (not the best right now)

Live class switching through console input (no restart needed)

Randomized aim jitter for humanization

Works with custom-trained YOLOv8 models

Requirements
Windows 10/11

Python 3.9+

NVIDIA GPU recommended (CUDA support)

# Installation
Install dependencies: 
pip install -r requirements.txt

Use my pre-trained cs2 yolov8 model or train your own model and put it in  (must be called best.pt):
runs/detect/train/weights/

# Run the bot:
python triggerbot.py

# Controls:
Mouse Button 5 (held) = activate trigger bot
CTRL (held) = activate aim bot
Change target Class: type it in console terror is for terrorist and counter is for ct's

# Disclaimer
This project is licensed under the GNU GPLv3 to ensure it remains open source forever.
Use at your own risk.
The author is not liable for any misuse, bans, or damages caused by this software.
This project is for educational purposes only.
