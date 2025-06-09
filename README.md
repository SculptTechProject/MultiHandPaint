# 🎨 MultiHandPaint – Draw in the Air with Hand Gestures & Face Detection

**AirPaint** is a fun and interactive Python application that lets you draw in the air using hand gestures detected by [MediaPipe](https://google.github.io/mediapipe/). You can switch between brush, eraser, and color modes using your **left hand**, draw with your **right hand**, and even **clear the canvas by opening your mouth** 👄!

---

---

## ✨ Features

- ✋ Detects left & right hands independently
- 🖌️ Draw mode, eraser mode, and color switching via gestures
- 💋 Mouth open = clear canvas (using MediaPipe FaceMesh)
- 🎨 Live camera feed + overlay with drawn strokes
- ⌨️ Keyboard shortcuts:
  - `c` → Clear canvas
  - `q` → Quit app

---

## 🧠 How it works

| Gesture                          |         Action           |
|----------------------------------|--------------------------|
| ✋ Left hand: 5 fingers up      | Brush mode               |
| ✊ Left hand: 0 fingers         | Eraser mode              |
| ✌️ Left hand: 2 fingers + pinch | Change color             |
| 🤚 Right hand (index tip)       | Draws or erases          |
| 👄 Mouth open                   | Clear canvas (Buziak!)   |

---

## 🛠 Tech Stack

- **Python**
- **OpenCV**
- **MediaPipe** (Hands + FaceMesh)
- **NumPy**

---

## 🚀 How to run *(Try!)*

```bash
git clone https://github.com/yourusername/AirPaint.git
cd AirPaint

# Create venv (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Have fun! Run the app!
python main.py
