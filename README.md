# 🪄 Gesture Invisibility (VanishCam)

A real-time, AI-powered browser application that turns your webcam into a gesture-controlled invisibility cloak. Powered by Google MediaPipe vision models, it detects hand gestures and segmentates your body live in the browser with zero server latency.

🚀 **[Live Demo](https://mahdihoggas.github.io/Gesture-Invisibility/)**

---

## ✨ Features

- **✊ Gesture-Controlled Invisibility**: Close your fist to vanish instantly; open your palm to reappear.
- **🧠 100% In-Browser AI**: Uses Google MediaPipe **Hand Landmarker** & **Selfie Segmenter** running entirely on client-side WebGL/GPU.
- **📦 Body Tracking HUD**: Interactive tracking bounding box that auto-fits your body height & width (Yellow when visible, Purple when invisible).
- **🖼️ Custom Background Capture**: Upload a photo of your empty room or snap one live from the camera.
- **🔒 Passcode Protected**: Features a glassmorphic security overlay screen requiring passcode (`1609`) to access.
- **⚡ Zero Server Dependencies**: Pure HTML5, Vanilla CSS3, and ES Module JavaScript — ultra fast and lightweight.

---

## ✋ Gesture Controls

| Gesture | Action | Visual Status |
| :--- | :--- | :--- |
| **✊ Closed Fist** | **Vanish** (Replace body with background) | Purple Dashed Bounding Box |
| **✋ Open Palm** | **Reappear** (Show live video feed) | Solid Yellow Bounding Box |

---

## 🛠️ Tech Stack

- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism, CSS Grid/Flexbox, Keyframe Animations)
- **Logic**: Vanilla JavaScript (ES6 Modules)
- **Computer Vision & AI**:
  - `@mediapipe/tasks-vision` (Hand Landmarker & Selfie Segmenter)
- **Hosting**: GitHub Pages

---

## 📂 Project Structure

```text
Gesture-Invisibility/
├── css/
│   └── styles.css       # Design tokens, HUD overlay styles & glassmorphism
├── app.js               # MediaPipe ML initialization, frame rendering & gesture detection
├── index.html           # Main HTML structure & passcode gate
└── README.md            # Project documentation
```

---

## 🚀 Getting Started Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mahdihoggas/Gesture-Invisibility.git
   ```

2. **Open the project**:
   Open `index.html` in any modern web browser (Chrome, Edge, Brave, Firefox, or Safari).

3. **Passcode Access**:
   Enter passcode **`1609`** to unlock the application.

4. **Allow Camera Permissions**:
   Grant webcam access when prompted, capture your empty room background, and enjoy the magic!

---

## 👤 Author

**Mahdi Hoggas**
- GitHub: [@mahdihoggas](https://github.com/mahdihoggas)
- Portfolio: [mahdihoggas.github.io/mahdi-portfolio](https://mahdihoggas.github.io/mahdi-portfolio/)

---

⭐ *If you enjoyed this project, feel free to give it a star on GitHub!*
