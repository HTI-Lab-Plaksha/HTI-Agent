# Meera — HTI Lab 3D Conversational Avatar Agent
**Human-Machine Interaction • Plaksha University**

An interactive, web-based 3D conversational avatar agent featuring real-time speech synthesis, realistic ARKit/Oculus lip synchronization (jaw, lips, teeth, and tongue), and authentic conversational body gestures (Namaste greeting, explaining, welcoming, thinking, and waving).

---

## 🚀 Quick Start (Instant 1-Click Run)

### Windows
Double-click `start.bat` (or run in PowerShell/Command Prompt):
```bat
start.bat
```
This automatically:
1. Starts the lightweight local Python server on port `5173`.
2. Launches your default web browser to [http://localhost:5173/web/index.html](http://localhost:5173/web/index.html).

### macOS / Linux
Run the shell script:
```bash
chmod +x start.sh
./start.sh
```

### Manual Command
```bash
python server.py
```
Then open [http://localhost:5173/web/index.html](http://localhost:5173/web/index.html) in any modern browser (Chrome, Edge, Brave, Firefox, Safari).

---

## 📦 System Requirements

- **Python 3.8+** (Uses only Python's built-in standard library — **zero `pip install` required!**)
- Any modern web browser with WebGL enabled (Google Chrome or Microsoft Edge recommended for best Web Speech API voices).
- Internet connection on first load (to fetch Three.js and Google Fonts from CDN).

---

## 📁 Package Structure

```
HTI_Avatar_Agent/
├── start.bat                   # 1-Click launcher for Windows
├── start.sh                    # 1-Click launcher for macOS / Linux
├── server.py                   # Lightweight portable HTTP server
├── README.md                   # Setup & usage guide
│
├── web/                        # Web interface files
│   ├── index.html              # Main HTML5 application shell
│   ├── app.js                  # Three.js 3D engine, mixer, speech & lip-sync controller
│   └── style.css               # Modern UI styling & responsive layout
│
└── final/                      # Production 3D avatar & motion assets
    ├── meera.glb               # Production 3D avatar (with 187 ARKit/Oculus blendshapes)
    └── anims/                  # Conversational gesture motion library (.glb)
        ├── Idle.glb            # Natural breathing idle loop
        ├── Namaste.glb         # Traditional Indian Namaste greeting
        ├── Talking.glb         # Natural speech hand movement
        ├── Talking_BothHands.glb # Expressive two-handed gestures
        ├── Talking_Explain.glb # Professional explaining hand gesture
        ├── Talking_Welcome.glb # Warm welcoming gesture
        ├── Thinking.glb        # Contemplative pose
        ├── Waving.glb          # Friendly waving greeting
        └── Thankful.glb        # Gratitude gesture
```

---

## 🌟 Key Features

1. **Realistic 60 FPS Lip Sync**:
   - The avatar (`final/meera.glb`) contains **187 facial morph targets** across the head, teeth, tongue, and eyes.
   - Synchronous articulation of `jawOpen`, `mouthOpen`, `viseme_aa`, `viseme_O`, `mouthSmile`, and organic eye blinking.
   - Teeth and tongue move naturally with the lower jaw during speech.

2. **Conversational Speech Synthesis**:
   - Uses Web Speech API with automatic detection of Indian English female voices (`en-IN`) or standard English.
   - Zero-latency transitions between speaking and idle states.
   - Asymmetric syllable modulation (~3.6 syllables/sec) matching natural human speech cadence.

3. **Authentic Gestures**:
   - **🙏 Namaste**: Traditional Indian folded-hands greeting.
   - **👐 Explain**: Demonstrative hand gestures for teaching or explaining concepts.
   - **🤲 Welcome**: Open-arm welcoming posture for introductions.
   - **Thinking & Waving**: Conversational conversational non-verbal cues.

4. **Camera Controls**:
   - **Portrait**: Close-up camera focused on facial expressions and lip articulation.
   - **Full Body**: Wide view showcasing complete body gestures and posture.
   - **Orbit Controls**: Drag with left mouse button to rotate, right click to pan, scroll to zoom.

5. **Avatar Studio & Hot-Swap**:
   - Click the **✨ Studio** button or drag-and-drop any custom `.glb` avatar into the viewport to instantly load or update the model.

---

## ❓ Troubleshooting

| Issue | Solution |
|---|---|
| **Port 5173 is already in use** | Open `server.py` and change `PORT = 5173` to `5174` (or any free port), then open `http://localhost:5174/web/index.html`. |
| **No audio / avatar mouth not moving** | Click anywhere on the webpage once. Modern browsers require user interaction before playing speech audio. |
| **Old cached avatar shows up** | Press `Ctrl + F5` (Windows) or `Cmd + Shift + R` (Mac) to hard-refresh and clear browser 3D asset cache. |
| **Chrome Web Speech silent** | Ensure your system volume is unmuted and the browser tab is not muted. |

---

## 🏛️ Credits
Developed for the **Human-Machine Interaction (HTI) Lab** at **Plaksha University**.
