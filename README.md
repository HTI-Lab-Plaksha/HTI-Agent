# HTI Agent — 3D Conversational AI Assistant
### Human Technology Interaction (HTI) Lab • Plaksha University

An intelligent, multimodal, real-time 3D conversational agent designed for the Human Technology Interaction (HTI) Lab at Plaksha University.

The agent features **Veda**, an interactive 3D persona with natural voice conversation, real-time facial lip-sync, gesture animations, computer vision face detection, and grounded knowledge about Plaksha University and HTI Lab research.

---

## 🌟 Key Features

- **3D Avatar & Gesture Engine**: Powered by Three.js with full 60 FPS blendshape lip-sync (ARKit & Oculus visemes), automatic natural eye blinking, smooth camera presets (Portrait & Full Body), and expressive gesture clips (`Namaste`, `Talking_Explain`, `Talking_Welcome`, `Thinking`, `Idle`).
- **Monochrome Studio Aesthetic**: Premium, modern black-and-white theme (`#0a0a0a` deep black, charcoal cards, pure white highlights, and neutral studio lighting).
- **Multimodal AI Backend**: Built on **Pydantic AI** with type-safe dependency injection, conversation history preservation, and structured tool calling.
- **Fast Multilingual Speech & STT**:
  - **Speech-to-Text**: Low-latency voice transcription via Groq Whisper (`whisper-large-v3-turbo`) with Energy VAD.
  - **Text-to-Speech**: Neural voice synthesis using Edge-TTS (`en-IN-NeerjaNeural` / `hi-IN-SwaraNeural`) with multi-chunk parallel streaming for instant audio playback.
  - **Adaptive Language Matching**: Seamlessly understands and responds in English, Hindi, or Hinglish with natural feminine grammar (*"samajh gayi"*, *"main aa gayi"*).
- **Computer Vision & Face Recognition**: Real-time webcam face detection via OpenCV and `face_recognition` to identify returning faculty/students or greet new visitors.
- **Plaksha Knowledge & RAG**: Deep knowledge base containing profiles of Plaksha founders, trustees, faculty, and HTI Lab research projects, backed by live web search via Tavily.

---

## 📁 Repository Structure

```
.
├── HTI_Avatar_Agent/           # 3D Web interface & avatar assets
│   ├── final/                  # 3D GLB model & gesture animations
│   │   ├── meera.glb           # Production avatar with ARKit/Oculus blendshapes
│   │   └── anims/              # Animation clips (Idle, Namaste, Talking, etc.)
│   ├── web/                    # Client interface
│   │   ├── index.html          # Web application shell
│   │   ├── app.js              # Three.js engine, Web Audio API lip-sync, WebSocket
│   │   └── style.css           # Black & white monochrome theme
│   ├── server.py               # Standalone test server
│   └── README.md
│
├── pydantic_ai_agent/          # Core backend agent & services
│   ├── app.py                  # FastAPI server, WebSockets & vision loops
│   ├── agent.py                # Pydantic AI agent, system prompts & tools
│   ├── tts.py                  # Edge-TTS voice streaming & audio pipelining
│   ├── models.py               # Pydantic models and schemas
│   ├── person_search.py        # Plaksha RAG search & Tavily web enrichment
│   ├── projects.py             # HTI Lab projects lookup tool
│   ├── plaksha_people_dataset.json # Ground truth Plaksha faculty/founder data
│   ├── hti_lab_projects.csv    # HTI Lab project details
│   └── scrape_plaksha.py       # Plaksha scraper utility
│
├── HTI Data/                   # Data assets, CSVs, and posters
│   ├── 1st_images/             # Faculty and founders profile photos
│   ├── 2nd_csv_details/        # Master CSV and individual profile CSVs
│   ├── HTI-LAB-POSTERS/        # Lab research posters
│   ├── hti_lab_projects.csv    # Lab project records
│   ├── plaksha_people_dataset.json
│   └── plaksha_scraper.py
│
├── known_faces.pkl             # Trained facial recognition encodings
├── requirements.txt            # Python dependencies
├── .env.example                # Template for environment variables
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.10+**
- **PortAudio** & **CMake** (for microphone audio and face recognition):
  ```bash
  # macOS (Homebrew)
  brew install portaudio cmake
  
  # Ubuntu / Debian
  sudo apt-get install python3-pyaudio portaudio19-dev cmake
  ```

### 2. Installation
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/HTI-Lab-Plaksha/HTI-Agent.git
cd HTI-Agent

python3 -m venv myenv
source myenv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file from the provided example:
```bash
cp .env.example .env
```
Edit `.env` with your API keys:
```ini
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
DEEPGRAM_API_KEY=your_deepgram_api_key_here
```

### 4. Running the System
Start the integrated agent server:
```bash
PYTHONUNBUFFERED=1 python -m pydantic_ai_agent.app
```

Then open your browser to:
**[http://localhost:8000](http://localhost:8000)**

---

## 🎭 Controls & Interactions

- **Voice Interaction**: Speak naturally into your microphone — the energy VAD will capture your voice, transcribe it with Whisper, and Veda will respond.
- **Chat Bar**: Type text directly into the chat input and click **Speak** or press Enter.
- **Quick Prompts**: Click any prompt chip (e.g., *Namaste Greet*, *HTI Mission*, *Explain Technology*) for instant interactive demos.
- **Camera Controls**: Switch between **Portrait** (head & torso) and **Full Body** views.
- **Studio Modal**: Drag and drop any `.glb` avatar generated with Avaturn or ReadyPlayerMe to instantly test new avatars with full blendshape binding.

---

## 📜 License & Credits

Developed for the **Human Technology Interaction (HTI) Lab** at **Plaksha University**.
