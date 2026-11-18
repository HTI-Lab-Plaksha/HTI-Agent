# HTI Lab Interactive Agent ("Veda") - Pydantic AI Edition

This directory contains the implementation of the HTI Lab Agent (**Veda**) using the **Pydantic AI** and **Pydantic** framework, preserving the exact same pipelines for face recognition, microphone energy-based VAD, Groq Whisper STT, Edge-TTS playback, and WebSocket dashboard.

---

## 2-Part Intro Flow (Zero Latency)

1. **Instant Greeting via Local Plaksha Dataset**:
   - As soon as a face is confirmed against `known_faces.pkl`, Veda matches the person against `plaksha_people_dataset.json` (scraped directly from `plaksha.edu.in`).
   - The local profile (role, education, research interests, bio) is retrieved instantly (0ms latency).
   - Veda's dedicated `greeting_agent` synthesizes a warm, witty Hinglish greeting and begins audio playback immediately.
2. **Background Web Search (Tavily)**:
   - Concurrently with greeting audio playback, an asynchronous background task queries Tavily for real-time web insights about the visitor.
   - Once completed, the search findings seamlessly enrich `session_context` for subsequent conversation turns without causing any initial greeting delay.

---

## Architecture & Module Structure

| File | Purpose |
|---|---|
| [`models.py`](models.py) | Pydantic `BaseModel` schemas for application config (`AppConfig`), visitor profile (`PersonProfile`), scraped Plaksha profiles (`PlakshaPerson`), Tavily results (`SearchResultItem`), injected dependencies (`AgentDeps`), and WebSocket telemetry events (`WebSocketEvent`). |
| [`agent.py`](agent.py) | Core conversational (`veda_agent`) and greeting (`greeting_agent`) agents defined using `pydantic_ai.Agent(GroqModel("openai/gpt-oss-20b"))`. Contains dynamic system prompts and real-time search tools. |
| [`person_search.py`](person_search.py) | Local Plaksha dataset querying (`find_plaksha_person`) + asynchronous Tavily web search and profile synthesis (`search_and_summarize_async`). |
| [`plaksha_people_dataset.json`](plaksha_people_dataset.json) | Comprehensive clean profile dataset for 42 faculty, founders, and trustees scraped directly from `plaksha.edu.in`. |
| [`scrape_plaksha.py`](scrape_plaksha.py) | Standalone scraper utility used to fetch and update profiles directly from `plaksha.edu.in`. |
| [`tts.py`](tts.py) | Voice service encapsulation for Microsoft Edge TTS (`en-IN-NeerjaNeural`, `rate="+6%"`) with pipelined playback and macOS `say` fallback. |
| [`app.py`](app.py) | Main FastAPI service orchestrating: 1) Camera & face detection loop (`face_recognition`), 2) Audio VAD & Groq Whisper STT loop, 3) Decoupled watchdog inactivity monitor, 4) Pydantic AI conversation turns, and 5) WebSockets + MJPEG video feed. |

---

## Key Improvements

1. **Veda Persona & Name**: Renamed from Kaveri to Veda across all prompts, system messages, and wake-word triggers (`veda`, `vedaa`, etc.).
2. **Decoupled Inactivity Detection**: Inactivity timer only runs during true static silence. When a user is speaking or audio is recording, the timer is suspended and continuously refreshed so long speeches (>10s) are never cut off.
3. **Pipelined TTS & Latency Reduction**: Edge-TTS rate calibrated to `+6%`, initial sentence pipelined to start speaking within ~1 second, with automatic fallback to macOS speech if network socket issues occur.
4. **Structured Two-Part Intro**: Local dataset gives instant 0ms greetings while Tavily runs in the background.

---

## How to Run

From the root directory `/Users/maankumawat15/HTI Lab System`:

```bash
PYTHONUNBUFFERED=1 ./myenv/bin/python -m uvicorn pydantic_ai_agent.app:app --host 0.0.0.0 --port 8000
```

Or run directly with python:
```bash
PYTHONUNBUFFERED=1 ./myenv/bin/python -m pydantic_ai_agent.app
```
