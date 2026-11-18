"""
HTI Lab Interactive Agent ("Veda") - Pydantic AI Framework Version.

Preserves exact same APIs, face detection, Groq Whisper STT, Edge-TTS playback,
and websocket dashboard, while migrating conversation logic, dependencies, tools,
and schemas to Pydantic and Pydantic AI.
"""
import os
import io
import re
import cv2
import json
import time
import math
import wave
import struct
import pickle
import asyncio
import threading
import urllib.request
from pathlib import Path
from typing import List, Optional, Set, Dict, Any

# Suppress Pydantic AI banner
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"

import pandas as pd
# pyrefly: ignore [missing-import]
import face_recognition
from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from groq import Groq

# Local Pydantic modules
from .models import AppConfig, AgentDeps, WebSocketEvent, PlakshaPerson
from .agent import veda_agent, generate_greeting
from .person_search import (
    search_and_summarize,
    search_and_summarize_async,
    find_plaksha_person,
    load_plaksha_people,
)
from .projects import load_hti_projects
from .tts import VoiceService

# ================= Load Environment & Paths =================
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

# Load .env from local or parent directory
if (BASE_DIR / ".env").exists():
    load_dotenv(BASE_DIR / ".env")
else:
    load_dotenv(ROOT_DIR / ".env")

config = AppConfig()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
voice_service = VoiceService(voice=config.edge_tts_voice, rate=config.edge_tts_rate)

# ================= FastAPI Application =================
app = FastAPI(title="HTI Lab Veda Agent (Pydantic AI)")

# ================= Mount Interface Folders =================
HTI_AVATAR_DIR = ROOT_DIR / "HTI_Avatar_Agent"
HTI_WEB_DIR = HTI_AVATAR_DIR / "web"
HTI_FINAL_DIR = HTI_AVATAR_DIR / "final"

if HTI_FINAL_DIR.exists():
    print(f"[AVATAR] Serving 3D avatar & animations from: {HTI_FINAL_DIR}")
    app.mount("/final", StaticFiles(directory=str(HTI_FINAL_DIR)), name="final")

if HTI_WEB_DIR.exists():
    print(f"[FRONTEND] Serving HTI Avatar Agent UI from: {HTI_WEB_DIR}")
    app.mount("/web", StaticFiles(directory=str(HTI_WEB_DIR)), name="web")

# Mount legacy static folder (prioritizing static1 if present)
static_candidates = [
    ROOT_DIR / "static1",
    BASE_DIR / "static1",
    ROOT_DIR / "static",
    BASE_DIR / "static",
]
static_path = next((p for p in static_candidates if p.exists()), BASE_DIR / "static1")

if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    app.mount("/static1", StaticFiles(directory=str(static_path)), name="static1")

active_websockets: List[WebSocket] = []
main_loop: Optional[asyncio.AbstractEventLoop] = None

# ================= Runtime State =================
camera: Optional[cv2.VideoCapture] = None
known_encodings: List[Any] = []
known_names: List[str] = []
consecutive_detections: Dict[str, int] = {}
plaksha_data: Optional[pd.DataFrame] = None

active_session: Optional[str] = None
session_context: str = ""
chat_history: List[Any] = []  # Pydantic AI ModelMessage list
last_speech_time: float = time.time()
is_processing: bool = False
is_speaking_user: bool = False

# Unknown visitor multi-step state machine:
# None -> "awaiting_name" -> "awaiting_intro" -> "conversing"
unknown_visitor_state: Optional[str] = None
unknown_visitor_name: str = ""


def extract_person_name(text: str) -> str:
    """Extract person's name cleanly from phrases like 'My name is John Doe' or 'Mera naam Rohan hai'."""
    cleaned = text.strip()
    match = re.search(r"(?:my name is|i am|i'm|this is|call me|naam hai|mera naam|naam|naam bolte)\s+([A-Za-z\u0900-\u097F]+(?:\s+[A-Za-z\u0900-\u097F]+)?)", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).title()
    # Filter filler words
    words = [w for w in cleaned.split() if w.lower() not in ["a", "the", "student", "at", "plaksha", "university", "and", "main", "mera", "meri", "ek", "hoon"]]
    if len(words) <= 3:
        return " ".join(words).strip(".").title()
    return cleaned


# ================= Initialization =================
def load_data():
    """Load Master CSV, Scraped Plaksha Dataset, and known face encodings."""
    global known_encodings, known_names, plaksha_data
    
    # Load clean local Plaksha dataset
    load_plaksha_people()
    # Load HTI Lab demonstrations & projects dataset
    load_hti_projects()

    # Locate Master CSV
    csv_candidates = [
        ROOT_DIR / "HTI Data" / "2nd_csv_details" / "all_plaksha_people_master.csv",
        BASE_DIR / "HTI Data" / "2nd_csv_details" / "all_plaksha_people_master.csv",
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path:
        try:
            plaksha_data = pd.read_csv(csv_path)
            print(f"[INIT] Loaded {len(plaksha_data)} rows from CSV ({csv_path.name}).")
        except Exception as e:
            print(f"[INIT] Failed to load CSV: {e}")
    else:
        print("[INIT] Warning: all_plaksha_people_master.csv not found.")

    # Locate known_faces.pkl
    pkl_candidates = [
        ROOT_DIR / "known_faces.pkl",
        BASE_DIR / "known_faces.pkl",
    ]
    pkl_path = next((p for p in pkl_candidates if p.exists()), None)
    if pkl_path:
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    known_encodings = [item["encoding"] for item in data]
                    known_names = [item["name"] for item in data]
                else:
                    known_encodings = data.get("encodings", [])
                    known_names = data.get("names", [])
            print(f"[INIT] Loaded {len(known_names)} face encodings.")
        except Exception as e:
            print(f"[INIT] Failed to load known_faces.pkl: {e}")
    else:
        print("[INIT] Warning: No known_faces.pkl found!")


def get_csv_context(name: str) -> str:
    """Retrieve up to 5 matching metadata columns from Plaksha Master CSV."""
    if plaksha_data is None:
        return ""
    matches = plaksha_data[plaksha_data["Name"].str.contains(name, case=False, na=False)]
    if not matches.empty:
        row = matches.iloc[0]
        context_items = [
            f"{col}: {row[col]}"
            for col in matches.columns
            if pd.notna(row[col]) and row[col] != ""
        ]
        return ", ".join(context_items[:5])
    return ""


# ================= Background Enrichment =================
async def background_enrichment(name: str, base_context: str):
    """
    Run Tavily search and profile synthesis in the background (non-blocking).
    Enriches conversation context while greeting is already being spoken.
    """
    global session_context
    try:
        print(f"[BACKGROUND] Starting background Tavily search for: {name}")
        await broadcast_event("search_query", {"query": name, "background": True})
        tavily_start = time.time()
        web_summary = await search_and_summarize_async(name, base_context)
        elapsed = int((time.time() - tavily_start) * 1000)
        print(f"[TIMING] Background Tavily search completed in {elapsed}ms")

        if web_summary:
            # Seamlessly merge web profile into session context for subsequent conversation turns
            session_context = f"{base_context}\n\nRecent Web Insights: {web_summary}".strip()
            await broadcast_event("search_result", {"query": name, "result": web_summary, "background": True})
            print(f"[BACKGROUND] Context enriched for {name}.")
    except Exception as e:
        print(f"[BACKGROUND] Background search failed for {name}: {e}")


# ================= WebSocket Telemetry =================
async def broadcast_event(event: str, data: dict = None):
    """Broadcast structured Pydantic event to all connected dashboard websockets."""
    event_payload = WebSocketEvent(event=event, data=data or {}).to_json()
    for ws in active_websockets:
        try:
            await ws.send_text(event_payload)
        except Exception:
            pass


# ================= Conversational Logic (Pydantic AI) =================
async def handle_conversation_turn(user_text: str):
    """Execute a turn with Veda using Pydantic AI Agent.
    
    Handles the multi-step unknown visitor pipeline:
      Step 1 (awaiting_name): Extract name from user speech, greet them, ask for intro.
      Step 2 (awaiting_intro): Capture intro, launch background search with name+intro,
                                transition to normal conversation.
      Normal: Run Pydantic AI agent turn with full context.
    """
    global chat_history, active_session, session_context, last_speech_time, is_processing
    global unknown_visitor_state, unknown_visitor_name

    if not user_text.strip():
        return

    is_processing = True
    last_speech_time = time.time()

    await broadcast_event("user_message", {"text": user_text})
    await broadcast_event("state_change", {"state": "active", "text": "Thinking..."})

    try:
        # ---- STEP 1: Unknown visitor just told us their name ----
        if unknown_visitor_state == "awaiting_name":
            extracted_name = extract_person_name(user_text)
            print(f"[AI] Extracted visitor name: '{extracted_name}' (from '{user_text}').")
            unknown_visitor_name = extracted_name
            active_session = extracted_name

            # Check if this person is actually in our local dataset
            person = find_plaksha_person(extracted_name)
            if person:
                # They ARE in the dataset — treat as known from here on
                local_summary = person.to_conversational_summary()
                session_context = local_summary
                print(f"[AI] Local dataset matched for '{extracted_name}': {local_summary[:80]}...")
                unknown_visitor_state = None  # skip intro, treat as known

                # Launch background web enrichment
                asyncio.create_task(background_enrichment(extracted_name, local_summary))

                # Generate a personalized welcome using the greeting agent
                greeting = await generate_greeting(extracted_name, local_summary)
                await broadcast_event("ai_response", {"text": greeting})
                await voice_service.speak(greeting, broadcast_event)
            else:
                # Not in dataset — move to intro step
                unknown_visitor_state = "awaiting_intro"
                deps = AgentDeps(
                    visitor_name=extracted_name,
                    session_context="",
                    broadcast_callback=broadcast_event
                )
                ask_prompt = (
                    f"The visitor just told you their name: '{extracted_name}'. Their input was: '{user_text}'. "
                    "1. Greet them warmly by name in 1 sentence. "
                    "2. Ask if they mind giving a short introduction about themselves. "
                    "3. CRITICAL: You are female assistant Veda. Always use feminine Hindi grammar (e.g. 'samajh gayi', 'kar sakti hoon', 'mujhe pata hai'). Match the visitor's language! If they spoke in Hindi, ask in Hindi. If in English, ask in English. If in Hinglish, ask in Hinglish. "
                    "Keep under 18 words for speech."
                )
                result = await veda_agent.run(ask_prompt, deps=deps)
                intro_ask = result.output
                await broadcast_event("ai_response", {"text": intro_ask})
                await voice_service.speak(intro_ask, broadcast_event)
            return

        # ---- STEP 2: Unknown visitor just gave their intro ----
        if unknown_visitor_state == "awaiting_intro":
            visitor_intro = user_text.strip()
            print(f"[AI] Received intro from '{unknown_visitor_name}': {visitor_intro[:100]}...")
            session_context = f"Visitor '{unknown_visitor_name}' introduced themselves as: {visitor_intro}"
            unknown_visitor_state = None  # transition to normal conversation

            # Launch background web search with name + intro for enrichment
            asyncio.create_task(background_enrichment(unknown_visitor_name, session_context))

            # Acknowledge the intro warmly and start conversing, strictly matching language
            deps = AgentDeps(
                visitor_name=unknown_visitor_name,
                session_context=session_context,
                broadcast_callback=broadcast_event
            )
            ack_prompt = (
                f"The visitor just introduced themselves. Their name is {unknown_visitor_name} "
                f"and here is their intro: {visitor_intro}. "
                "Respond warmly as female assistant Veda (using feminine grammar e.g. 'samajh gayi', 'bata sakti hoon'), mention something from their intro naturally, and keep the conversation going. "
                "CRITICAL: Match the visitor's language! If their intro was in English, reply in English. If in Hindi, reply in Hindi. If in Hinglish, reply in Hinglish. "
                "Keep response very concise (1 to 2 short sentences, maximum 25 words)."
            )
            result = await veda_agent.run(ack_prompt, deps=deps, message_history=chat_history)
            chat_history = result.all_messages()
            ai_reply = result.output

            await broadcast_event("ai_response", {"text": ai_reply})
            await voice_service.speak(ai_reply, broadcast_event)
            return

        # ---- NORMAL CONVERSATION TURN ----
        # Prepare injected dependencies for Pydantic AI Agent
        deps = AgentDeps(
            visitor_name=active_session or "Friend",
            session_context=session_context,
            broadcast_callback=broadcast_event
        )

        llm_start = time.time()
        # Run turn with Pydantic AI, preserving message history and auto-calling search tools if needed
        result = await veda_agent.run(
            user_text,
            deps=deps,
            message_history=chat_history
        )
        print(f"[TIMING] Pydantic AI turn: {int((time.time() - llm_start) * 1000)}ms")

        # Update message history
        chat_history = result.all_messages()
        ai_reply = result.output

        await broadcast_event("ai_response", {"text": ai_reply})
        # Speak response (blocks until audio finishes)
        await voice_service.speak(ai_reply, broadcast_event)

    except Exception as e:
        print(f"[AI] Error during conversation turn: {e}")
        error_msg = "I'm sorry, I encountered an issue processing that. Could you say that again?"
        await broadcast_event("ai_response", {"text": error_msg})
        await voice_service.speak(error_msg, broadcast_event)
    finally:
        last_speech_time = time.time()
        is_processing = False


async def start_session_async(name: str):
    """
    Initiate an active session when face is confirmed or wake word is heard.

    Pipeline:
    KNOWN VISITOR:
      1. Greet them warmly, include a brief HTI Lab intro, welcome them.
      2. Start interaction based on ground data (local dataset / CSV bio).
      3. Launch background web search to enrich context for later turns.

    UNKNOWN VISITOR:
      1. Greet them warmly, include a brief HTI Lab intro, welcome them.
      2. Ask their name.  (→ awaiting_name state)
      3. On receiving name, ask for a short intro.  (→ awaiting_intro state)
      4. On receiving intro, launch background web search with name+intro.
      5. Start interaction based on their intro + web results.
    """
    global active_session, chat_history, session_context, last_speech_time, is_processing
    global unknown_visitor_state, unknown_visitor_name

    active_session = name
    chat_history = []
    session_context = ""
    last_speech_time = time.time()
    is_processing = True
    unknown_visitor_state = None
    unknown_visitor_name = ""

    is_known = (name != "Unknown Visitor")
    await broadcast_event("face_detected", {"name": name, "known": is_known})

    try:
        if is_known:
            # ---- KNOWN VISITOR PIPELINE ----
            # PART 1: Instant local dataset lookup
            person = find_plaksha_person(name)
            if person:
                local_summary = person.to_conversational_summary()
                print(f"[AI] Local Plaksha dataset match for '{name}': {local_summary[:90]}...")
            else:
                local_summary = get_csv_context(name)

            session_context = local_summary

            # Generate greeting that includes HTI Lab intro + personalized welcome
            llm_start = time.time()
            greeting = await generate_greeting(name, local_summary)
            print(f"[TIMING] Instant Pydantic AI greeting: {int((time.time() - llm_start) * 1000)}ms")

            # PART 2: Launch background Tavily web search (non-blocking)
            asyncio.create_task(background_enrichment(name, local_summary))

            # Speak greeting immediately without waiting for Tavily
            await broadcast_event("ai_response", {"text": greeting})
            await voice_service.speak(greeting, broadcast_event)
        else:
            # ---- UNKNOWN VISITOR PIPELINE ----
            # Set state machine to await their name next
            unknown_visitor_state = "awaiting_name"

            greeting = (
                "Namaste! Welcome to the Human Technology Interaction Lab at Plaksha University. "
                "I'm Veda, your AI lab assistant. "
                "The HTI Lab blends multimodal bio-sensing with affective computing to build AI that truly "
                "understands and responds to human emotions and behaviour in real-time. "
                "May I know your name?"
            )
            await broadcast_event("ai_response", {"text": greeting})
            await voice_service.speak(greeting, broadcast_event)
    except Exception as e:
        print(f"[AI] Session init error: {e}")
    finally:
        last_speech_time = time.time()
        is_processing = False


# ================= Watchdog Thread =================
def inactivity_monitor():
    """Reset to idle after true silence. Never fires while someone is talking or recording audio."""
    global active_session, last_speech_time, is_processing, is_speaking_user
    global unknown_visitor_state, unknown_visitor_name
    while True:
        time.sleep(0.5)
        # ONLY check inactivity if nobody is speaking, recording audio, or running an LLM turn
        if (
            active_session is not None
            and not voice_service.is_speaking
            and not is_processing
            and not is_speaking_user
        ):
            if time.time() - last_speech_time > config.inactivity_timeout_seconds:
                print(f"[AI] {config.inactivity_timeout_seconds}s of complete silence. Going to sleep.")
                is_processing = True
                try:
                    asyncio.run_coroutine_threadsafe(
                        voice_service.speak("Main yahaan hoon, jab zaroorat ho bula lena!", broadcast_event),
                        main_loop
                    )
                finally:
                    active_session = None
                    unknown_visitor_state = None
                    unknown_visitor_name = ""
                    is_processing = False
                    last_speech_time = time.time()
                asyncio.run_coroutine_threadsafe(
                    broadcast_event("state_change", {"state": "idle", "text": "System Ready"}),
                    main_loop
                )


# ================= Groq Whisper STT Thread =================
def transcribe_audio(audio_bytes: bytes) -> str:
    """Send recorded audio buffer to Groq Whisper."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.audio_rate)
        wf.writeframes(audio_bytes)
    wav_buffer.seek(0)

    try:
        transcription = groq_client.audio.transcriptions.create(
            file=("recording.wav", wav_buffer.read()),
            model=config.stt_model,
            response_format="text"
            # Omit language parameter so Whisper auto-detects English, Hindi, and multilingual audio
        )
        return transcription.strip() if isinstance(transcription, str) else str(transcription).strip()
    except Exception as e:
        print(f"[STT] Whisper error: {e}")
        return ""


def stt_loop():
    """Continuously monitor microphone, run energy VAD, and transcribe via Groq Whisper."""
    global active_session, is_speaking_user, last_speech_time
    import pyaudio

    # Launch watchdog thread
    threading.Thread(target=inactivity_monitor, daemon=True).start()

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=config.audio_rate,
        input=True,
        frames_per_buffer=config.audio_chunk
    )
    print("[STT] PyAudio + Groq Whisper microphone stream started.")

    audio_buffer = []
    is_speaking_user = False
    silence_start = None
    speech_start = None

    while True:
        try:
            data = stream.read(config.audio_chunk, exception_on_overflow=False)

            # Skip listening while Veda is speaking or processing
            if voice_service.is_speaking or is_processing:
                continue

            # Calculate RMS audio energy
            count = len(data) // 2
            shorts = struct.unpack(f"{count}h", data)
            rms = math.sqrt(sum(s * s for s in shorts) / count) if count > 0 else 0

            if rms > config.silence_threshold:
                if not is_speaking_user:
                    is_speaking_user = True
                    speech_start = time.time()
                    audio_buffer = []
                    print("[STT] Speech detected, recording...")
                silence_start = None
                audio_buffer.append(data)
                # Constantly refresh last_speech_time while audio is being recorded
                last_speech_time = time.time()
            elif is_speaking_user:
                audio_buffer.append(data)
                last_speech_time = time.time()  # Still in active user utterance window
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= config.silence_duration:
                    duration = time.time() - speech_start
                    if duration >= config.min_speech_duration:
                        print(f"[STT] Speech ended ({duration:.1f}s). Transcribing with Whisper...")
                        raw_audio = b"".join(audio_buffer)

                        stt_start = time.time()
                        transcript = transcribe_audio(raw_audio)
                        print(f"[TIMING] STT transcription: {int((time.time() - stt_start) * 1000)}ms")

                        if transcript:
                            print(f"[STT] Heard: {transcript}")
                            transcript_lower = transcript.lower()

                            # Wake word trigger if idle
                            if active_session is None:
                                wake_triggers = ["veda", "vedaa", "vedha", "vaida", "wayda", "kaveri"]
                                if any(w in transcript_lower for w in wake_triggers):
                                    print("[AI] Wake word 'Veda' detected!")
                                    asyncio.run_coroutine_threadsafe(start_session_async("Unknown Visitor"), main_loop)
                            else:
                                # Stop word trigger
                                if any(w in transcript_lower for w in ["bye", "goodbye", "alvida"]):
                                    print("[AI] End of conversation detected.")
                                    asyncio.run_coroutine_threadsafe(
                                        voice_service.speak("Bye bye! Have a great day!", broadcast_event),
                                        main_loop
                                    )
                                    active_session = None
                                    unknown_visitor_state = None
                                    unknown_visitor_name = ""
                                    asyncio.run_coroutine_threadsafe(
                                        broadcast_event("state_change", {"state": "idle", "text": "System Ready"}),
                                        main_loop
                                    )
                                else:
                                    asyncio.run_coroutine_threadsafe(handle_conversation_turn(transcript), main_loop)
                    else:
                        print("[STT] Short audio burst ignored.")

                    is_speaking_user = False
                    silence_start = None
                    audio_buffer = []
                    last_speech_time = time.time()

        except Exception as e:
            print(f"[STT] Audio loop error: {e}")


latest_jpeg_frame: Optional[bytes] = None
frame_lock: threading.Lock = threading.Lock()

# ================= Face Recognition Thread =================
def camera_loop():
    """Continuously capture frames and recognize faces against known_faces.pkl."""
    global camera, consecutive_detections, latest_jpeg_frame
    try:
        camera = cv2.VideoCapture(config.camera_index)
    except Exception as e:
        print(f"[CAMERA] Init error: {e}")
        return

    frame_count = 0

    while True:
        try:
            if camera is None or not camera.isOpened():
                time.sleep(0.5)
                continue

            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1

            # Thread-safe JPEG frame caching for /video_feed stream
            if frame_count % 2 == 0:
                ret_enc, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret_enc:
                    with frame_lock:
                        latest_jpeg_frame = buffer.tobytes()

            # Only check faces if idle and not currently processing/speaking
            if frame_count % config.process_nth_frame == 0 and active_session is None and not is_processing:
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                locations = face_recognition.face_locations(rgb, model="hog")
                encodings = face_recognition.face_encodings(rgb, locations)

                detected_names: Set[str] = set()
                for enc in encodings:
                    distances = face_recognition.face_distance(known_encodings, enc)
                    if len(distances) > 0 and distances.min() <= config.face_distance_threshold:
                        idx = distances.argmin()
                        name = known_names[idx]
                    else:
                        name = "Unknown Visitor"
                    detected_names.add(name)

                for n in list(consecutive_detections.keys()):
                    if n not in detected_names:
                        consecutive_detections[n] = 0

                for n in detected_names:
                    consecutive_detections[n] = consecutive_detections.get(n, 0) + 1
                    if consecutive_detections[n] >= config.consecutive_frames:
                        print(f"[FACE] Confirmed: {n}")
                        consecutive_detections = {}
                        asyncio.run_coroutine_threadsafe(start_session_async(n), main_loop)
                        break

            time.sleep(0.01)
        except Exception as cam_err:
            print(f"[CAMERA] Frame loop error: {cam_err}")
            time.sleep(0.1)


# ================= FastAPI Lifecycle & Endpoints =================
@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()
    load_data()

    # Launch camera & STT threads
    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=stt_loop, daemon=True).start()
    print("[OK] Pydantic AI Server started. Camera and STT threads running.")


@app.get("/")
def index():
    if HTI_WEB_DIR.exists():
        hti_index = HTI_WEB_DIR / "index.html"
        if hti_index.exists():
            with open(hti_index, "r") as f:
                return HTMLResponse(f.read())
    html_file = static_path / "index.html"
    if html_file.exists():
        with open(html_file, "r") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>HTI Lab Veda (Pydantic AI) Running</h1>")


@app.post("/api/upload-avatar")
async def upload_avatar(request: Request):
    try:
        data = await request.body()
        target = HTI_FINAL_DIR / "meera.glb"
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            f.write(data)
        print(f"[API] Uploaded new avatar to {target} ({len(data)} bytes)")
        return JSONResponse({"status": "success", "saved": True})
    except Exception as e:
        print(f"[API] Upload error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/save-avatar")
async def save_avatar(request: Request):
    try:
        payload = await request.json()
        url = payload.get("url")
        if url:
            if "morphTargets=" not in url:
                url += "?morphTargets=ARKit,Oculus+Visemes"
            target = HTI_FINAL_DIR / "meera.glb"
            target.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, str(target))
            print(f"[API] Saved avatar from URL to {target}")
            return JSONResponse({"status": "success", "saved": True})
        return JSONResponse({"status": "error", "message": "Missing URL"}, status_code=400)
    except Exception as e:
        print(f"[API] Save error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def generate_mjpeg():
    """Stream cached camera frames asynchronously without thread blocking."""
    try:
        while True:
            with frame_lock:
                jpeg = latest_jpeg_frame
            if jpeg is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            await asyncio.sleep(0.04)
    except asyncio.CancelledError:
        pass


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    voice_service.has_active_web_audio = True
    print(f"[WS] Client connected ({len(active_websockets)} active). Web audio enabled for 3D avatar.")
    try:
        while True:
            text = await websocket.receive_text()
            try:
                msg = json.loads(text)
                event_type = msg.get("event")
                if event_type == "audio_mode":
                    voice_service.has_active_web_audio = bool(msg.get("data", {}).get("web_audio", True))
                    print(f"[WS] Audio mode updated: web_audio={voice_service.has_active_web_audio}")
                elif event_type in ("user_text", "user_message"):
                    user_input = msg.get("text") or msg.get("data", {}).get("text")
                    if user_input and user_input.strip():
                        print(f"[WS] Received user message: '{user_input.strip()}'")
                        asyncio.create_task(handle_conversation_turn(user_input.strip()))
            except Exception as parse_err:
                print(f"[WS] Message parse error: {parse_err}")
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
        if len(active_websockets) == 0:
            voice_service.has_active_web_audio = False
            print("[WS] All web clients disconnected. Falling back to system afplay.")


# Mount root static fallback for UI assets (style.css, app.js, fonts, etc.)
if HTI_WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(HTI_WEB_DIR), html=True), name="root_web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pydantic_ai_agent.app:app", host="0.0.0.0", port=8000, reload=False)
