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
from fastapi.middleware.cors import CORSMiddleware
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

# Enable CORS for website embeds and GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= Mount Interface Folders =================
HTI_AVATAR_DIR = ROOT_DIR / "HTI_Avatar_Agent"
HTI_WEB_DIR = HTI_AVATAR_DIR / "web"
HTI_FINAL_DIR = HTI_AVATAR_DIR / "final"
HTI_WEBSITE_DIR = ROOT_DIR / "Website-copy"

if HTI_FINAL_DIR.exists():
    print(f"[AVATAR] Serving 3D avatar & animations from: {HTI_FINAL_DIR}")
    app.mount("/final", StaticFiles(directory=str(HTI_FINAL_DIR)), name="final")

if HTI_WEB_DIR.exists():
    print(f"[FRONTEND] Serving HTI Avatar Agent UI from: {HTI_WEB_DIR}")
    app.mount("/web", StaticFiles(directory=str(HTI_WEB_DIR)), name="web")

if HTI_WEBSITE_DIR.exists():
    print(f"[WEBSITE] Serving HTI Lab Website from: {HTI_WEBSITE_DIR}")
    app.mount("/site", StaticFiles(directory=str(HTI_WEBSITE_DIR), html=True), name="site")

# Mount legacy static folder (prioritizing static1 if present)
static_candidates = [
    ROOT_DIR / "static1",
    BASE_DIR / "static1",
    ROOT_DIR / "static",
    BASE_DIR / "static",
]
static_path = next((p for p in static_candidates if p.exists()), BASE_DIR / "static1")

if static_path.exists():
    print(f"[STATIC] Mounting static test interface from: {static_path}")
    app.mount("/static", StaticFiles(directory=str(static_path), html=True), name="static")
    app.mount("/static1", StaticFiles(directory=str(static_path), html=True), name="static1")

@app.get("/test")
def test_static():
    index_file = static_path / "index.html"
    if index_file.exists():
        with open(index_file, "r") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Static test not found</h1>")

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

# Active control toggle (Start / Stop Veda)
is_veda_active: bool = False

# Group tour & visitor state machine:
# None -> "awaiting_day_response" -> "awaiting_demo_or_chat_choice" -> "awaiting_individual_name" -> "awaiting_individual_intro" -> "individual_chat" / "completed_tour_handoff"
group_visit_state: Optional[str] = None
current_group_known: List[str] = []
current_group_unknown_count: int = 0
unknown_visitor_name: str = ""

HTI_VISION_PARAGRAPH: str = (
    "The HTI Lab pioneers the next frontier of Human-AI Interaction by seamlessly blending multimodal bio-sensing with affective computing to build systems that truly understand and dynamically respond to human emotions and behavior in real-time."
)


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
    
    Supports:
      Stage 1 (awaiting_day_response):
        - Acknowledges how their day was in 1 warm sentence matching visitor language.
        - Introduces handout & 6 lab demonstrations (Dhvani, PosePilot, ATLAS, VR, Robotic Arm).
        - Asks: 'Would you like to continue chatting with me, or would you like to proceed for the live demonstrations?'
        - Transitions to 'awaiting_demo_or_chat_choice'.
      Stage 2 (awaiting_demo_or_chat_choice):
        - If they choose live demos: Veda cheerfully hands off to team and returns to standby.
        - If they choose to chat with Veda:
          - If a known visitor: enters personalized 1-on-1 chat with RAG context.
          - If unknown: asks for their name ('awaiting_individual_name').
      Stage 3 (awaiting_individual_name):
        - Extracts name -> checks local Plaksha dataset -> if match, personalizes + background Tavily search.
        - If not in dataset -> asks for short intro ('awaiting_individual_intro').
      Stage 4 (awaiting_individual_intro):
        - Receives intro -> background Tavily search -> starts rich conversation.
      Normal Conversation:
        - Full Pydantic AI turn with conversation history, projects tool, Tavily search tool.
    """
    global chat_history, active_session, session_context, last_speech_time, is_processing
    global group_visit_state, current_group_known, current_group_unknown_count, unknown_visitor_name

    if not user_text.strip():
        return

    is_processing = True
    last_speech_time = time.time()

    await broadcast_event("user_message", {"text": user_text})
    await broadcast_event("state_change", {"state": "active", "text": "Thinking..."})

    try:
        # =========================================================================
        # STAGE 1: Visitors replied to "How was your day so far?"
        # =========================================================================
        if group_visit_state == "awaiting_day_response":
            print(f"[AI] Visitors answered 'how was your day': {user_text}")

            ack_prompt = (
                f"The visitors were just asked 'How was your day so far?' and their reply was: '{user_text}'. "
                "Respond in 1 very short, warm, empathetic sentence (under 12 words) acknowledging their day. "
                "CRITICAL: Match the visitor's language (English, Hindi, or Hinglish) using feminine grammar ('sun kar accha laga', 'glad to hear that')."
            )
            deps = AgentDeps(
                visitor_name=active_session or "Visitors",
                session_context="",
                broadcast_callback=broadcast_event
            )
            result = await veda_agent.run(ack_prompt, deps=deps)
            day_ack = result.output.strip()

            demos_overview = (
                "Our team has just handed you a quick overview of our lab tour. "
                "We have some exciting live demonstrations set up for you today — including Dhvani for Hindi speech feedback, "
                "PosePilot for real-time movement correction, ATLAS for assistive spatial navigation, "
                "an immersive VR experience, and a robotic arm you can play Rock-Paper-Scissors with! "
                "Would you like to continue chatting with me, or would you like to proceed for the live demonstrations?"
            )

            full_reply = f"{day_ack} {demos_overview}"
            group_visit_state = "awaiting_demo_or_chat_choice"

            await broadcast_event("ai_response", {"text": full_reply})
            await voice_service.speak(full_reply, broadcast_event)
            return

        # =========================================================================
        # STAGE 2: Visitors choice (Demos vs Chat with Veda)
        # =========================================================================
        if group_visit_state == "awaiting_demo_or_chat_choice":
            text_lower = user_text.lower()
            demo_keywords = ["demo", "demonstration", "tour", "robotic", "arm", "vr", "posepilot", "dhvani", "atlas", "proceed", "walk", "see", "dekhna", "aage", "chalo", "let's go"]
            is_demo = any(w in text_lower for w in demo_keywords) and not any(w in text_lower for w in ["chat with you", "talk to you"])
            
            if is_demo:
                tour_reply = (
                    "Wonderful! Please enjoy the live demonstrations with our team. "
                    "I will be right here whenever you would like to chat again!"
                )
                group_visit_state = "completed_tour_handoff"
                active_session = None
                await broadcast_event("ai_response", {"text": tour_reply})
                await voice_service.speak(tour_reply, broadcast_event)
                await broadcast_event("state_change", {"state": "idle", "text": "Tour in Progress • Veda Standby"})
                return
            else:
                # User chose to chat with Veda!
                if current_group_known:
                    known_person = current_group_known[0]
                    active_session = known_person
                    group_visit_state = "individual_chat"
                    chat_intro = f"I would love to chat! {known_person}, what would you or the group like to explore first?"
                    await broadcast_event("ai_response", {"text": chat_intro})
                    await voice_service.speak(chat_intro, broadcast_event)
                    return
                else:
                    group_visit_state = "awaiting_individual_name"
                    ask_name = "I would love to chat! May I know your name?"
                    await broadcast_event("ai_response", {"text": ask_name})
                    await voice_service.speak(ask_name, broadcast_event)
                    return

        # =========================================================================
        # STAGE 3: Individual visitor tells us their name
        # =========================================================================
        if group_visit_state == "awaiting_individual_name":
            extracted_name = extract_person_name(user_text)
            print(f"[AI] Extracted visitor name: '{extracted_name}' (from '{user_text}').")
            unknown_visitor_name = extracted_name
            active_session = extracted_name

            person = find_plaksha_person(extracted_name)
            if person:
                local_summary = person.to_conversational_summary()
                session_context = local_summary
                print(f"[AI] Local dataset matched for '{extracted_name}': {local_summary[:80]}...")
                group_visit_state = "individual_chat"

                asyncio.create_task(background_enrichment(extracted_name, local_summary))

                greeting = await generate_greeting(extracted_name, local_summary)
                await broadcast_event("ai_response", {"text": greeting})
                await voice_service.speak(greeting, broadcast_event)
            else:
                group_visit_state = "awaiting_individual_intro"
                deps = AgentDeps(
                    visitor_name=extracted_name,
                    session_context="",
                    broadcast_callback=broadcast_event
                )
                ask_prompt = (
                    f"The visitor just told you their name: '{extracted_name}'. Their input was: '{user_text}'. "
                    "1. Greet them warmly by name in 1 sentence. "
                    "2. Ask if they mind giving a short introduction about themselves. "
                    "3. CRITICAL: You are female assistant Veda. Always use feminine Hindi grammar ('samajh gayi', 'kar sakti hoon'). Match their language! "
                    "Keep under 18 words for speech."
                )
                result = await veda_agent.run(ask_prompt, deps=deps)
                intro_ask = result.output
                await broadcast_event("ai_response", {"text": intro_ask})
                await voice_service.speak(intro_ask, broadcast_event)
            return

        # =========================================================================
        # STAGE 4: Individual visitor gives intro
        # =========================================================================
        if group_visit_state == "awaiting_individual_intro":
            visitor_intro = user_text.strip()
            session_context = f"Visitor '{unknown_visitor_name}' introduced themselves as: {visitor_intro}"
            group_visit_state = "individual_chat"

            asyncio.create_task(background_enrichment(unknown_visitor_name, session_context))

            deps = AgentDeps(
                visitor_name=unknown_visitor_name,
                session_context=session_context,
                broadcast_callback=broadcast_event
            )
            ack_prompt = (
                f"The visitor just introduced themselves. Their name is {unknown_visitor_name} "
                f"and here is their intro: {visitor_intro}. "
                "Respond warmly as female assistant Veda, mention something from their intro naturally, and keep the conversation going. "
                "Match their language! Keep under 25 words."
            )
            result = await veda_agent.run(ack_prompt, deps=deps, message_history=chat_history)
            chat_history = result.all_messages()
            ai_reply = result.output

            await broadcast_event("ai_response", {"text": ai_reply})
            await voice_service.speak(ai_reply, broadcast_event)
            return

        # =========================================================================
        # NORMAL 1-ON-1 / GROUP CHAT TURN
        # =========================================================================
        deps = AgentDeps(
            visitor_name=active_session or "Friend",
            session_context=session_context,
            broadcast_callback=broadcast_event
        )

        llm_start = time.time()
        result = await veda_agent.run(
            user_text,
            deps=deps,
            message_history=chat_history
        )
        print(f"[TIMING] Pydantic AI turn: {int((time.time() - llm_start) * 1000)}ms")

        chat_history = result.all_messages()
        ai_reply = result.output

        await broadcast_event("ai_response", {"text": ai_reply})
        await voice_service.speak(ai_reply, broadcast_event)

    except Exception as e:
        print(f"[AI] Error during conversation turn: {e}")
        error_msg = "I'm sorry, I encountered an issue processing that. Could you say that again?"
        await broadcast_event("ai_response", {"text": error_msg})
        await voice_service.speak(error_msg, broadcast_event)
    finally:
        last_speech_time = time.time()
        is_processing = False


async def start_group_session_async(known_names_detected: List[str], unknown_count: int):
    """
    Initiate group greeting session when 4-5 people walk into the HTI Lab room.
    
    Logic:
    - If all people are known (1 or more known names, 0 unknown):
      Personalized greeting naming each person!
    - If mixed group (some known, some unknown):
      Greets everyone, specifically welcoming the known person(s) warmly.
    - If completely unknown or mostly unindexed:
      Universal group greeting without assuming names.
    
    Then speaks the HTI Lab vision paragraph, asks 'How was your day so far?',
    and transitions to group_visit_state = 'awaiting_day_response'.
    """
    global active_session, chat_history, session_context, last_speech_time, is_processing
    global group_visit_state, current_group_known, current_group_unknown_count, unknown_visitor_name

    current_group_known = list(known_names_detected)
    current_group_unknown_count = unknown_count
    chat_history = []
    session_context = ""
    last_speech_time = time.time()
    is_processing = True
    unknown_visitor_name = ""

    total_people = len(known_names_detected) + unknown_count
    primary_name = known_names_detected[0] if known_names_detected else "Group Visit"
    active_session = primary_name

    await broadcast_event("face_detected", {
        "names": known_names_detected,
        "unknown_count": unknown_count,
        "total_people": total_people,
        "name": primary_name
    })

    try:
        # 1. Determine Welcome line
        if known_names_detected and unknown_count == 0:
            if len(known_names_detected) == 1:
                welcome_line = f"Namaste {known_names_detected[0]}! Welcome to the Human-Technology Interaction Lab at Plaksha University. I am Veda, your AI lab assistant."
            else:
                names_joined = " and ".join(known_names_detected) if len(known_names_detected) <= 2 else f"{', '.join(known_names_detected[:-1])}, and {known_names_detected[-1]}"
                welcome_line = f"Namaste {names_joined}! Welcome to the Human-Technology Interaction Lab at Plaksha University. I am Veda, your AI lab assistant."
        elif known_names_detected and unknown_count > 0:
            names_joined = " and ".join(known_names_detected)
            welcome_line = f"Hello everyone, and a warm Namaste to {names_joined}! Welcome to the Human-Technology Interaction Lab at Plaksha University. I am Veda, your AI lab assistant."
        else:
            welcome_line = "Hello everyone! A very warm welcome to the Human-Technology Interaction Lab at Plaksha University. I am Veda, your AI lab assistant."

        # 2. Build full greeting speech
        greeting_speech = (
            f"{welcome_line} "
            f"{HTI_VISION_PARAGRAPH} "
            "How was your day so far?"
        )

        # Pre-load context if a known person is present
        if known_names_detected:
            person = find_plaksha_person(known_names_detected[0])
            if person:
                session_context = person.to_conversational_summary()
                asyncio.create_task(background_enrichment(known_names_detected[0], session_context))
            else:
                session_context = get_csv_context(known_names_detected[0])

        await broadcast_event("ai_response", {"text": greeting_speech})
        await voice_service.speak(greeting_speech, broadcast_event)

        group_visit_state = "awaiting_day_response"

    except Exception as e:
        print(f"[AI] Group session init error: {e}")
        group_visit_state = "awaiting_day_response"
    finally:
        last_speech_time = time.time()
        is_processing = False


async def start_session_async(name: str):
    """Compatibility wrapper for single face or wake-word triggers."""
    if name != "Unknown Visitor":
        await start_group_session_async([name], 0)
    else:
        await start_group_session_async([], 1)


# ================= Watchdog Thread =================
def inactivity_monitor():
    """Reset to idle after true silence. Never fires while someone is talking or recording audio."""
    global active_session, last_speech_time, is_processing, is_speaking_user
    global group_visit_state, unknown_visitor_name, is_veda_active
    while True:
        time.sleep(0.5)
        # ONLY check inactivity if Veda is ACTIVE and nobody is speaking, recording audio, or running an LLM turn
        if (
            is_veda_active
            and active_session is not None
            and not voice_service.is_speaking
            and not is_processing
            and not is_speaking_user
        ):
            # If in completed_tour_handoff state, don't interrupt live demos
            if group_visit_state == "completed_tour_handoff":
                continue
            if time.time() - last_speech_time > config.inactivity_timeout_seconds:
                print(f"[AI] {config.inactivity_timeout_seconds}s of complete silence. Resetting session to ready.")
                active_session = None
                group_visit_state = None
                unknown_visitor_name = ""
                is_processing = False
                last_speech_time = time.time()
                asyncio.run_coroutine_threadsafe(
                    broadcast_event("state_change", {"state": "idle", "text": "Veda Active • Watching Camera"}),
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
    global active_session, is_speaking_user, last_speech_time, is_veda_active
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

            # Skip listening if Veda is NOT active (Paused) or is speaking/processing
            if not is_veda_active or voice_service.is_speaking or is_processing:
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


def open_camera():
    global camera
    if camera is None or not camera.isOpened():
        try:
            camera = cv2.VideoCapture(config.camera_index)
            print(f"[CAMERA] Hardware webcam opened on index {config.camera_index}")
        except Exception as e:
            print(f"[CAMERA] Open error: {e}")
            camera = None


def close_camera():
    global camera, latest_jpeg_frame
    if camera is not None:
        try:
            camera.release()
            print("[CAMERA] Hardware webcam released and turned OFF.")
        except Exception as e:
            print(f"[CAMERA] Release error: {e}")
        camera = None
    with frame_lock:
        latest_jpeg_frame = None


# ================= Face Recognition Thread =================
def camera_loop():
    """Continuously capture frames and recognize groups/faces only when active."""
    global camera, consecutive_detections, latest_jpeg_frame, is_veda_active

    frame_count = 0
    group_window_start: Optional[float] = None
    group_known_seen: Set[str] = set()
    group_max_unknown: int = 0
    group_frame_hits: int = 0

    while True:
        try:
            # If camera is disabled in config or Veda is paused, ensure camera is released
            if not config.enable_camera or not is_veda_active:
                if camera is not None:
                    close_camera()
                time.sleep(0.5)
                continue

            # Open hardware camera on demand only when Veda is explicitly started
            if camera is None or not camera.isOpened():
                open_camera()
                if camera is None or not camera.isOpened():
                    time.sleep(1.0)
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

            # ONLY check faces if Veda is ACTIVE and idle (not currently in a session or speaking)
            if is_veda_active and frame_count % config.process_nth_frame == 0 and active_session is None and not is_processing:
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                locations = face_recognition.face_locations(rgb, model="hog")
                if locations:
                    encodings = face_recognition.face_encodings(rgb, locations)

                    frame_known = set()
                    frame_unknown_count = 0

                    for enc in encodings:
                        distances = face_recognition.face_distance(known_encodings, enc)
                        if len(distances) > 0 and distances.min() <= config.face_distance_threshold:
                            idx = distances.argmin()
                            frame_known.add(known_names[idx])
                        else:
                            frame_unknown_count += 1

                    # Group accumulation window across frames (captures all 4-5 people as they walk in)
                    if group_window_start is None:
                        group_window_start = time.time()
                        group_known_seen = set(frame_known)
                        group_max_unknown = frame_unknown_count
                        group_frame_hits = 1
                    else:
                        group_known_seen.update(frame_known)
                        group_max_unknown = max(group_max_unknown, frame_unknown_count)
                        group_frame_hits += 1

                    elapsed = time.time() - group_window_start
                    # Once we have at least 3 hits across >= 1.0 second of stable presence
                    if group_frame_hits >= 3 and elapsed >= 1.0:
                        final_known = list(group_known_seen)
                        final_unknown = group_max_unknown
                        print(f"[GROUP CONFIRMED] Known: {final_known}, Unknown count: {final_unknown}")

                        # Reset window
                        group_window_start = None
                        group_known_seen = set()
                        group_max_unknown = 0
                        group_frame_hits = 0

                        asyncio.run_coroutine_threadsafe(
                            start_group_session_async(final_known, final_unknown),
                            main_loop
                        )
                else:
                    # No faces in frame - reset window if > 1.2s of no faces
                    if group_window_start and (time.time() - group_window_start > 1.2):
                        group_window_start = None
                        group_known_seen = set()
                        group_max_unknown = 0
                        group_frame_hits = 0

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
    if config.enable_camera:
        threading.Thread(target=camera_loop, daemon=True).start()
        print("[CAMERA] Camera monitoring thread started (enable_camera=True).")
    else:
        print("[CAMERA] Camera is DISABLED. Hardware webcam will NOT be opened.")

    threading.Thread(target=stt_loop, daemon=True).start()
    print("[OK] Pydantic AI Server started. STT thread running.")


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


@app.get("/api/status")
async def get_status():
    return JSONResponse({
        "status": "success",
        "active": is_veda_active,
        "group_visit_state": group_visit_state,
        "current_person": current_person
    })


@app.post("/api/toggle-active")
async def toggle_active_api(request: Request):
    global is_veda_active, active_session, group_visit_state
    try:
        body = await request.json()
        target = body.get("active")
        if target is None:
            is_veda_active = not is_veda_active
        else:
            is_veda_active = bool(target)
    except Exception:
        is_veda_active = not is_veda_active

    print(f"[API] Veda active state toggled to: {is_veda_active}")
    await broadcast_event("veda_status", {"active": is_veda_active})

    if is_veda_active:
        if active_session is None and not is_processing:
            print("[API] Starting Veda group visit session and greeting pipeline...")
            asyncio.create_task(start_group_session_async([], 0))
    else:
        close_camera()
        active_session = None
        group_visit_state = "idle"
        voice_service.is_speaking = False

    return JSONResponse({"status": "success", "active": is_veda_active})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global is_veda_active, active_session, group_visit_state
    await websocket.accept()
    active_websockets.append(websocket)
    voice_service.has_active_web_audio = True
    print(f"[WS] Client connected ({len(active_websockets)} active). Web audio enabled for 3D avatar.")

    # Immediately report current Veda state to newly connected client
    try:
        await websocket.send_text(json.dumps({"event": "veda_status", "data": {"active": is_veda_active}}))
    except Exception as e:
        print(f"[WS] Failed to send initial status: {e}")

    try:
        while True:
            text = await websocket.receive_text()
            try:
                msg = json.loads(text)
                event_type = msg.get("event")
                if event_type == "audio_mode":
                    voice_service.has_active_web_audio = bool(msg.get("data", {}).get("web_audio", True))
                    print(f"[WS] Audio mode updated: web_audio={voice_service.has_active_web_audio}")
                elif event_type == "toggle_active":
                    req_active = msg.get("data", {}).get("active")
                    if req_active is None:
                        is_veda_active = not is_veda_active
                    else:
                        is_veda_active = bool(req_active)

                    print(f"[WS] Veda active state toggled: {is_veda_active}")
                    await broadcast_event("veda_status", {"active": is_veda_active})

                    if is_veda_active:
                        if active_session is None and not is_processing:
                            print("[WS] Starting Veda group visit session and greeting pipeline...")
                            asyncio.create_task(start_group_session_async([], 0))
                    else:
                        close_camera()
                        active_session = None
                        group_visit_state = "idle"
                        voice_service.is_speaking = False
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
