"""
Edge TTS voice synthesis and pipelined audio playback module.
"""
import os
import re
import time
import tempfile
import asyncio
import subprocess
from typing import Optional, Callable, Awaitable, List
import edge_tts


def _chunk_sentences(text: str, min_chars: int = 45) -> List[str]:
    """
    Split text into natural conversational chunks.
    Avoids fragmenting short exclamations or phrases (e.g. 'Namaste!', 'Hey Maan,')
    into tiny isolated clips that cause awkward pauses between lines.
    """
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not raw_sentences:
        return [text.strip()]

    chunks: List[str] = []
    current = ""
    for s in raw_sentences:
        if not current:
            current = s
        elif len(current) < min_chars:
            current += " " + s
        else:
            chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks


import base64


def get_audio_duration(file_path: str) -> float:
    """Accurately compute MP3 duration pure-python without subprocess overhead (Edge-TTS 48kbps = 6000 B/s)."""
    try:
        size = os.path.getsize(file_path)
        return max(0.4, round(size / 6000.0, 2))
    except Exception:
        return 2.0


class VoiceService:
    def __init__(self, voice: str = "en-IN-NeerjaNeural", rate: str = "+8%"):
        self.voice = voice
        self.rate = rate
        self.is_speaking = False
        # Flag indicating whether connected web clients are handling audio playback
        self.has_active_web_audio = False

    async def speak(
        self,
        text: str,
        broadcast_callback: Optional[Callable[[str, dict], Awaitable[None]]] = None
    ) -> None:
        """
        Synthesize text using edge-tts and play it.
        Pipelined with parallel prefetching:
        - Chunk 0 starts playing within ~1s.
        - Broadcasts audio chunks to web clients via WebSocket as base64 for 3D avatar lip sync.
        - If web client is active, browser plays audio and server sleeps for duration.
        - If no web client is connected, afplay plays audio locally on the host machine.
        """
        if not text or not text.strip():
            print("[TTS] Empty text received. Skipping.")
            return

        print(f"[TTS] Generating audio for: {text}")
        tts_start = time.time()

        chunks = _chunk_sentences(text)
        self.is_speaking = True
        if broadcast_callback:
            await broadcast_callback("state_change", {"state": "speaking", "text": "AI Speaking..."})

        temp_files: List[str] = []

        try:
            async def generate_chunk_to_file(chunk_text: str) -> str:
                tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                tmp.close()
                # Auto-select Hindi voice if Devanagari characters are detected
                voice = "hi-IN-SwaraNeural" if bool(re.search(r"[\u0900-\u097F]", chunk_text)) else self.voice
                c = edge_tts.Communicate(chunk_text, voice, rate=self.rate)
                await c.save(tmp.name)
                return tmp.name

            async def play_or_wait_audio(file_path: str, chunk_text: str):
                duration = get_audio_duration(file_path)
                # Broadcast audio chunk to web client for 3D avatar Web Audio API & lip sync
                if broadcast_callback:
                    try:
                        with open(file_path, "rb") as af:
                            audio_data = af.read()
                        b64_audio = base64.b64encode(audio_data).decode("ascii")
                        await broadcast_callback("speak_audio", {
                            "audio_base64": b64_audio,
                            "text": chunk_text,
                            "duration": duration
                        })
                    except Exception as b_err:
                        print(f"[TTS] Broadcast speak_audio error: {b_err}")

                if self.has_active_web_audio:
                    # Browser is actively playing audio and driving lip-sync; sleep for exact duration
                    await asyncio.sleep(duration)
                else:
                    # No web client connected, play via macOS system audio
                    await asyncio.to_thread(subprocess.run, ["afplay", file_path], check=True)

            if len(chunks) == 1:
                # Single chunk: generate and play directly
                fpath = await generate_chunk_to_file(chunks[0])
                temp_files.append(fpath)
                first_ready = int((time.time() - tts_start) * 1000)
                print(f"[TIMING] TTS audio ready in: {first_ready}ms")
                await play_or_wait_audio(fpath, chunks[0])
            else:
                # Multi-chunk: generate chunk 0 for instant start, while launching all remaining in parallel
                chunk0_task = asyncio.create_task(generate_chunk_to_file(chunks[0]))
                rest_tasks = [asyncio.create_task(generate_chunk_to_file(c)) for c in chunks[1:]]

                # Wait for Chunk 0 to start speaking immediately
                fpath0 = await chunk0_task
                temp_files.append(fpath0)
                first_ready = int((time.time() - tts_start) * 1000)
                print(f"[TIMING] TTS first audio ready in: {first_ready}ms")

                # Play Chunk 0 while gathering and concatenating all remaining chunks
                async def prepare_rest_stream() -> str:
                    part_files = await asyncio.gather(*rest_tasks)
                    temp_files.extend(part_files)
                    combined_tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                    combined_tmp.close()
                    # Directly concatenate valid MP3 bitstreams for zero-pause audio
                    with open(combined_tmp.name, "wb") as fout:
                        for pf in part_files:
                            with open(pf, "rb") as fin:
                                fout.write(fin.read())
                    temp_files.append(combined_tmp.name)
                    return combined_tmp.name

                rest_prep_task = asyncio.create_task(prepare_rest_stream())

                # Play chunk 0
                await play_or_wait_audio(fpath0, chunks[0])

                # Rest is already prepared during chunk 0 playback, play immediately with zero pause
                combined_path = await rest_prep_task
                rest_text = " ".join(chunks[1:])
                await play_or_wait_audio(combined_path, rest_text)

            total_elapsed = int((time.time() - tts_start) * 1000)
            print(f"[TIMING] Total TTS and playback completed: {total_elapsed}ms")

        except Exception as e:
            print(f"[TTS] Error during edge-tts playback: {e}")
            # Ensure we never spawn a default male voice
            try:
                # If native fallback is attempted, strictly use female voice
                subprocess.run(["say", "-v", "Samantha", text], check=False)
            except Exception as fallback_err:
                print(f"[TTS] Fallback error: {fallback_err}")
        finally:
            self.is_speaking = False
            for tf in temp_files:
                if os.path.exists(tf):
                    try:
                        os.unlink(tf)
                    except Exception:
                        pass
            if broadcast_callback:
                await broadcast_callback("state_change", {"state": "active", "text": "Listening..."})
