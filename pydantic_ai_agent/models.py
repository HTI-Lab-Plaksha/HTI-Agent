"""
Pydantic data models and schemas for HTI Lab Agent (Veda).
"""
import os
from typing import Optional, List, Dict, Any, Callable, Union
from pydantic import BaseModel, Field


class SearchResultItem(BaseModel):
    """Single search result from Tavily."""
    title: str = ""
    url: str = ""
    content: str = ""


class PersonProfile(BaseModel):
    """Synthesized profile of a person."""
    name: str
    summary: str
    sources: List[str] = Field(default_factory=list)


class WebSocketEvent(BaseModel):
    """Schema for WebSocket events sent to client dashboard."""
    event: str
    data: Dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        return self.model_dump_json()


class AgentDeps(BaseModel):
    """Dependencies injected into Pydantic AI Veda Agent."""
    visitor_name: str = "Unknown Visitor"
    session_context: str = ""
    # Optional event emitter callable to notify UI in real-time
    broadcast_callback: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True


class PlakshaPerson(BaseModel):
    """Clean person profile scraped directly from plaksha.edu.in."""
    name: str
    category: str = ""
    designation: str = ""
    email: str = ""
    source_url: str = ""
    image_url: str = ""
    image_filename: str = ""
    page_title: str = ""
    meta_description: str = ""
    education: Union[str, List[str]] = ""
    about: str = ""
    research_summary: str = ""
    research_interests: List[str] = Field(default_factory=list)
    publications_sample: List[str] = Field(default_factory=list)
    courses: List[str] = Field(default_factory=list)
    listed_under: List[str] = Field(default_factory=list)

    def to_conversational_summary(self) -> str:
        """Format a rich, concise summary for instant local greeting & conversational context."""
        parts = []
        role = self.page_title or self.designation
        if role:
            parts.append(f"Role: {role}")
        if self.about:
            # First 2 sentences of about
            sentences = [s.strip() for s in self.about.split(".") if s.strip()]
            parts.append(f"Background: {'. '.join(sentences[:2])}.")
        if self.research_summary:
            parts.append(f"Research: {self.research_summary}")
        if self.education:
            clean_edu = ", ".join(self.education) if isinstance(self.education, list) else str(self.education)
            clean_edu = clean_edu.replace("&amp;", "&")
            parts.append(f"Education: {clean_edu}")
        return " | ".join(parts)


class AppConfig(BaseModel):
    """Configuration settings for HTI Lab Agent."""
    face_distance_threshold: float = 0.5
    consecutive_frames: int = 2
    process_nth_frame: int = 2
    camera_index: int = 0
    enable_camera: bool = os.getenv("ENABLE_CAMERA", "0").lower() in ("1", "true", "yes")
    
    stt_model: str = "whisper-large-v3"
    llm_model: str = "openai/gpt-oss-20b"
    edge_tts_voice: str = "en-IN-NeerjaNeural"
    edge_tts_rate: str = "+8%"
    
    audio_rate: int = 16000
    audio_chunk: int = 1024
    silence_threshold: float = 500.0
    silence_duration: float = 0.6
    min_speech_duration: float = 0.5
    inactivity_timeout_seconds: float = 20.0
