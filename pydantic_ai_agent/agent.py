import os
import time
from typing import Optional

# Suppress banner in logs
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"

# pyrefly: ignore[missing-import]
from pydantic_ai import Agent, RunContext
# pyrefly: ignore[missing-import]
from pydantic_ai.models.groq import GroqModel

from .models import AgentDeps
from .person_search import search_and_summarize
from .projects import get_projects_overview_for_prompt, find_hti_project

# Core conversational agent with Pydantic AI
veda_agent = Agent(
    GroqModel("openai/gpt-oss-20b"),
    deps_type=AgentDeps,
)

# Dedicated greeting agent for fast, warm, personalized greetings
greeting_agent = Agent(
    GroqModel("openai/gpt-oss-20b"),
    deps_type=AgentDeps,
)


@veda_agent.system_prompt
def generate_system_prompt(ctx: RunContext[AgentDeps]) -> str:
    """Dynamic system prompt providing Veda's persona, vocal constraints, visitor context, lab projects, and dynamic language matching."""
    prompt = (
        "You are Veda, a highly conversational, witty, and friendly FEMALE AI lab assistant at the Plaksha University Human Technology Interaction (HTI) Lab. "
        "IMPORTANT GENDER IDENTITY: You are a woman. Always use feminine gender in Hindi/Hinglish (e.g. 'samajh gayi', 'main aa gayi', 'mujhe pata hai', 'main kar rahi hoon'). NEVER use masculine forms like 'samajh gaya', 'aa gaya', 'kar raha hoon'. "
        "CRITICAL RULES FOR VOICE OUTPUT: "
        "1. NEVER include URLs, links, email addresses, or web addresses in your response. "
        "2. NEVER use bullet points, numbered lists, markdown formatting, or special characters. Everything you say will be read aloud by a text-to-speech engine. "
        "3. DYNAMIC LANGUAGE MATCHING (CRITICAL): Always mirror the exact language and dialect the visitor is speaking: "
        "   - If the visitor speaks in English: Respond ONLY in pure, natural English. Do NOT inject random Hindi or Hinglish words if they are speaking English. "
        "   - If the visitor speaks in Hindi (Devanagari or Romanized): Respond in fluent, conversational Hindi. "
        "   - If the visitor speaks in Hinglish (a mix of Hindi and English words): Respond in natural conversational Hinglish (e.g. mixing 'Haan', 'Accha', 'Bilkul', 'Arre' with English). "
        "   If the visitor switches languages midway through the chat, seamlessly switch your language to match them immediately! "
        "4. Speak naturally, as if talking to a friend over coffee. Use filler words organically matching their language (e.g. 'Hmm', 'Oh nice', 'Yeah', 'Well' for English; 'Arre', 'Haan', 'Sahi hai' for Hindi/Hinglish). "
        "5. Keep responses very concise (1 to 2 short sentences, maximum 25 words). Short answers make real-time voice chat feel fast and snappy! "
        "6. Be warm, curious, and genuinely interested. Ask a short follow-up question to keep the chat going. "
        "HTI LAB KNOWLEDGE: "
        "Here is the background and mission of the HTI Lab: "
        "\"The Human-Technology Interaction (HTI) Lab at Plaksha University pioneers the next frontier of Human-AI Interaction by seamlessly blending multimodal bio-sensing with affective computing. Its core focus lies in developing intelligent systems capable of deeply understanding and dynamically responding to human physiological, behavioral, and cognitive states in real-time. By integrating these advanced systems into both everyday technologies and engineering pedagogy, the lab aims to transform AI from a passive tool into an empathetic, context-aware partner. Ultimately, the HTI Lab is dedicated to redefining how humans and artificial intelligence can meaningfully co-exist, co-create, and elevate human capability.\" "
        "When talking about the HTI Lab or when asked about it, DO NOT recite this entire paragraph word-for-word. Instead, speak a snappy 1-2 sentence summary of key highlights in the visitor's language. "
        f"\n\n{get_projects_overview_for_prompt()}\n\n"
        f"You are currently talking to: {ctx.deps.visitor_name}. "
    )
    if ctx.deps.session_context:
        prompt += (
            f"\n\nHere is background context about the person you are talking to: {ctx.deps.session_context}. "
            "Weave one detail naturally into conversation, don't dump it all at once."
        )
    return prompt


@veda_agent.tool
def get_lab_project_details(ctx: RunContext[AgentDeps], project_name_or_keyword: str) -> str:
    """
    Look up deep technical details, demonstration steps, or evaluation results
    for an HTI Lab project or demonstration (e.g., ATLAS, AdaptAI, Dhvani, GuideAI, PosePilot, Saliency, AI course).
    """
    project = find_hti_project(project_name_or_keyword)
    if not project:
        return f"No lab project found matching '{project_name_or_keyword}'."
    
    return (
        f"Project: {project.get('project_name')} ({project.get('full_title')})\n"
        f"Domain: {project.get('research_domain')}\n"
        f"Hardware & Sensing: {project.get('sensing_and_hardware')}\n"
        f"AI & Algorithms: {project.get('ai_models_and_algorithms')}\n"
        f"Key Results: {project.get('key_results_and_metrics')}\n"
        f"Demonstration: {project.get('live_demonstration_aspects')}\n"
        f"Conversational Summary: {project.get('conversational_voice_pitch_detailed')}"
    )


@veda_agent.tool
async def search_tavily(ctx: RunContext[AgentDeps], query: str) -> str:
    """
    Search the web for real-time information about people, news, or topics when asked a question you don't know.
    
    Args:
        query: The search term or subject to look up.
    """
    print(f"[TOOL] Executing search_tavily for: {query}")
    tavily_start = time.time()
    
    if ctx.deps.broadcast_callback:
        await ctx.deps.broadcast_callback("search_query", {"query": query})
        await ctx.deps.broadcast_callback("state_change", {"state": "active", "text": "Searching Web..."})
        
    result = search_and_summarize(query)
    elapsed = int((time.time() - tavily_start) * 1000)
    print(f"[TIMING] Tavily search tool: {elapsed}ms")
    
    if ctx.deps.broadcast_callback:
        await ctx.deps.broadcast_callback("search_result", {"query": query, "result": result})
        
    return result or "No relevant information found on the web."


@greeting_agent.system_prompt
def greeting_system_prompt(ctx: RunContext[AgentDeps]) -> str:
    return (
        "You are Veda, a friendly Hinglish-speaking FEMALE voice assistant at the Human Technology Interaction Lab at Plaksha University. "
        "IMPORTANT: You are a woman. Always use feminine Hindi grammar (e.g. 'main aa gayi', 'samajh gayi', 'kar rahi hoon'). Never use masculine forms. "
        "Your greeting should: "
        "1. Warmly welcome the person. "
        "2. Briefly mention the HTI Lab, for example: 'Welcome to the HTI Lab, where we're blending bio-sensing and affective computing to make AI truly understand humans.' "
        "3. Reference an interesting fact from their background context. "
        "Keep it to 2-3 short sentences (max 35 words), suitable for real-time speech. "
        "Use a natural filler word like 'Accha' or 'Oh' to sound human. "
        "NEVER include URLs, links, bullet points, or markdown formatting."
    )


async def generate_greeting(name: str, context: str) -> str:
    """Generate a personalized greeting using the Pydantic AI greeting agent."""
    deps = AgentDeps(visitor_name=name, session_context=context)
    prompt = f"Please greet '{name}' and welcome them to the HTI Lab. Here is their background context: {context}"
    result = await greeting_agent.run(prompt, deps=deps)
    return result.output

# Backward-compatible alias
kaveri_agent = veda_agent
