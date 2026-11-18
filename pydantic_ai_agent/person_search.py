"""
Person search tool using Local Plaksha Dataset + Tavily + Pydantic AI (Groq Model)
"""
import os
import re
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Optional, Dict
from dotenv import load_dotenv
from tavily import TavilyClient
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel

from .models import SearchResultItem, PersonProfile, PlakshaPerson

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

tavily_client: Optional[TavilyClient] = None
if TAVILY_API_KEY:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Dedicated Pydantic AI Agent for profiling and synthesizing search results
profile_agent = Agent(
    GroqModel("openai/gpt-oss-20b"),
    system_prompt=(
        "You compile factual, concise public-information summaries about people for a conversational voice assistant. "
        "Summarize their role, recent work, affiliations, and interesting facts. "
        "Keep it conversational, natural, and never include raw markdown bullets or hyperlinks."
    )
)

# ================= Local Plaksha Website Scraped Dataset =================
plaksha_people_db: List[PlakshaPerson] = []


def clean_person_name(name: str) -> str:
    """Strip Dr., Prof., etc. without damaging root words like Rudra."""
    return re.sub(r"^(dr\.?|prof\.?)\s+", "", name.strip(), flags=re.IGNORECASE).strip().lower()


def load_plaksha_people(file_path: Optional[str] = None) -> List[PlakshaPerson]:
    """Load local Plaksha dataset into memory."""
    global plaksha_people_db
    if plaksha_people_db:
        return plaksha_people_db

    candidates = [
        Path(file_path) if file_path else None,
        Path(__file__).parent / "plaksha_people_dataset.json",
        Path(__file__).parent.parent / "HTI Data" / "plaksha_people_dataset.json",
    ]
    target_path = next((p for p in candidates if p and p.exists()), None)
    if not target_path:
        print("[Dataset] Warning: plaksha_people_dataset.json not found.")
        return []

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            plaksha_people_db = [PlakshaPerson(**item) for item in data]
            print(f"[Dataset] Loaded {len(plaksha_people_db)} Plaksha profiles from {target_path.name}.")
            return plaksha_people_db
    except Exception as e:
        print(f"[Dataset] Error loading {target_path}: {e}")
        return []


def find_plaksha_person(name: str) -> Optional[PlakshaPerson]:
    """Find a person in the local scraped Plaksha dataset by exact, substring, or token match."""
    global plaksha_people_db
    if not plaksha_people_db:
        load_plaksha_people()

    c = clean_person_name(name)
    if not c or c in ["unknown", "unknown visitor", "friend"]:
        return None

    # 1. Exact cleaned match
    for p in plaksha_people_db:
        if clean_person_name(p.name) == c:
            return p

    # 2. Substring match
    for p in plaksha_people_db:
        p_c = clean_person_name(p.name)
        if c in p_c or p_c in c:
            return p

    # 3. Token overlap match (>= 2 common words)
    c_tokens = set(c.split())
    for p in plaksha_people_db:
        p_tokens = set(clean_person_name(p.name).split())
        if len(c_tokens.intersection(p_tokens)) >= 2:
            return p

    return None


# ================= Tavily Web Search =================
def search_person(name: str, context: str = "") -> List[SearchResultItem]:
    """Run a single fast Tavily search and return validated Pydantic models."""
    if not tavily_client:
        print("[Tavily] Warning: TAVILY_API_KEY not configured.")
        return []

    query = f'"{name}" {context}'.strip()
    try:
        resp = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=3,
            include_answer=False,
        )
        raw_results = resp.get("results", [])
        return [
            SearchResultItem(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", "")[:800]
            )
            for r in raw_results
        ]
    except Exception as e:
        print(f"[Tavily] Search failed for '{query}': {e}")
        return []


async def synthesize_async(name: str, results: List[SearchResultItem]) -> str:
    """Async synthesis using Pydantic AI Agent."""
    if not results:
        return ""

    context_block = "\n\n".join(
        f"Source: {r.url}\nTitle: {r.title}\nContent: {r.content}"
        for r in results
    )

    prompt = (
        f'Here is information found online about "{name}":\n\n'
        f'{context_block}\n\n'
        f'Provide a 2-3 sentence conversational brief about them for greeting and conversation.'
    )

    try:
        run_result = await profile_agent.run(prompt)
        return run_result.output
    except Exception as e:
        print(f"[ProfileAgent] Synthesis error: {e}")
        return ""


def synthesize(name: str, results: List[SearchResultItem]) -> str:
    """Synchronous wrapper for synthesize_async."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                return executor.submit(asyncio.run, synthesize_async(name, results)).result()
        return asyncio.run(synthesize_async(name, results))
    except Exception:
        return asyncio.run(synthesize_async(name, results))


async def search_and_summarize_async(name: str, context: str = "") -> str:
    """Async search via Tavily and synthesis with Pydantic AI without blocking the event loop."""
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, search_person, name, context)
    if not results:
        return ""
    return await synthesize_async(name, results)


def search_and_summarize(name: str, context: str = "") -> str:
    """High-level synchronous helper for searching and summarizing a person."""
    results = search_person(name, context)
    if not results:
        return ""
    return synthesize(name, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for a person and summarize public info.")
    parser.add_argument("name", help="Full name to search for")
    parser.add_argument("--context", default="", help="Extra context to disambiguate")
    args = parser.parse_args()

    print(f"Checking local Plaksha dataset for: {args.name}")
    local_p = find_plaksha_person(args.name)
    if local_p:
        print("[Local Match Found!]")
        print(local_p.to_conversational_summary())
        print()

    print(f"Searching Tavily for: {args.name} ({args.context or 'no extra context'})\n")
    res = search_person(args.name, args.context)
    print(f"Found {len(res)} results. Synthesizing with Pydantic AI...")
    summary = synthesize(args.name, res)
    print("=" * 60)
    print(summary)
    print("=" * 60)
