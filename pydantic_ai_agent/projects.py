"""
HTI Lab Demonstrations and Projects Knowledge Base.
Loads and queries project information extracted from HTI Lab posters and research slides.
"""
import os
import csv
from pathlib import Path
from typing import List, Dict, Optional

_PROJECTS_CACHE: Optional[List[Dict[str, str]]] = None


def get_csv_path() -> Path:
    """Locate hti_lab_projects.csv across common directory layouts."""
    candidates = [
        Path(__file__).parent / "hti_lab_projects.csv",
        Path(__file__).parent.parent / "HTI Data" / "hti_lab_projects.csv",
        Path(__file__).parent.parent / "HTI Data" / "2nd_csv_details" / "hti_lab_projects.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def load_hti_projects() -> List[Dict[str, str]]:
    """Load HTI lab projects from CSV into a list of dictionaries."""
    global _PROJECTS_CACHE
    if _PROJECTS_CACHE is not None:
        return _PROJECTS_CACHE

    csv_path = get_csv_path()
    projects = []
    if csv_path.exists():
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                projects.append(dict(row))
        _PROJECTS_CACHE = projects
        print(f"[PROJECTS] Loaded {len(projects)} HTI lab projects from {csv_path.name}.")
    else:
        print(f"[PROJECTS] Warning: Projects CSV not found at {csv_path}")
        _PROJECTS_CACHE = []

    return _PROJECTS_CACHE


def find_hti_project(query: str) -> Optional[Dict[str, str]]:
    """
    Look up a project by acronym, name, or keywords.
    E.g. 'atlas', 'guide ai', 'pose pilot', 'hindi', 'posture', 'stress'.
    """
    projects = load_hti_projects()
    q = query.strip().lower()

    # Exact or substring match on project_id or project_name
    for p in projects:
        p_id = p.get("project_id", "").lower()
        p_name = p.get("project_name", "").lower()
        if q == p_id or q == p_name or q in p_name or p_name in q:
            return p

    # Match in keywords or full title
    for p in projects:
        keywords = p.get("keywords", "").lower()
        title = p.get("full_title", "").lower()
        if q in keywords or q in title:
            return p

    return None


def get_projects_overview_for_prompt() -> str:
    """
    Return a concise summary of all HTI lab demonstrations/projects
    suitable for injecting directly into the conversational system prompt.
    """
    projects = load_hti_projects()
    if not projects:
        return ""

    lines = ["HTI LAB DEMONSTRATIONS & KEY RESEARCH PROJECTS:"]
    for p in projects:
        name = p.get("project_name", "")
        pitch = p.get("conversational_voice_pitch_1_sentence", "")
        domain = p.get("research_domain", "").split(",")[0]
        lines.append(f"- {name} ({domain}): {pitch}")

    lines.append(
        "When visitors ask about the lab's projects or demonstrations, you can introduce these highlights naturally. "
        "Keep project explanations snappy (1-2 sentences for voice). Never recite long technical paragraphs."
    )
    return "\n".join(lines)
