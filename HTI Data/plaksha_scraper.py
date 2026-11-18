#!/usr/bin/env python3
"""
plaksha_scraper.py - crawl people profiles from https://plaksha.edu.in

Categories
    Founding Member | Founder & Distinguished Trustee | Founder & Trustee | Founder |
    Full Time Faculty | Visiting Faculty | Adjunct Faculty | Guest Faculty

Site structure this script is built around (server-rendered HTML, paginated):
    /founding-members                       (all founding members, ?/page/N)
    /founding-members/distinguished-trustees | founders-trustees | founders
    /faculty/full-time | visiting | adjunct | guest      (each with /page/N)
    profile pages:  /faculty/<slug>   and   /founding-members/<category>/<slug>

Outputs (in --out-dir, default ./output)
    plaksha_people_dataset.json      nested, indented UTF-8
    all_plaksha_people_master.csv    flattened (lists joined with " | ")
    scrape_report.json               counts, broken links, missing images/fields
    plaksha_scrape.log               full log

Usage
    python plaksha_scraper.py                      # plain HTTP (fast, default)
    python plaksha_scraper.py --render             # use Playwright/Chromium for every page
    python plaksha_scraper.py --limit 5 -v         # smoke test: 5 profiles per listing
    python plaksha_scraper.py --cache-dir .cache   # cache HTML so re-runs are instant
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import sys
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE = "https://plaksha.edu.in"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": BASE + "/",
}

# (label, listing url). Specific founder sub-lists first, then the umbrella list.
LISTINGS = [
    ("Founder & Distinguished Trustee", f"{BASE}/founding-members/distinguished-trustees"),
    ("Founder & Trustee", f"{BASE}/founding-members/founders-trustees"),
    ("Founder", f"{BASE}/founding-members/founders"),
    ("Founding Member", f"{BASE}/founding-members"),
    ("Full Time Faculty", f"{BASE}/faculty/full-time"),
    ("Visiting Faculty", f"{BASE}/faculty/visiting"),
    ("Adjunct Faculty", f"{BASE}/faculty/adjunct"),
    ("Guest Faculty", f"{BASE}/faculty/guest"),
]
CATEGORY_ORDER = [c for c, _ in LISTINGS]
FACULTY_CATEGORIES = {"Full Time Faculty", "Visiting Faculty", "Adjunct Faculty", "Guest Faculty"}

# URL segment under /founding-members/<segment>/<slug>
FOUNDING_SEGMENT_CATEGORY = {
    "distinguished-trustees": "Founder & Distinguished Trustee",
    "founders-trustees": "Founder & Trustee",
    "founders": "Founder",
}
IGNORED_FOUNDING_SEGMENTS = {"corporate-donors", "changemakers"}  # not requested
RESERVED_FACULTY_SLUGS = {"full-time", "visiting", "adjunct", "guest", "page"}

FIELDS = [
    "name", "category", "designation", "email", "source_url", "image_url", "image_filename",
    "linkedin", "other_links", "education", "about", "research_summary",
    "research_interests", "publications_sample", "courses", "listed_under",
]
LIST_FIELDS = {"education", "research_interests", "publications_sample", "courses", "listed_under"}

# Master CSV column layout (matching legacy 2nd_csv_details/all_plaksha_people_master.csv + new rich columns)
CSV_COLUMNS_MAPPING = [
    ("Source_URL", "source_url"),
    ("Name", "name"),
    ("Category", "category"),
    ("Designation", "designation"),
    ("Email", "email"),
    ("LinkedIn", "linkedin"),
    ("Other_Links", "other_links"),
    ("Bio", "about"),
    ("Image_URL", "image_url"),
    ("Image_Filename", "image_filename"),
    ("Education", "education"),
    ("Research_Summary", "research_summary"),
    ("Research_Interests", "research_interests"),
    ("Publications_Sample", "publications_sample"),
    ("Courses", "courses"),
    ("Listed_Under", "listed_under"),
]
MASTER_CSV_COLUMNS = [col for col, _ in CSV_COLUMNS_MAPPING]

log = logging.getLogger("plaksha")


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
_QUOTES = {"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
           "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"'}
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)


def clean_text(s: Optional[str]) -> str:
    """Normalise whitespace, NBSP, zero-width chars and typographic quotes."""
    if not s:
        return ""
    s = s.translate(_ZERO_WIDTH).replace("\xa0", " ").replace("\u2009", " ").replace("\u202f", " ")
    for k, v in _QUOTES.items():
        s = s.replace(k, v)
    return re.sub(r"\s+", " ", s).strip()


def clean_block(s: Optional[str]) -> str:
    """Like clean_text but keeps paragraph breaks (used for multi-paragraph fields)."""
    if not s:
        return ""
    paras = [clean_text(p) for p in re.split(r"\n\s*\n", s)]
    return "\n\n".join(p for p in paras if p)


_PREFIX_RE = re.compile(r"^((dr|prof|professor|mr|mrs|ms|shri|smt)\b\.?\s*)+", re.I)


def clean_name(raw: str) -> str:
    n = clean_text(raw)
    n = _PREFIX_RE.sub("", n)
    n = re.sub(r"\s*,.*$", "", n)  # drop ", PhD" style suffixes
    if n.isupper() or n.islower():
        n = n.title()
    return n.strip()


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
URL_ONLY_RE = re.compile(r"^(https?://)?[\w.\-]+\.[a-z]{2,}(/\S*)?$", re.I)
DEGREE_RE = re.compile(
    r"\b(ph\.?d|m\.?tech|b\.?tech|mba|emba|m\.?sc?|b\.?sc?|b\.?e|m\.?e|mphil|opm|pgdbm|"
    r"bcom|b\.?a|m\.?a|llb|llm|bachelor|master|doctor|diploma|postdoc)\b", re.I)


def dedupe(seq):
    seen, out = set(), []
    for x in seq:
        k = x.lower()
        if x and k not in seen:
            seen.add(k)
            out.append(x)
    return out


def clean_filename(name: str) -> str:
    """Remove special characters and replace spaces with underscores for filenames."""
    cleaned = re.sub(r'[^a-zA-Z0-9\s-]', '', name)
    return cleaned.strip().replace(' ', '_')


def download_image_file(session: requests.Session, url: str, person_name: str, images_dir: Path) -> str:
    """Download image to images_dir/<clean_name>.<ext> and return filename."""
    if not url:
        return ""
    clean_n = clean_filename(person_name)
    if not clean_n:
        return ""
    ext = ".jpg"
    u_low = url.lower()
    if ".png" in u_low:
        ext = ".png"
    elif ".jpeg" in u_low:
        ext = ".jpeg"
    elif ".webp" in u_low:
        ext = ".webp"

    img_filename = f"{clean_n}{ext}"
    target_path = images_dir / img_filename
    if target_path.exists() and target_path.stat().st_size > 0:
        return img_filename

    try:
        r = session.get(url, headers=HEADERS, timeout=15, stream=True)
        if r.status_code == 200:
            with open(target_path, "wb") as f:
                for chunk in r.iter_content(4096):
                    f.write(chunk)
            return img_filename
    except Exception as e:
        log.warning("Could not download image for %s from %s: %s", person_name, url, e)
    return ""


def load_existing_master_csv(csv_path: Path) -> dict[str, dict]:
    """Load existing master CSV indexed by lowercased name and by source_url."""
    existing = {}
    if not csv_path.exists():
        return existing
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                name_k = clean_name(row.get("Name", "")).lower()
                url_k = (row.get("Source_URL") or "").strip().lower()
                if name_k:
                    existing[name_k] = row
                if url_k:
                    existing[url_k] = row
    except Exception as e:
        log.warning("Could not load existing CSV from %s: %s", csv_path, e)
    return existing


def flatten_to_master_row(rec: dict, existing_row: Optional[dict] = None) -> dict:
    """Flatten structured record into legacy master CSV row layout with all columns preserved."""
    row = {}
    for col_name, field_key in CSV_COLUMNS_MAPPING:
        val = rec.get(field_key, "")
        if field_key in LIST_FIELDS:
            if isinstance(val, list):
                val = " | ".join(val)
        val = "" if val is None else str(val).strip()

        # If empty in scraped data, fall back to existing master CSV value so nothing is lost
        if not val and existing_row:
            val = str(existing_row.get(col_name, "")).strip()

        row[col_name] = val
    return row


# --------------------------------------------------------------------------- #
# HTTP / Playwright fetcher
# --------------------------------------------------------------------------- #
class Fetcher:
    def __init__(self, min_delay=0.5, max_delay=1.0, retries=5, render=False,
                 cache_dir: Optional[Path] = None, respect_robots=True):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.min_delay, self.max_delay = min_delay, max_delay
        self.retries = retries
        self.render = render
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.respect_robots = respect_robots
        self._robots: Optional[robotparser.RobotFileParser] = None
        self._last = 0.0
        self._pw = self._browser = self._ctx = None
        self.failures: dict[str, str] = {}  # url -> reason

    # -- politeness --------------------------------------------------------- #
    def _throttle(self):
        wait = random.uniform(self.min_delay, self.max_delay)
        elapsed = time.monotonic() - self._last
        if elapsed < wait:
            time.sleep(wait - elapsed)
        self._last = time.monotonic()

    def _allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        if self._robots is None:
            self._robots = robotparser.RobotFileParser()
            try:
                r = self.session.get(BASE + "/robots.txt", timeout=15)
                self._robots.parse(r.text.splitlines() if r.status_code == 200 else [])
            except requests.RequestException:
                self._robots.parse([])
        return self._robots.can_fetch(USER_AGENT, url)

    def _backoff(self, attempt: int, retry_after: Optional[str] = None):
        if retry_after and retry_after.isdigit():
            delay = min(int(retry_after), 60)
        else:
            delay = min(2 ** attempt, 60) + random.uniform(0, 1)
        log.warning("  backing off %.1fs (attempt %d/%d)", delay, attempt, self.retries)
        time.sleep(delay)

    # -- backends ----------------------------------------------------------- #
    def _fetch_http(self, url: str, referer: Optional[str]) -> tuple[Optional[str], str]:
        reason = "unknown"
        for attempt in range(1, self.retries + 1):
            self._throttle()
            try:
                r = self.session.get(url, headers={"Referer": referer or BASE + "/"}, timeout=(10, 30))
            except (requests.Timeout, requests.ConnectionError) as e:
                reason = type(e).__name__
                log.warning("  %s on %s", reason, url)
                self._backoff(attempt)
                continue
            if r.status_code == 200:
                return r.content.decode("utf-8", errors="replace"), ""
            reason = f"HTTP {r.status_code}"
            if r.status_code in (404, 410):
                return None, reason
            if r.status_code in (429, 500, 502, 503, 504):
                self._backoff(attempt, r.headers.get("Retry-After"))
                continue
            return None, reason
        return None, f"gave up after {self.retries} attempts ({reason})"

    def _start_browser(self):
        if self._browser:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise RuntimeError("Playwright not installed: pip install playwright && playwright install chromium")
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._ctx = self._browser.new_context(user_agent=USER_AGENT, locale="en-IN")

    def _fetch_render(self, url: str, referer: Optional[str]) -> tuple[Optional[str], str]:
        self._start_browser()
        reason = "unknown"
        for attempt in range(1, self.retries + 1):
            self._throttle()
            page = self._ctx.new_page()
            try:
                if referer:
                    page.set_extra_http_headers({"Referer": referer})
                resp = page.goto(url, wait_until="networkidle", timeout=45_000)
                status = resp.status if resp else 0
                if status == 200:
                    return page.content(), ""
                reason = f"HTTP {status}"
                if status in (404, 410):
                    return None, reason
                self._backoff(attempt)
            except Exception as e:  # playwright timeouts etc.
                reason = type(e).__name__
                log.warning("  %s on %s", reason, url)
                self._backoff(attempt)
            finally:
                page.close()
        return None, f"gave up after {self.retries} attempts ({reason})"

    # -- public ------------------------------------------------------------- #
    def get(self, url: str, referer: Optional[str] = None, *, force_render=False,
            record_failure=True) -> Optional[str]:
        cache_file = None
        if self.cache_dir and not force_render:
            cache_file = self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".html")
            if cache_file.exists():
                return cache_file.read_text("utf-8")
        if not self._allowed(url):
            self.failures[url] = "disallowed by robots.txt"
            log.error("robots.txt disallows %s", url)
            return None
        use_render = self.render or force_render
        html, reason = (self._fetch_render if use_render else self._fetch_http)(url, referer)
        if html is None:
            if record_failure:
                self.failures[url] = reason
                log.error("FAILED %s -> %s", url, reason)
            return None
        if cache_file:
            cache_file.write_text(html, "utf-8")
        return html

    def head_ok(self, url: str) -> tuple[bool, str]:
        """Check an asset URL (used for image validation)."""
        try:
            self._throttle()
            r = self.session.head(url, headers={"Referer": BASE + "/"}, allow_redirects=True, timeout=20)
            if r.status_code in (403, 405, 501):  # some CDNs reject HEAD
                r = self.session.get(url, headers={"Referer": BASE + "/"}, stream=True, timeout=20)
                r.close()
            return (r.status_code < 400), f"HTTP {r.status_code}"
        except requests.RequestException as e:
            return False, type(e).__name__

    def close(self):
        for obj in (self._ctx, self._browser):
            try:
                obj and obj.close()
            except Exception:
                pass
        try:
            self._pw and self._pw.stop()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Listing pages
# --------------------------------------------------------------------------- #
@dataclass
class Stub:
    url: str
    name: str = ""
    designation: str = ""
    expertise: str = ""
    education: str = ""
    email: str = ""
    image: str = ""


def soup_of(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def best_img_src(img, base: str) -> str:
    """Largest candidate from srcset, else data-src / src."""
    best, best_w = "", -1.0
    for attr in ("srcset", "data-srcset"):
        for part in (img.get(attr) or "").split(","):
            bits = part.strip().split()
            if not bits:
                continue
            w = 0.0
            if len(bits) > 1:
                m = re.match(r"([\d.]+)[wx]", bits[1])
                w = float(m.group(1)) if m else 0.0
            if w > best_w:
                best, best_w = bits[0], w
    if not best:
        best = img.get("data-src") or img.get("data-lazy-src") or img.get("src") or ""
    return urljoin(BASE, best) if best and not best.startswith("data:") else ""


def classify_profile_url(url: str) -> Optional[tuple[str, Optional[str]]]:
    """Return (section, founding_segment) if URL is an individual profile, else None."""
    p = urlparse(url)
    if p.netloc and p.netloc != urlparse(BASE).netloc:
        return None
    parts = [x for x in p.path.split("/") if x]
    if len(parts) == 2 and parts[0] == "faculty" and parts[1] not in RESERVED_FACULTY_SLUGS:
        return "faculty", None
    if parts and parts[0] == "founding-members":
        if len(parts) == 3 and parts[1] != "page":
            return "founding", parts[1]
        if len(parts) == 2 and parts[1] not in FOUNDING_SEGMENT_CATEGORY and parts[1] not in (
                "page", *IGNORED_FOUNDING_SEGMENTS):
            return "founding", None
    return None


def parse_listing(html: str, page_url: str) -> list[Stub]:
    soup = soup_of(html)
    stubs: "OrderedDict[str, Stub]" = OrderedDict()
    for a in soup.select("h2 a[href], h3 a[href], h4 a[href]"):
        url = urljoin(page_url, a["href"]).split("#")[0].split("?")[0]
        if not classify_profile_url(url) or url in stubs:
            continue
        card = a.find_parent("li") or a.find_parent("article") or a.parent.parent
        name = clean_text(a.get_text())
        lines = []
        for line in card.get_text("\n").split("\n"):
            line = clean_text(line)
            if line and line != name and line.lower() != "view profile":
                lines.append(line)
        s = Stub(url=url, name=name)
        for line in lines:
            if line.lower().startswith("expertise:"):
                s.expertise = clean_text(line.split(":", 1)[1])
            elif EMAIL_RE.fullmatch(line):
                s.email = line.lower()
            elif not s.designation:
                s.designation = line
            elif not s.education:
                s.education = line
        mail = card.select_one('a[href^="mailto:"]')
        if mail and not s.email:
            s.email = mail["href"][7:].split("?")[0].strip().lower()
        img = card.find("img")
        if img:
            s.image = best_img_src(img, page_url)
        stubs[url] = s
    return list(stubs.values())


def detect_max_page(html: str) -> Optional[int]:
    nums = [int(n) for n in re.findall(r"/page/(\d+)", html)]
    m = re.search(r"Page\s+\d+\s+of\s+(\d+)", html)
    if m:
        nums.append(int(m.group(1)))
    return max(nums) if nums else None


def crawl_listing(f: Fetcher, label: str, url: str, limit: Optional[int]) -> list[Stub]:
    stubs: "OrderedDict[str, Stub]" = OrderedDict()
    page, max_page, prev_url = 1, None, None
    while page <= 100:
        page_url = url if page == 1 else f"{url}/page/{page}"
        html = f.get(page_url, referer=prev_url or BASE + "/", record_failure=(page == 1 or max_page is not None))
        if html is None:
            break
        if page == 1:
            max_page = detect_max_page(html)
        new = [s for s in parse_listing(html, page_url) if s.url not in stubs]
        if page == 1 and not new and not f.render:
            log.warning("No profiles found on %s via plain HTTP; retrying with Playwright", page_url)
            try:
                html = f.get(page_url, force_render=True)
                new = [s for s in parse_listing(html or "", page_url)]
                max_page = detect_max_page(html or "") or max_page
            except RuntimeError as e:
                log.error("%s", e)
        log.info("[%s] page %s/%s -> %d new profile links", label, page, max_page or "?", len(new))
        for s in new:
            stubs[s.url] = s
        if not new or (max_page and page >= max_page) or (limit and len(stubs) >= limit):
            break
        prev_url, page = page_url, page + 1
    out = list(stubs.values())
    return out[:limit] if limit else out


# --------------------------------------------------------------------------- #
# Profile pages (heading-driven parser: does not depend on CSS class names)
# --------------------------------------------------------------------------- #
H_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]
LEAF_TAGS = H_TAGS + ["p", "li", "div", "dt", "dd", "td", "th", "label"]
BLOCK_TAGS = H_TAGS + ["p", "li", "div", "ul", "ol", "section", "article", "table", "aside"]
STOP_LABELS = {"on this page", "enquire now", "information about", "information for",
               "quick links", "follow us on", "reimagining higher education"}
EXACT_LABELS = {
    "about": "about", "biography": "about", "bio": "about", "overview": "about",
    "research": "research", "research summary": "research", "research focus": "research",
    "research overview": "research",
    "publications": "publications", "selected publications": "publications",
    "key publications": "publications", "books": "publications",
    "courses": "courses", "courses taught": "courses", "teaching": "courses",
    "courses at plaksha": "courses", "courses taught at plaksha": "courses",
    "education": "education", "qualifications": "education", "academic background": "education",
    "research interests": "research_interests",
    "areas of expertise": "expertise", "expertise": "expertise",
    "contact": "contact", "contact details": "contact",
    "current position": "current_position", "current positions": "current_position",
    "past associations": "past", "past experience": "past", "past affiliations": "past",
}


def resolve_heading(text: str, is_h_tag: bool) -> Optional[str]:
    t = clean_text(text).lower().rstrip(":")
    if t in EXACT_LABELS:
        return EXACT_LABELS[t]
    if not is_h_tag:
        return None
    if "research interest" in t:
        return "research_interests"
    if "expertise" in t:
        return "expertise"
    if "publication" in t:
        return "publications"
    if "course" in t or "teaching" in t:
        return "courses"
    if "education" in t or "qualification" in t:
        return "education"
    if t.startswith("about"):
        return "about"
    return "_other:" + t


def leaf_lines(el) -> list[str]:
    for br in el.find_all("br"):
        br.replace_with("\n")
    return [clean_text(x) for x in el.get_text("").split("\n") if clean_text(x)]


def parse_profile_sections(soup: BeautifulSoup):
    """Return (h1_text, preamble_lines, sections{key: [(tag, text)]})."""
    for t in soup(["script", "style", "noscript", "nav", "footer"]):
        t.decompose()
    h1 = soup.find("h1")
    root = soup.body or soup
    preamble: list[str] = []
    sections: dict[str, list[tuple[str, str]]] = {}
    current: Optional[str] = None
    started = h1 is None
    for el in root.find_all(LEAF_TAGS):
        if not started:
            started = el is h1
            continue
        if el.find(BLOCK_TAGS):
            continue  # container, not a leaf
        a = el.find("a")
        if a and (a.get("href") or "").startswith("#") and clean_text(a.get_text()) == clean_text(el.get_text()):
            continue  # "On this page" anchor links
        for line in leaf_lines(el):
            low = line.lower().rstrip(":")
            if low in STOP_LABELS:
                return (clean_text(h1.get_text()) if h1 else ""), preamble, sections
            is_h = el.name in H_TAGS
            key = resolve_heading(line, is_h) if (is_h or el.name in ("p", "div", "dt", "th", "label")) else None
            if key and (is_h or low in EXACT_LABELS):
                current = key
                sections.setdefault(key, [])
                continue
            if current is None:
                if not URL_ONLY_RE.match(line):
                    preamble.append(line)
            else:
                sections[current].append((el.name, line))
    return (clean_text(h1.get_text()) if h1 else ""), preamble, sections


def texts(section, tags=None) -> list[str]:
    return [t for tag, t in section if not tags or tag in tags]


def pick_email(soup_full: BeautifulSoup, sections, name: str, fallback: str) -> str:
    contact_txt = " ".join(texts(sections.get("contact", [])))
    cands = [e.lower() for e in EMAIL_RE.findall(contact_txt)]
    plaksha = [e for e in cands if e.endswith("plaksha.edu.in")]
    if plaksha:
        return plaksha[0]
    if cands:
        return cands[0]
    parts = [p.lower() for p in re.findall(r"[A-Za-z]{3,}", name)]
    for a in soup_full.select('a[href^="mailto:"]'):
        mail = a["href"][7:].split("?")[0].strip().lower()
        if EMAIL_RE.fullmatch(mail) and any(p in mail.split("@")[0] for p in parts):
            return mail
    return fallback or ""


def pick_image(soup_full: BeautifulSoup, name: str, fallback: str) -> str:
    cands = []
    for img in soup_full.find_all("img"):
        blob = " ".join(str(img.get(a, "")) for a in ("src", "srcset", "data-src", "data-srcset"))
        if "/media/people/" in blob:
            cands.append(img)
    for img in cands:
        if clean_text(img.get("alt", "")).lower() == name.lower():
            return best_img_src(img, BASE)
    if cands:
        return best_img_src(cands[0], BASE)
    for prop in ("og:image", "twitter:image"):
        m = soup_full.find("meta", attrs={"property": prop}) or soup_full.find("meta", attrs={"name": prop})
        if m and m.get("content"):
            return urljoin(BASE, m["content"])
    return fallback or ""


def parse_profile(html: str, url: str, stub: Stub, category: str, listed_under: list[str],
                  max_pubs: int) -> dict:
    soup_full = soup_of(html)  # kept intact for email/image lookup (parser mutates its own copy)
    soup = soup_of(html)
    h1_text, preamble, sections = parse_profile_sections(soup)

    name = clean_name(h1_text or stub.name)
    if not name:
        og = soup_full.find("meta", attrs={"property": "og:title"})
        name = clean_name((og.get("content", "") if og else "").split(",")[0])

    # designation
    designation = preamble[0] if preamble else ""
    designation = designation or (texts(sections.get("current_position", [])) or [""])[0] or stub.designation
    designation = re.sub(r",\s*Plaksha University\s*$", "", designation, flags=re.I)

    # education
    education = texts(sections.get("education", []), {"li", "p", "div", "dd", "td"})
    if not education:
        for line in preamble[1:]:
            if DEGREE_RE.search(line):
                education = [line]
                break
    if not education and stub.education:
        education = [stub.education]
    education = dedupe(education)

    # about / research
    about = clean_block("\n\n".join(texts(sections.get("about", []))))
    research = clean_block("\n\n".join(texts(sections.get("research", []))))
    if not research and category not in FACULTY_CATEGORIES:
        bg = texts(sections.get("current_position", [])) + texts(sections.get("past", []))
        research = "; ".join(dedupe(bg))

    interests = dedupe(texts(sections.get("research_interests", []))
                       + texts(sections.get("expertise", [])))
    if not interests and stub.expertise:
        interests = [x.strip() for x in re.split(r"[;,]", stub.expertise) if x.strip()]

    pubs = dedupe(texts(sections.get("publications", [])))
    if max_pubs > 0:
        pubs = pubs[:max_pubs]
    courses = dedupe(texts(sections.get("courses", [])))

    # social links (LinkedIn, Google Scholar, GitHub, Twitter)
    linkedin = ""
    other_links = []
    for a in soup_full.find_all("a", href=True):
        href = a["href"].strip()
        href_lower = href.lower()
        if "linkedin.com" in href_lower:
            linkedin = href
        elif any(domain in href_lower for domain in ["scholar.google", "twitter.com", "x.com", "github.com", "orcid.org"]):
            if href not in other_links:
                other_links.append(href)

    return {
        "name": name,
        "category": category,
        "designation": clean_text(designation),
        "email": pick_email(soup_full, sections, name, stub.email),
        "source_url": url,
        "image_url": pick_image(soup_full, name, stub.image),
        "image_filename": "",
        "linkedin": linkedin,
        "other_links": " | ".join(other_links) if other_links else "",
        "education": education,
        "about": about,
        "research_summary": research,
        "research_interests": interests,
        "publications_sample": pubs,
        "courses": courses,
        "listed_under": listed_under,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def discover(f: Fetcher, limit: Optional[int], only: Optional[set[str]]):
    """Return OrderedDict url -> {'stub', 'category', 'listed_under'}."""
    registry: "OrderedDict[str, dict]" = OrderedDict()
    skipped: list[str] = []
    per_listing_counts: dict[str, int] = {}
    for label, url in LISTINGS:
        if only and label not in only:
            continue
        stubs = crawl_listing(f, label, url, limit)
        per_listing_counts[label] = len(stubs)
        for s in stubs:
            _, seg = classify_profile_url(s.url)
            if label == "Founding Member":
                if seg in IGNORED_FOUNDING_SEGMENTS:
                    skipped.append(s.url)
                    continue
                category = FOUNDING_SEGMENT_CATEGORY.get(seg, "Founding Member")
            else:
                category = label
            entry = registry.get(s.url)
            if entry is None:
                registry[s.url] = {"stub": s, "category": category, "listed_under": [label]}
            else:
                if label not in entry["listed_under"]:
                    entry["listed_under"].append(label)
                if not entry["stub"].email and s.email:
                    entry["stub"].email = s.email
    return registry, per_listing_counts, skipped


def flatten(rec: dict) -> dict:
    row = {}
    for k in FIELDS:
        v = rec.get(k, "")
        if k in LIST_FIELDS:
            v = " | ".join(v or [])
        row[k] = "" if v is None else str(v)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scrape Plaksha University people profiles.")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--min-delay", type=float, default=0.5)
    ap.add_argument("--max-delay", type=float, default=1.0)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--render", action="store_true", help="use Playwright/Chromium for all pages")
    ap.add_argument("--cache-dir", default=None, help="cache fetched HTML here (resume-friendly)")
    ap.add_argument("--limit", type=int, default=None, help="max profiles per listing (testing)")
    ap.add_argument("--only", nargs="*", help='restrict to category labels, e.g. --only "Visiting Faculty"')
    ap.add_argument("--max-pubs", type=int, default=15, help="max publications per person (0 = all)")
    ap.add_argument("--skip-image-check", action="store_true")
    ap.add_argument("--ignore-robots", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(out / "plaksha_scrape.log", "w", "utf-8")],
    )

    f = Fetcher(args.min_delay, args.max_delay, args.retries, args.render,
                Path(args.cache_dir) if args.cache_dir else None, not args.ignore_robots)
    started = datetime.now(timezone.utc)
    try:
        registry, listing_counts, skipped = discover(f, args.limit, set(args.only) if args.only else None)
        log.info("Discovered %d unique profiles. Fetching profile pages...", len(registry))

        people: list[dict] = []
        for i, (url, e) in enumerate(registry.items(), 1):
            log.info("(%d/%d) %s [%s]", i, len(registry), e["stub"].name or url, e["category"])
            html = f.get(url, referer=url.rsplit("/", 1)[0])
            if html is None:
                # fall back to listing-card data so the person is not lost
                s = e["stub"]
                people.append({
                    "name": clean_name(s.name), "category": e["category"],
                    "designation": re.sub(r",\s*Plaksha University\s*$", "", s.designation, flags=re.I),
                    "email": s.email, "source_url": url, "image_url": s.image,
                    "education": [s.education] if s.education else [], "about": "",
                    "research_summary": "",
                    "research_interests": [x.strip() for x in re.split(r"[;,]", s.expertise) if x.strip()],
                    "publications_sample": [], "courses": [], "listed_under": e["listed_under"],
                })
                continue
            try:
                people.append(parse_profile(html, url, e["stub"], e["category"], e["listed_under"], args.max_pubs))
            except Exception:
                log.exception("Parse error for %s", url)
                f.failures[url] = "parse error (see log)"

        # ---- image validation & downloading --------------------------------- #
        broken_images: dict[str, str] = {}
        script_dir = Path(__file__).resolve().parent
        images_dir = script_dir / "1st_images"
        csv_dir = script_dir / "2nd_csv_details"
        ind_csv_dir = csv_dir / "individual_csvs"

        images_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)
        ind_csv_dir.mkdir(parents=True, exist_ok=True)

        existing_db = load_existing_master_csv(csv_dir / "all_plaksha_people_master.csv")
        log.info("Loaded %d existing entries from legacy master CSV for merging.", len(existing_db))

        log.info("Downloading images and updating Image_Filename in %s...", images_dir)
        for p in people:
            u = p["image_url"]
            name = p["name"]
            # Look up existing record
            existing_match = existing_db.get(name.lower()) or existing_db.get(p["source_url"].lower())

            # Download or locate local image
            if u:
                fname = download_image_file(f.session, u, name, images_dir)
                p["image_filename"] = fname
            elif existing_match and existing_match.get("Image_Filename"):
                p["image_filename"] = existing_match["Image_Filename"]

            # Preserve existing social links / bio if missing in scrape
            if not p.get("linkedin") and existing_match and existing_match.get("LinkedIn"):
                p["linkedin"] = existing_match["LinkedIn"]
            if not p.get("other_links") and existing_match and existing_match.get("Other_Links"):
                p["other_links"] = existing_match["Other_Links"]
            if not p.get("about") and existing_match and existing_match.get("Bio"):
                p["about"] = existing_match["Bio"]

        # Preserve legacy records that weren't visited in this scrape session
        scraped_names = {p["name"].lower() for p in people}
        for k, ex in existing_db.items():
            ex_name = ex.get("Name", "")
            if ex_name and ex_name.lower() not in scraped_names:
                scraped_names.add(ex_name.lower())
                people.append({
                    "name": ex_name,
                    "category": ex.get("Category", "Plaksha Community"),
                    "designation": ex.get("Designation", ""),
                    "email": ex.get("Email", ""),
                    "source_url": ex.get("Source_URL", ""),
                    "image_url": ex.get("Image_URL", ""),
                    "image_filename": ex.get("Image_Filename", ""),
                    "linkedin": ex.get("LinkedIn", ""),
                    "other_links": ex.get("Other_Links", ""),
                    "education": [ex.get("Education")] if ex.get("Education") else [],
                    "about": ex.get("Bio", ""),
                    "research_summary": ex.get("Research_Summary", ""),
                    "research_interests": [x.strip() for x in ex.get("Research_Interests", "").split("|") if x.strip()],
                    "publications_sample": [x.strip() for x in ex.get("Publications_Sample", "").split("|") if x.strip()],
                    "courses": [x.strip() for x in ex.get("Courses", "").split("|") if x.strip()],
                    "listed_under": [ex.get("Category", "Plaksha Community")],
                })

        # ---- exports --------------------------------------------------------- #
        people.sort(key=lambda p: (
            CATEGORY_ORDER.index(p["category"]) if p["category"] in CATEGORY_ORDER else 99,
            p["name"].lower()
        ))
        records = [{k: p.get(k, [] if k in LIST_FIELDS else "") for k in FIELDS} for p in people]

        # 1. Structured JSON for agent
        json_targets = [
            out / "plaksha_people_dataset.json",
            script_dir / "plaksha_people_dataset.json",
            Path("pydantic_ai_agent/plaksha_people_dataset.json")
        ]
        for jt in json_targets:
            try:
                jt.parent.mkdir(parents=True, exist_ok=True)
                with open(jt, "w", encoding="utf-8") as fh:
                    json.dump(records, fh, indent=2, ensure_ascii=False)
            except Exception as e:
                log.warning("Could not write JSON to %s: %s", jt, e)

        # 2. Master CSV with ALL legacy columns preserved + new columns added
        master_rows = []
        for p in people:
            ex_match = existing_db.get(p["name"].lower()) or existing_db.get(p["source_url"].lower())
            master_rows.append(flatten_to_master_row(p, ex_match))

        master_csv_targets = [
            out / "all_plaksha_people_master.csv",
            csv_dir / "all_plaksha_people_master.csv"
        ]
        for mct in master_csv_targets:
            try:
                mct.parent.mkdir(parents=True, exist_ok=True)
                with open(mct, "w", encoding="utf-8-sig", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=MASTER_CSV_COLUMNS, quoting=csv.QUOTE_ALL)
                    w.writeheader()
                    for r in master_rows:
                        w.writerow(r)
            except Exception as e:
                log.warning("Could not write master CSV to %s: %s", mct, e)

        # 3. Individual CSVs (one per person) in 2nd_csv_details/individual_csvs/
        log.info("Writing %d individual CSVs to %s...", len(master_rows), ind_csv_dir)
        for r in master_rows:
            clean_n = clean_filename(r["Name"])
            if clean_n:
                ind_csv_file = ind_csv_dir / f"{clean_n}.csv"
                try:
                    with open(ind_csv_file, "w", encoding="utf-8-sig", newline="") as fh:
                        w = csv.DictWriter(fh, fieldnames=MASTER_CSV_COLUMNS, quoting=csv.QUOTE_ALL)
                        w.writeheader()
                        w.writerow(r)
                except Exception as e:
                    log.warning("Could not write individual CSV %s: %s", ind_csv_file, e)

        # ---- summary --------------------------------------------------------- #
        primary = Counter(p["category"] for p in people)
        membership = Counter(c for p in people for c in p["listed_under"])
        missing_img = [p["source_url"] for p in people if not p["image_url"]]
        missing = {fld: [p["source_url"] for p in people if not p[fld]]
                   for fld in ("email", "about", "education", "designation")}
        report = {
            "run_started_utc": started.isoformat(),
            "unique_profiles": len(people),
            "profiles_by_primary_category": {c: primary.get(c, 0) for c in CATEGORY_ORDER},
            "profiles_by_listing_membership": {c: membership.get(c, 0) for c in CATEGORY_ORDER},
            "raw_listing_counts": listing_counts,
            "skipped_unrequested_founding_segments": skipped,
            "broken_links": f.failures,
            "missing_images": missing_img,
            "broken_images": broken_images,
            "missing_fields_counts": {k: len(v) for k, v in missing.items()},
            "missing_fields_urls": missing,
        }
        with open(out / "scrape_report.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)

        log.info("=" * 64)
        log.info("EXECUTION SUMMARY  (%d unique profiles)", len(people))
        for c in CATEGORY_ORDER:
            log.info("  %-34s primary=%-4d listed=%d", c, primary.get(c, 0), membership.get(c, 0))
        log.info("  (a Founder/Trustee is also 'listed' under Founding Member; primary = most specific)")
        log.info("Skipped (corporate donors / changemakers): %d", len(skipped))
        log.info("Broken / failed links: %d", len(f.failures))
        for u, why in f.failures.items():
            log.info("    %s -> %s", u, why)
        log.info("Missing image_url: %d | broken image URLs: %d", len(missing_img), len(broken_images))
        for u in missing_img:
            log.info("    no image: %s", u)
        for u, why in broken_images.items():
            log.info("    broken image: %s -> %s", u, why)
        log.info("Missing fields: %s", report["missing_fields_counts"])
        log.info("Wrote: %s", ", ".join(str(out / n) for n in (
            "plaksha_people_dataset.json", "all_plaksha_people_master.csv", "scrape_report.json")))
    finally:
        f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
