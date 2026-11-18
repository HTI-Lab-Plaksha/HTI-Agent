"""
Plaksha Website Scraper: Extracts clean bio data from plaksha.edu.in
for all known faculty/founders/trustees and writes to a JSON dataset.
"""
import re
import json
import time
import sys
import csv
import urllib.request
import urllib.error
from html.parser import HTMLParser


class PlakshaProfileParser(HTMLParser):
    """Parse faculty/team profile pages to extract structured bio data."""
    
    def __init__(self):
        super().__init__()
        self.in_main = False
        self.in_prose = False
        self.in_h1 = False
        self.in_h2 = False
        self.in_li = False
        self.current_section = "about"
        self.depth = 0
        
        self.title = ""
        self.meta_description = ""
        self.meta_title = ""
        self.image_url = ""
        self.education = ""
        
        self.about_paragraphs = []
        self.research_paragraphs = []
        self.research_interests = []
        self.publications = []
        self.courses = []
        self.current_text = ""
        
        # Track specific div regions
        self.capture_text = False
        self.in_research_interests_list = False
        self.in_education_div = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == "main":
            self.in_main = True
            
        if tag == "title":
            self.in_h1 = True  # reuse to capture title tag
            
        if tag == "meta":
            name = attrs_dict.get("name", "")
            prop = attrs_dict.get("property", "")
            content = attrs_dict.get("content", "")
            if name == "description" or prop == "og:description":
                if not self.meta_description:
                    self.meta_description = content
            if prop == "og:title":
                self.meta_title = content
            if prop == "og:image" or name == "image":
                if not self.image_url and "plaksha" in content:
                    self.image_url = content
        
        if not self.in_main:
            return
            
        if tag == "h1":
            self.in_h1 = True
            self.current_text = ""
            
        if tag == "h2":
            self.in_h2 = True
            self.current_text = ""
            
        # Detect education section (usually has specific classes or text like PhD, etc.)
        cls = attrs_dict.get("class", "")
        
        if tag == "div" and "prose" in cls:
            self.in_prose = True
            self.current_text = ""
            
        if tag == "li":
            self.in_li = True
            self.current_text = ""
            
        # Track section IDs
        section_id = attrs_dict.get("id", "")
        if section_id == "research":
            self.current_section = "research"
        elif section_id in ("publications", "selected-publications"):
            self.current_section = "publications"
        elif section_id in ("teaching", "courses"):
            self.current_section = "courses"
            
    def handle_endtag(self, tag):
        if tag == "main":
            self.in_main = False
            
        if tag == "title" and self.in_h1 and not self.in_main:
            self.in_h1 = False
            
        if not self.in_main:
            return
            
        if tag == "h1":
            self.in_h1 = False
            if self.current_text.strip():
                self.title = self.current_text.strip()
                
        if tag == "h2":
            self.in_h2 = False
            section_text = self.current_text.strip().lower()
            if "research" in section_text and "publication" not in section_text:
                self.current_section = "research"
            elif "publication" in section_text:
                self.current_section = "publications"
            elif "teach" in section_text or "course" in section_text:
                self.current_section = "courses"
            elif "about" in section_text or "bio" in section_text:
                self.current_section = "about"
                
        if tag == "div" and self.in_prose:
            self.in_prose = False
            text = self.current_text.strip()
            if text and len(text) > 20:
                if self.current_section == "research":
                    self.research_paragraphs.append(text)
                elif self.current_section == "publications":
                    pass  # handled by li
                else:
                    self.about_paragraphs.append(text)
                    
        if tag == "li" and self.in_li:
            self.in_li = False
            text = self.current_text.strip()
            if text:
                if self.current_section == "publications":
                    self.publications.append(text)
                elif self.current_section == "research":
                    self.research_interests.append(text)
                elif self.current_section == "courses":
                    self.courses.append(text)
                    
    def handle_data(self, data):
        if self.in_main and (self.in_prose or self.in_h1 or self.in_h2 or self.in_li):
            self.current_text += data
        elif not self.in_main and self.in_h1:
            # Capture <title> tag content
            self.current_text += data
            
    def get_education_from_page(self, html: str) -> str:
        """Extract education lines that typically appear near the top of the profile."""
        # Look for common education patterns like "PhD, University of ..."
        patterns = [
            r'(?:PhD|Ph\.D|MSc|M\.Sc|MS|M\.S|BS|B\.S|BTech|B\.Tech|MTech|M\.Tech|MBA|MA|MPhil)[\s,]+[^<]{5,80}',
        ]
        matches = []
        for pat in patterns:
            for m in re.finditer(pat, html):
                text = m.group().strip()
                if len(text) > 10:
                    matches.append(text)
        return "; ".join(list(dict.fromkeys(matches))[:3])  # deduplicate, limit to 3
        

def fetch_page(url: str) -> str:
    """Fetch HTML content from a URL."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [ERROR] Failed to fetch {url}: {e}")
        return ""


def scrape_profile(url: str) -> dict:
    """Scrape a single Plaksha profile page and return structured data."""
    html = fetch_page(url)
    if not html:
        return {}
    
    parser = PlakshaProfileParser()
    try:
        parser.feed(html)
    except Exception as e:
        print(f"  [WARN] Parse error for {url}: {e}")
    
    education = parser.get_education_from_page(html)
    
    return {
        "title": parser.meta_title or parser.title,
        "meta_description": parser.meta_description,
        "image_url": parser.image_url,
        "education": education,
        "about": " ".join(parser.about_paragraphs),
        "research_summary": " ".join(parser.research_paragraphs),
        "research_interests": parser.research_interests[:10],
        "publications": parser.publications[:5],
        "courses": parser.courses[:10],
    }


def main():
    csv_path = "HTI Data/2nd_csv_details/all_plaksha_people_master.csv"
    output_path = "pydantic_ai_agent/plaksha_people_dataset.json"
    
    # Read existing CSV
    people = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        seen_names = set()
        for row in reader:
            name = row.get("Name", "").strip()
            if name and name not in seen_names:
                seen_names.add(name)
                people.append({
                    "name": name,
                    "category": row.get("Category", ""),
                    "designation": row.get("Designation", ""),
                    "email": row.get("Email", ""),
                    "source_url": row.get("Source_URL", ""),
                    "csv_image_url": row.get("Image_URL", ""),
                })
    
    print(f"[SCRAPER] Found {len(people)} unique people in CSV. Starting scrape...\n")
    
    dataset = []
    for i, person in enumerate(people):
        url = person["source_url"]
        name = person["name"]
        print(f"[{i+1}/{len(people)}] Scraping: {name}")
        print(f"  URL: {url}")
        
        if not url:
            print("  [SKIP] No URL available.")
            profile_data = {}
        else:
            profile_data = scrape_profile(url)
            time.sleep(0.3)  # Be polite to the server
        
        # Build enriched record
        record = {
            "name": name,
            "category": person["category"],
            "designation": person["designation"],
            "email": person["email"],
            "source_url": url,
            "image_url": profile_data.get("image_url") or person.get("csv_image_url", ""),
            "page_title": profile_data.get("title", ""),
            "meta_description": profile_data.get("meta_description", ""),
            "education": profile_data.get("education", ""),
            "about": profile_data.get("about", ""),
            "research_summary": profile_data.get("research_summary", ""),
            "research_interests": profile_data.get("research_interests", []),
            "publications_sample": profile_data.get("publications", []),
            "courses": profile_data.get("courses", []),
        }
        
        about_preview = record["about"][:120] + "..." if len(record.get("about", "")) > 120 else record.get("about", "")
        print(f"  Education: {record['education'] or '(not found)'}")
        print(f"  About: {about_preview or '(not found)'}")
        print()
        
        dataset.append(record)
    
    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\n[DONE] Wrote {len(dataset)} records to {output_path}")
    
    # Summary stats
    has_about = sum(1 for d in dataset if d.get("about"))
    has_edu = sum(1 for d in dataset if d.get("education"))
    has_research = sum(1 for d in dataset if d.get("research_summary"))
    print(f"  With About bio: {has_about}/{len(dataset)}")
    print(f"  With Education: {has_edu}/{len(dataset)}")
    print(f"  With Research:  {has_research}/{len(dataset)}")


if __name__ == "__main__":
    main()
