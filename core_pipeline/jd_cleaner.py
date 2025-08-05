import re
import unicodedata
from typing import List

# words/phrases that mark sections we want to DROP
DROP_HEADERS: List[str] = [
    "about the company", "company overview", "who we are", "what we do",
    "featured benefits", "benefits include", "perks", "insurance",
    "medical insurance", "vision insurance", "dental insurance", "401(k)",
    "equal opportunity", "diversity and inclusion", "location:", "headquartered",
    "verification", "background check", "salary range", "compensation and benefits",
    "compensation", "base pay", "salary", "pay range", "stipend", "relocation",
    "wellness", "healthcare", "parental leave", "pto", "vacation",
    "notice to applicants", "covey", "fair chance", "non-discrimination",
    "diversity", "inclusion", "equal opportunity employer", "statement of",
    "pursuant to", "ordinance", "regulation", "legal", "compliance",
    "our commitment", "our mission", "we're committed", "we hire",
    "we value", "we believe", "thank you to", "level playing field",
]

# words/phrases that mark sections we want to KEEP (acts as a whitelist)
KEEP_ANCHORS: List[str] = [
    "responsibilities", "key responsibilities",
    "qualifications", "required", "must have",
    "preferred", "desired", "preferred skills", "preferred qualifications", "you will"
    "skills", "requirements",
]

def clean_job_description(text: str) -> str:
    """
    Remove paragraphs that are likely irrelevant for skill matching
    (benefits, company blurb, EEO statements, etc.).
    """
    # normalise to NFC & lower-case for matching
    text_n = unicodedata.normalize("NFC", text)
    paragraphs = re.split(r"\n\s*\n", text_n)          # split on blank lines

    kept: List[str] = []
    for para in paragraphs:
        p = para.strip()
        if not p:
            continue
        low = p.lower()

        # If it contains any KEEP anchor, always keep
        if any(k in low for k in KEEP_ANCHORS):
            kept.append(p)
            continue

        # If it starts with or contains any DROP header, drop
        if any(h in low for h in DROP_HEADERS):
            continue

        # fallback: keep only short paragraphs (<100 words) to be more aggressive
        if len(p.split()) < 100:
            kept.append(p)

    return "\n\n".join(kept) 