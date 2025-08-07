"""
Cosine-similarity ranking between a job description and candidate resumes.
"""

from __future__ import annotations

import re
import numpy as np
from numpy.typing import NDArray
from typing import List, Dict

from .embedder import Embedder


def _cosine(a: NDArray, b: NDArray) -> float:
    return float(np.dot(a, b))


def _extract_candidate_name(resume_text: str, filename: str) -> str:
    """Extract candidate name from resume text, fallback to filename"""
    lines = resume_text.split('\n')[:10]  # Check first 10 lines
    
    # Common patterns for names at the start of resumes
    name_patterns = [
        r'^([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # "John Smith" or "John Smith Jr"
        r'^([A-Z][A-Z\s]+)$',  # "JOHN SMITH" (all caps)
        r'Name:\s*([A-Za-z\s]+)',  # "Name: John Smith"
        r'^([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)',  # "John A. Smith"
    ]
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 5:  # Skip very short lines
            continue
            
        for pattern in name_patterns:
            match = re.search(pattern, line)
            if match:
                name = match.group(1).strip()
                # Validate name (2-4 words, reasonable length)
                words = name.split()
                if 2 <= len(words) <= 4 and 5 <= len(name) <= 50:
                    return name
    
    # Fallback: clean up filename
    clean_name = filename.replace('.pdf', '').replace('.docx', '').replace('.txt', '')
    clean_name = re.sub(r'[_-]', ' ', clean_name)  # Replace underscores/dashes with spaces
    clean_name = clean_name.title()  # Title case
    return clean_name


def rank_candidates(
    job_description: str,
    resumes: List[Dict[str, str]],
    top_k: int = 10,
    embedder: Embedder | None = None,
) -> List[Dict[str, float]]:
    """
    Two-stage ranking skills+normal resume cosine similarity
    """
    if embedder is None:
        embedder = Embedder()
    
    # Stage 1: Extract likely skills sections
    def extract_skills_section(text: str) -> str:
        lines = text.split('\n')
        skills_text = []
        for i, line in enumerate(lines):
            if 'skill' in line.lower() or 'technical' in line.lower():
                # Get this line and next few
                skills_text.extend(lines[i:min(i+5, len(lines))])
        return ' '.join(skills_text) if skills_text else text[:300]
    
    # Get embeddings for full text
    all_texts = [job_description] + [r["text"] for r in resumes]
    full_embs = embedder.encode(all_texts)
    
    # Get embeddings for skills-focused sections
    skills_texts = [extract_skills_section(job_description)] + [extract_skills_section(r["text"]) for r in resumes]
    skills_embs = embedder.encode(skills_texts)
    
    jd_full, res_full = full_embs[0], full_embs[1:]
    jd_skills, res_skills = skills_embs[0], skills_embs[1:]
    
    # Weighted combination (still cosine similarity)
    similarities = []
    for i in range(len(resumes)):
        full_sim = _cosine(jd_full, res_full[i])
        skills_sim = _cosine(jd_skills, res_skills[i])
        
        # Weight skills match higher
        combined_sim = 0.4 * full_sim + 0.6 * skills_sim
        similarities.append(combined_sim)
    
    # Extract candidate names and build results
    results = []
    for resume, similarity in zip(resumes, similarities):
        candidate_name = _extract_candidate_name(resume["text"], resume["id"])
        results.append({
            "id": candidate_name,
            "filename": resume["id"],  # Keep original filename for reference
            "similarity": similarity
        })
    
    ranked = sorted(results, key=lambda d: d["similarity"], reverse=True)
    return ranked[:top_k or len(ranked)]
