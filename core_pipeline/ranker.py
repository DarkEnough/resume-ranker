"""
Cosine-similarity ranking between a job description and candidate resumes.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Set
from .embedder import Embedder
from functools import lru_cache
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import re

@lru_cache(maxsize=1)
def load_skill_extractor():
    """Load LinkedIn Skills Recognition model once and cache it"""
    tokenizer = AutoTokenizer.from_pretrained("algiraldohe/lm-ner-linkedin-skills-recognition")
    model = AutoModelForTokenClassification.from_pretrained("algiraldohe/lm-ner-linkedin-skills-recognition")
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def _cosine(a: NDArray, b: NDArray) -> float:
    return float(np.dot(a, b))

def _extract_candidate_name(resume_text: str, filename: str) -> str:
    """Extract candidate name from resume text, fallback to filename"""
    lines = resume_text.split('\n')[:10]
    
    # Common patterns for names at the start of resumes
    name_patterns = [
        r'^([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'^([A-Z][A-Z\s]+)$',
        r'Name:\s*([A-Za-z\s]+)',
        r'^([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)',
    ]
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 5:
            continue
            
        for pattern in name_patterns:
            match = re.search(pattern, line)
            if match:
                name = match.group(1).strip()
                words = name.split()
                if 2 <= len(words) <= 4 and 5 <= len(name) <= 50:
                    return name
    
    # Fallback: clean up filename
    clean_name = filename.replace('.pdf', '').replace('.docx', '').replace('.txt', '')
    clean_name = re.sub(r'[_-]', ' ', clean_name)
    clean_name = clean_name.title()
    
    return clean_name

def extract_skills_with_transformer(text: str, max_length: int = None) -> Set[str]:
    """
    Use SkillNER model for skill extraction
    
    Args:
        text: Text to extract skills from
        max_length: Maximum text length to process (for performance)
    """
    nlp = load_skill_extractor()
    
    # Limit text length if specified
    if max_length:
        text = text[:max_length]
    
    # Process text in chunks since BERT has 512 token limit
    skills = set()
    chunk_size = 400  # Smaller chunks to be safe with token limit
    
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        
        try:
            entities = nlp(chunk)
            
            for entity in entities:
                # LinkedIn Skills Recognition model labels skills as 'BUS', 'TECHNOLOGY', 'TECHNICAL', 'SOFT'
                if entity['entity_group'] in ['BUS', 'TECHNOLOGY', 'TECHNICAL', 'SOFT']:
                    skill = entity['word'].strip()
                    # Clean up BERT tokenization artifacts
                    skill = skill.replace('##', '').strip()
                    # Normalize spaces and case
                    skill = ' '.join(skill.split())
                    
                    if len(skill) > 2 and len(skill) < 50:  # Reasonable skill length
                        skills.add(skill.lower())
        except Exception as e:
            print(f"Error processing chunk: {e}")
            continue
    
    return skills

def extract_skills_from_text(text: str, focus_sections: bool = True) -> List[str]:
    """
    Generic skill extraction that can be used for both JD and resumes
    
    Args:
        text: Text to extract from
        focus_sections: Whether to focus on skill-related sections
    """
    skills_set = set()
    
    if focus_sections:
        # Focus on sections likely to contain skills
        lines = text.split('\n')
        skill_sections = []
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check for skill section headers
            if any(keyword in line_lower for keyword in 
                   ['skill', 'technical', 'expertise', 'competenc', 'proficien',
                    'requirement', 'qualification', 'experience']):
                # Include this line and next few lines
                skill_sections.extend(lines[i:min(i+10, len(lines))])
            
            # Check for bullet points (often contain skills)
            elif line.strip().startswith(('-', '•', '*', '·', '◦')):
                skill_sections.append(line)
        
        # Extract skills from focused sections
        if skill_sections:
            section_text = '\n'.join(skill_sections)
            skills_set.update(extract_skills_with_transformer(section_text, max_length=3000))
    
    # Also extract from full text (limited length)
    skills_set.update(extract_skills_with_transformer(text, max_length=2000))
    
    # Convert to list and remove duplicates
    skills_list = list(skills_set)
    
    # Sort by frequency in original text (more mentions = more important)
    text_lower = text.lower()
    skills_list.sort(key=lambda s: text_lower.count(s), reverse=True)
    
    return skills_list

def extract_skill_focused_sections(text: str, target_skills: List[str]) -> str:
    """
    Extract sections of text that are most relevant to target skills
    
    Args:
        text: Full text (resume or JD)
        target_skills: Skills to focus on
    """
    if not target_skills:
        return text[:1000]
    
    lines = text.split('\n')
    relevance_scores = []
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        
        line_lower = line.lower()
        score = 0
        
        # Check how many target skills appear in this line
        for skill in target_skills:
            if skill.lower() in line_lower:
                score += 3
        
        # Boost skill section headers
        if any(header in line_lower for header in ['skill', 'technical', 'expertise']):
            score += 2
        
        # Boost experience indicators
        if any(exp in line_lower for exp in ['developed', 'built', 'implemented', 'designed', 'created']):
            score += 1
        
        if score > 0:
            # Get context (line before and after)
            context_start = max(0, i-1)
            context_end = min(len(lines), i+2)
            context = ' '.join(lines[context_start:context_end])
            relevance_scores.append((context, score))
    
    # Sort by relevance
    relevance_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take top relevant sections
    relevant_sections = []
    seen = set()
    
    for section, score in relevance_scores[:10]:
        if section not in seen:
            relevant_sections.append(section)
            seen.add(section)
    
    result = ' '.join(relevant_sections)
    return result[:1500] if result else text[:1000]

def rank_candidates(
    job_description: str,
    resumes: List[Dict[str, str]],
    top_k: int = 10,
    embedder: Embedder | None = None,
) -> List[Dict[str, float]]:
    """
    Fair ranking using same SkillNER model for both JD and resumes
    """
    if embedder is None:
        embedder = Embedder()
    
    # Extract skills from JD using SkillNER
    print("Extracting skills from job description using SkillNER...")
    jd_skills = extract_skills_from_text(job_description, focus_sections=True)[:20]
    print(f"Found {len(jd_skills)} skills in JD")
    
    if jd_skills:
        print(f"Top JD skills: {', '.join(jd_skills[:5])}...")
    
    # Process each resume with the SAME model
    resume_data = []
    print(f"Processing {len(resumes)} resumes...")
    
    for resume in resumes:
        # Extract skills from resume using SAME SkillNER model
        resume_skills = extract_skills_from_text(resume["text"], focus_sections=True)
        
        # Calculate skill overlap
        matched_skills = [skill for skill in jd_skills if skill in resume_skills]
        missing_skills = [skill for skill in jd_skills if skill not in resume_skills]
        
        resume_data.append({
            "text": resume["text"],
            "id": resume["id"],
            "skills": resume_skills,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "skill_match_rate": len(matched_skills) / len(jd_skills) if jd_skills else 0
        })
    
    # Get embeddings for full texts
    all_texts = [job_description] + [r["text"] for r in resumes]
    full_embs = embedder.encode(all_texts)
    
    # Extract skill-focused sections for better matching
    jd_skill_section = extract_skill_focused_sections(job_description, jd_skills)
    resume_skill_sections = []
    
    for resume_info in resume_data:
        # Focus on sections containing matched skills
        skill_section = extract_skill_focused_sections(
            resume_info["text"], 
            resume_info["matched_skills"] + jd_skills[:5]  # Include top JD skills
        )
        resume_skill_sections.append(skill_section)
    
    # Encode skill-focused sections
    skills_texts = [jd_skill_section] + resume_skill_sections
    skills_embs = embedder.encode(skills_texts)
    
    jd_full, res_full = full_embs[0], full_embs[1:]
    jd_skills_emb, res_skills_emb = skills_embs[0], skills_embs[1:]
    
    # Calculate final rankings
    results = []
    
    for i, resume_info in enumerate(resume_data):
        # Cosine similarities
        full_sim = _cosine(jd_full, res_full[i])
        skills_sim = _cosine(jd_skills_emb, res_skills_emb[i])
        
        # Weighted combination (skills more important)
        combined_sim = 0.35 * full_sim + 0.65 * skills_sim
        
        # Optional: Boost based on skill match rate
        # This gives a small bonus for having more of the required skills
        skill_bonus = resume_info["skill_match_rate"] * 0.1  # Max 10% bonus
        final_score = min(combined_sim + skill_bonus, 1.0)  # Cap at 1.0
        
        candidate_name = _extract_candidate_name(resume_info["text"], resume_info["id"])
        
        results.append({
            "id": candidate_name,
            "filename": resume_info["id"],
            "similarity": final_score,
            "matched_skills": resume_info["matched_skills"],
            "missing_skills": resume_info["missing_skills"],
            "skill_count": len(resume_info["matched_skills"]),
            "total_skills": len(resume_info["skills"]),
            "skill_match_rate": resume_info["skill_match_rate"]
        })
    
    # Sort by similarity
    ranked = sorted(results, key=lambda d: d["similarity"], reverse=True)
    
    # Attach JD skills to results for visualization
    for result in ranked:
        result['jd_skills'] = jd_skills
    
    return ranked[:top_k or len(ranked)]

# Export functions
__all__ = ['rank_candidates', 'extract_skills_from_text', 'extract_skills_with_transformer']