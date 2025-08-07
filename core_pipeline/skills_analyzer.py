"""
Skills analysis and visualization for candidate matching
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple
import streamlit as st

def match_skill_in_resume(skill: str, resume_text: str) -> float:
    """
    Check if a skill exists in resume text
    Returns 1.0 if found, 0.0 if not
    """
    skill_lower = skill.lower()
    resume_lower = resume_text.lower()
    
    # Direct match
    if skill_lower in resume_lower:
        return 1.0
    
    # Check for partial matches (for compound skills)
    skill_parts = skill_lower.split()
    if len(skill_parts) > 1:
        matches = sum(1 for part in skill_parts if len(part) > 3 and part in resume_lower)
        if matches >= len(skill_parts) * 0.6:  # 60% of words match
            return 0.8
    
    return 0.0

def analyze_skills_coverage(jd: str, ranked_candidates: list, resumes: list) -> Tuple[pd.DataFrame, List[str]]:
    """
    Analyze which skills each candidate has/lacks
    """
    from .ranker import extract_skills_from_text
    
    # Extract skills from JD
    jd_skills = extract_skills_from_text(jd)
    
    if not jd_skills:
        st.warning("Could not extract skills from job description")
        return pd.DataFrame(), []
    
    # Analyze each candidate
    coverage_data = []
    
    for candidate in ranked_candidates[:10]:  # Top 10 candidates
        # Find resume text
        resume_text = next(
            r["text"] for r in resumes 
            if r["id"] == candidate.get("filename", candidate["id"])
        )
        
        matched_skills = []
        missing_skills = []
        
        for skill in jd_skills:
            if match_skill_in_resume(skill, resume_text) > 0.5:
                matched_skills.append(skill)
            else:
                missing_skills.append(skill)
        
        coverage_data.append({
            'candidate': candidate["id"],
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'coverage_percentage': (len(matched_skills) / len(jd_skills)) * 100 if jd_skills else 0,
            'match_count': len(matched_skills),
            'missing_count': len(missing_skills)
        })
    
    return pd.DataFrame(coverage_data), jd_skills

def create_skills_heatmap(jd: str, ranked_candidates: list, resumes: list, top_n: int = 5):
    """
    Create interactive skills heatmap
    """
    df_coverage, jd_skills = analyze_skills_coverage(jd, ranked_candidates, resumes)
    
    if df_coverage.empty:
        return None, None
    
    # Create heatmap data
    heatmap_data = []
    
    for _, row in df_coverage.head(top_n).iterrows():
        for skill in jd_skills[:10]:  # Limit to 10 skills for readability
            heatmap_data.append({
                'Candidate': row['candidate'][:25],  # Truncate long names
                'Skill': skill.title(),
                'Match': 1.0 if skill in row['matched_skills'] else 0.0
            })
    
    df_heatmap = pd.DataFrame(heatmap_data)
    
    # Handle duplicate entries by aggregating (taking max value)
    df_heatmap = df_heatmap.groupby(['Candidate', 'Skill'])['Match'].max().reset_index()
    
    pivot = df_heatmap.pivot(index='Candidate', columns='Skill', values='Match')
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=[[0, '#FF4444'], [1, '#00CC00']],  # Red to Green
        text=pivot.values,
        texttemplate="%{text:.0f}",
        textfont={"size": 12},
        hovertemplate="Candidate: %{y}<br>Skill: %{x}<br>Match: %{z}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Skills Coverage Matrix",
        xaxis_title="Required Skills (from JD)",
        yaxis_title="Top Candidates",
        height=400,
        xaxis={'side': 'top'},
        margin=dict(t=100)
    )
    
    return fig, df_coverage

def create_missing_skills_chart(df_coverage: pd.DataFrame, top_n: int = 5):
    """
    Create a chart showing what skills are most commonly missing
    """
    # Aggregate missing skills across all candidates
    all_missing = []
    for missing_list in df_coverage.head(top_n)['missing_skills']:
        all_missing.extend(missing_list)
    
    if not all_missing:
        return None
    
    # Count frequency
    from collections import Counter
    missing_counts = Counter(all_missing)
    
    # Create bar chart
    df_missing = pd.DataFrame(
        missing_counts.most_common(10),
        columns=['Skill', 'Candidates Missing']
    )
    
    fig = px.bar(
        df_missing,
        x='Candidates Missing',
        y='Skill',
        orientation='h',
        title=f'Most Commonly Missing Skills (Top {top_n} Candidates)',
        color='Candidates Missing',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Number of Candidates Missing This Skill",
        yaxis_title=""
    )
    
    return fig

def create_skills_gap_analysis(df_coverage: pd.DataFrame):
    """
    Create detailed gap analysis for each candidate
    """
    gap_analysis = []
    
    for _, row in df_coverage.head(5).iterrows():
        gap_analysis.append({
            'Candidate': row['candidate'],
            'Coverage': f"{row['coverage_percentage']:.0f}%",
            'Has': len(row['matched_skills']),
            'Missing': len(row['missing_skills']),
            'Key Gaps': ', '.join(row['missing_skills'][:3]) if row['missing_skills'] else 'None'
        })
    
    return pd.DataFrame(gap_analysis)
