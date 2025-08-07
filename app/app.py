"""
Streamlit v0.3.1  –  summaries only on explicit button press
"""

from __future__ import annotations
import tempfile
from pathlib import Path
from typing import List, Dict

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from core_pipeline import (
    extract_text,
    Embedder,
    rank_candidates,
    generate_fit_summary,
    summaries_available,
    clean_job_description,
)
from core_pipeline.skills_analyzer import create_skills_heatmap, create_missing_skills_chart, create_skills_gap_analysis

# ───────────────────── Config ───────────────────── #
st.set_page_config("Candidate Recommender", "", layout="wide")
st.session_state.setdefault("ranked", None)          # list[dict] or None
st.session_state.setdefault("summaries_done", False)
st.session_state.setdefault("resumes", None)         # store processed resumes

MAX_SIZE_MB = 5
ALLOWED_EXT = {"pdf", "docx", "txt"}
MAX_RESUMES = 30
DEFAULT_TOP_K = 10

# ─────────────────── Helpers ────────────────────── #
def _cache_path(upload) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix)
    tmp.write(upload.getbuffer())
    tmp.flush()
    return Path(tmp.name)


def _load_resumes(files) -> List[Dict]:
    rows = []
    for up in files[:MAX_RESUMES]:
        if up.size > MAX_SIZE_MB * 1024 * 1024:
            st.warning(f"Skipped {up.name} (> {MAX_SIZE_MB} MB).")
            continue
        text = extract_text(_cache_path(up))
        if text:
            rows.append({"id": up.name, "text": text})
        else:
            st.warning(f"Unsupported / empty file: {up.name}")
    return rows


# ───────────────────── UI ───────────────────────── #
st.title("Candidate Recommendation Engine")

jd = st.text_area("Job description", height=200, placeholder="Paste JD here…")
files = st.file_uploader(
    "Upload resumes (PDF · DOCX · TXT)",
    type=list(ALLOWED_EXT),
    accept_multiple_files=True,
)
top_k = st.number_input("Show top-N", 1, 20, value=DEFAULT_TOP_K)

# ----------  Rank button ---------- #
if st.button("Rank Candidates", disabled=(not jd or not files)):
    with st.spinner("Embedding & ranking…"):
        resumes = _load_resumes(files)
        if not resumes:
            st.error("No valid resumes processed."); st.stop()

        st.session_state["resumes"] = resumes  # store for later use
        
        # Clean the job description to remove irrelevant sections
        jd_clean = clean_job_description(jd)
        st.session_state["ranked"] = rank_candidates(
            jd_clean, resumes, top_k=top_k, embedder=Embedder()
        )
        st.session_state["summaries_done"] = False
        st.success("Ranking complete!  Scroll down 🡣")

# ---------- Show results if available ---------- #
if st.session_state["ranked"]:
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Rankings", "🎯 Skills Analysis", "📈 Analytics", "📋 Export"])
    
    with tab1:
        # Existing ranking display
        df = pd.DataFrame(st.session_state["ranked"])
        st.subheader("Top Candidates")
        st.dataframe(
            df[["id", "similarity"] + (["summary"] if "summary" in df.columns else [])],
            hide_index=True,
            use_container_width=True,
            column_config={
                "id": st.column_config.TextColumn("Candidate", width="medium"),
                "similarity": st.column_config.ProgressColumn(
                    "Match Score",
                    format="%.1%%",
                    min_value=0,
                    max_value=1,
                ),
                "summary": st.column_config.TextColumn(
                    "Why They Match",
                    width="large",
                    help="AI-generated analysis"
                ) if "summary" in df.columns else None
            }
        )
        
        # ---------- Summaries button ---------- #
        if summaries_available() and not st.session_state["summaries_done"]:
            if st.button("Generate fit summaries"):
                with st.spinner("Calling Groq…"):
                    for row in st.session_state["ranked"]:
                        # Find the resume text by matching the filename
                        resume_text = next(r["text"] for r in st.session_state["resumes"] if r["id"] == row["filename"])
                        row["summary"] = generate_fit_summary(
                            jd,
                            resume_text,
                        )
                    st.session_state["summaries_done"] = True
                    st.rerun()
    
    with tab2:
        st.subheader("🎯 Skills Coverage Analysis")
        
        # Create skills heatmap
        with st.spinner("Analyzing skills coverage..."):
            heatmap_fig, df_coverage = create_skills_heatmap(
                jd, 
                st.session_state["ranked"],
                st.session_state["resumes"],
                top_n=min(5, len(st.session_state["ranked"]))
            )
        
        if heatmap_fig:
            # Display heatmap
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # Display missing skills analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Skills Gap Overview")
                gap_df = create_skills_gap_analysis(df_coverage)
                st.dataframe(
                    gap_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Coverage": st.column_config.TextColumn("Match %"),
                        "Has": st.column_config.NumberColumn("Skills Found"),
                        "Missing": st.column_config.NumberColumn("Skills Missing"),
                    }
                )
            
            with col2:
                missing_fig = create_missing_skills_chart(df_coverage)
                if missing_fig:
                    st.plotly_chart(missing_fig, use_container_width=True)
            
            # Expandable detailed view
            with st.expander("🔍 Detailed Skills Analysis per Candidate"):
                for idx, row in df_coverage.head(5).iterrows():
                    st.markdown(f"### {row['candidate']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**✅ Has these skills:**")
                        if row['matched_skills']:
                            for skill in row['matched_skills']:
                                st.caption(f"• {skill}")
                        else:
                            st.caption("No matching skills found")
                    
                    with col2:
                        st.markdown("**❌ Missing these skills:**")
                        if row['missing_skills']:
                            for skill in row['missing_skills']:
                                st.caption(f"• {skill}")
                        else:
                            st.caption("Has all required skills!")
                    
                    st.divider()
            
            # Insights
            st.info(
                f"💡 **Insights:** The top candidate has {df_coverage.iloc[0]['coverage_percentage']:.0f}% "
                f"skill coverage. The most commonly missing skill across candidates is "
                f"'{df_coverage.iloc[0]['missing_skills'][0] if df_coverage.iloc[0]['missing_skills'] else 'None'}'."
            )
        else:
            st.warning("Unable to extract skills from job description for analysis")
    
    with tab3:
        # Your existing analytics (similarity distribution, etc.)
        st.subheader("📈 Ranking Analytics")
        
        # Similarity distribution
        similarities = [r["similarity"] for r in st.session_state["ranked"]]
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=similarities,
            nbinsx=20,
            marker_color='lightblue',
            name='Candidates'
        ))
        fig_dist.add_vline(x=0.7, line_dash="dash", line_color="green", 
                          annotation_text="Good Match")
        fig_dist.update_layout(
            title="Distribution of Match Scores",
            xaxis_title="Similarity Score",
            yaxis_title="Count",
            xaxis=dict(tickformat='.0%')
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab4:
        # Export functionality
        st.subheader("📋 Export Results")
        
        # Prepare export data with skills analysis
        export_df = df.copy()
        if 'df_coverage' in locals() and df_coverage is not None and not df_coverage.empty:
            # Add skills coverage to export
            coverage_dict = df_coverage.set_index('candidate').to_dict()
            export_df['skills_coverage'] = export_df['id'].map(
                lambda x: coverage_dict['coverage_percentage'].get(x, 0)
            )
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Complete Analysis (CSV)",
            data=csv,
            file_name="candidate_analysis_with_skills.csv",
            mime="text/csv"
        )

st.caption("SproutsAI demo — built by Anish")
