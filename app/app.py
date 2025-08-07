"""
Streamlit v0.3.1  –  summaries only on explicit button press
"""

from __future__ import annotations
import tempfile
from pathlib import Path
from typing import List, Dict

import pandas as pd
import streamlit as st

from core_pipeline import (
    extract_text,
    Embedder,
    rank_candidates,
    generate_fit_summary,
    summaries_available,
    clean_job_description,
)

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
    df = pd.DataFrame(st.session_state["ranked"])
    st.subheader("Top candidates")
    st.dataframe(
        df[["id", "similarity"] + (["summary"] if "summary" in df.columns else [])],
        hide_index=True,
        use_container_width=True,
    )
    st.download_button(
        label="⬇️  Export results as CSV",
        data=df.to_csv(index=False),
        file_name="candidate_ranking.csv",
        mime="text/csv",
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

st.caption("SproutsAI demo — built by Anish")
