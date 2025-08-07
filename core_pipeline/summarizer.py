from __future__ import annotations

import os
from typing import Final
from .snippetizer import top_k_snippets, Embedder

from dotenv import load_dotenv

load_dotenv()

_GROQ_KEY: Final[str | None] = os.getenv("GROQ_API_KEY")
_ENABLED: Final[bool] = True

_client = None
if _GROQ_KEY:  # Only try to use Groq if API key is available
    try:
        import groq
        _client = groq.Client(api_key=_GROQ_KEY)
    except ImportError:
        print("Warning: groq package not installed. Using fallback summaries.")


def summaries_available() -> bool:
    """Utility for UI layers to know whether summaries can be generated."""
    return True


def generate_fit_summary(job_desc: str, resume_text: str, *, k_snippets: int = 5) -> str:
    """
    Use Groq to craft a 2-sentence rationale *based on the K most-relevant
    snippets*, not the whole résumé.
    """
    # ── 1.  extract evidence sentences  ──────────────────────────────────
    snippets = top_k_snippets(job_desc, resume_text, k=k_snippets, embedder=Embedder())
    if not snippets:
        snippets = [resume_text[:400]]        # fallback: first 400 chars

    evidence = " • ".join(snippets)

    # Check if we can use Groq API
    if _client is not None:
        # ── 2.  build LLM prompt  ────────────────────────────────────────────
        prompt = (
        "You are a recruiting assistant. In 2-3 concise sentences, explain why "
        "this candidate is a strong match for the role. Specifically mention:\n"
        "1. Which technical skills/technologies from the job description they possess\n"
        "2. Relevant experience or achievements that align with the role\n"
        "Base your answer ONLY on the evidence provided.\n\n"
        "JOB DESCRIPTION:\n"
        f"{job_desc}\n\n"
        "EVIDENCE FROM RESUME (most relevant sections):\n"
        f"{evidence}\n\n"
        "Focus on specific skill matches and experiences. Be concrete, not generic."
        )

        # ── 3.  call Groq  ──────────────────────────────────────────────────
        try:
            response = _client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
            # Fall through to fallback summary
    
    # Fallback summary when API key is not available or Groq fails
    evidence = " • ".join(snippets[:2])  # Use first 2 snippets
    return f"Strong candidate based on relevant experience. Key highlights: {evidence[:200]}..."
