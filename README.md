![CI](https://img.shields.io/github/actions/workflow/status/DarkEnough/resume-ranker/ci.yml?branch=main&label=CI)

# 🎯 Candidate Recommendation Engine

An intelligent resume screening system that uses advanced NLP and transformer models to match candidates with job descriptions, built for the SproutsAI ML Engineer Internship assignment.

## 🌟 Key Features

### Core Functionality
- **Smart Resume Ranking**: Uses semantic similarity to rank candidates based on job description relevance
- **Multi-Format Support**: Processes PDF, DOCX, and TXT resume formats
- **Advanced Skill Extraction**: Leverages LinkedIn's NER model for accurate skill recognition
- **AI-Powered Summaries**: Generates explanations for why each candidate matches using Groq LLM

### Advanced Analytics
- **Skills Coverage Heatmap**: Visual matrix showing which candidates have which required skills
- **Gap Analysis**: Identifies missing skills across top candidates
- **Match Score Distribution**: Statistical visualization of candidate pool quality
- **Export Functionality**: Download complete analysis as CSV for further processing

## 🚀 Live Demo

**App URL**: https://resume-ranker-e7vxsgq6xsdx9mfe64vaje.streamlit.app

## 🛠️ Technical Architecture

### Approach

1. **Intelligent Preprocessing**
   - Removes irrelevant sections from JDs (benefits, EEO statements, company descriptions)
   - Focuses on technical requirements and responsibilities

2. **Dual-Layer Matching System**
   - **Document-level similarity** (35%): Overall semantic match using sentence-transformers
   - **Skill-focused similarity** (65%): Targeted matching on skill-relevant sections
   - **Skill bonus** (up to 10%): Rewards candidates with higher skill coverage

3. **Transformer-Based Skill Extraction**
   - Uses `algiraldohe/lm-ner-linkedin-skills-recognition` model
   - Processes both JDs and resumes with the same model for fair comparison
   - Handles BERT's token limits through intelligent chunking

4. **Production-Ready Design**
   - Thread-safe embedding with caching
   - Graceful API failure handling
   - Efficient batch processing

### Tech Stack

- **Frontend**: Streamlit
- **ML/NLP**: 
  - Sentence-Transformers (all-mpnet-base-v2)
  - Hugging Face Transformers (LinkedIn Skills NER)
  - NumPy, Pandas
- **Visualization**: Plotly
- **LLM Integration**: Groq API (llama-3.3-70b)
- **Document Processing**: pdfplumber, python-docx

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/resume-ranker.git
cd resume-ranker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional, for AI summaries):
```bash
# Create .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

4. Run the application:
```bash
streamlit run app.py
```

## 📊 How It Works

### Ranking Algorithm

```python
# Weighted scoring formula
combined_similarity = 0.35 * full_document_similarity + 
                     0.65 * skill_section_similarity
skill_bonus = skill_match_rate * 0.1
final_score = min(combined_similarity + skill_bonus, 1.0)
```

### Why This Approach?

- **Semantic Understanding**: Goes beyond keyword matching to understand context
- **Skill-Centric**: Prioritizes technical skills while considering overall fit
- **Fair Comparison**: Uses same NER model for both JDs and resumes
- **Actionable Insights**: Provides specific skill gaps for informed decision-making

## 🎮 Usage Guide

1. **Input Job Description**: Paste or type the JD in the text area
2. **Upload Resumes**: Select multiple resume files (PDF/DOCX/TXT)
3. **Set Top-N**: Choose how many top candidates to display
4. **Rank Candidates**: Click to process and rank
5. **Explore Results**:
   - **Rankings Tab**: View similarity scores and generate AI summaries
   - **Skills Analysis Tab**: Explore skill coverage heatmap and gaps
   - **Analytics Tab**: Review score distribution
   - **Export Tab**: Download results as CSV

## 🔑 Key Assumptions

- **Language**: Resumes and JDs are in English
- **Format**: Resumes follow standard formatting (contact info → experience → skills)
- **Skills Focus**: Technical skills are primary differentiators for ranking
- **Context Window**: First 1500 characters of skill-relevant sections are most important

## 🚧 Known Limitations

- Maximum 30 resumes per batch (configurable)
- 5MB file size limit per resume
- Groq API required for AI summaries (falls back gracefully)
- Processing time scales linearly with number of resumes

## 🔮 Future Enhancements

- **Experience Level Detection**: Automatic classification of junior/senior candidates
- **Salary Expectation Analysis**: Extract and match compensation requirements
- **Location Matching**: Consider geographical preferences and remote work options
- **Redis Caching**: Store embeddings for frequently processed resumes
- **Batch Processing API**: REST endpoint for programmatic access
- **Fine-tuned Models**: Domain-specific models for different industries

## 📄 License

MIT License - feel free to use this code for your own projects


## 🙏 Acknowledgments

- SproutsAI team for the interesting challenge
- Hugging Face for transformer models
- Streamlit for the amazing framework

---

*Built with ❤️ for SproutsAI*
