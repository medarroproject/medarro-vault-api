import os
import json
from typing import List
import re
from datetime import datetime, date, timedelta

import fitz  # PyMuPDF
import httpx
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()

# ---------------------------------------------------------------------------
# TWO GEMINI KEYS
# GEMINI_API_KEY     → AI queries
# new_gemini_api_key → Study plan
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_STUDY_PLAN_KEY = os.getenv("new_gemini_api_key")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY required")
if not GEMINI_STUDY_PLAN_KEY:
    raise ValueError("❌ new_gemini_api_key required")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Primary Gemini configured")
except Exception as e:
    raise RuntimeError(f"❌ Gemini config error: {e}")

try:
    st_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    print("✅ FastEmbed loaded (384 dims)")
except Exception as e:
    raise RuntimeError(f"❌ FastEmbed error: {e}")

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY"),
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Medarro API",
    description="AI-powered medical education backend.",
    version="5.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "http://localhost:5173",
        "http://localhost:3000",
        "https://*.lovable.app",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class UploadPDFRequest(BaseModel):
    user_id: str
    pdf_url: str
    pdf_name: str

class UploadPDFResponse(BaseModel):
    status: str
    chunks_count: int

class SearchVaultRequest(BaseModel):
    user_id: str
    query: str

class VaultSearchResult(BaseModel):
    chunk_text: str
    pdf_name: str
    page_number: int
    similarity: float

class SearchVaultResponse(BaseModel):
    results: List[VaultSearchResult]

class AiQueryRequest(BaseModel):
    query: str
    user_id: str = None
    mode: str = "deep-explanation"
    track: str = "NEET"

class AiQueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

class QueryRequest(BaseModel):
    query: str
    mode: str = "deep-explanation"
    track: str = "NEET"
    context: str = ""  # For vault answers

class StudyPlanRequest(BaseModel):
    user_id: str
    target_exam: str = "NEET UG"
    target_date: str
    daily_hours: int = 6
    weak_subjects: List[str] = []
    current_progress: dict = {}
    track: str = "NEET"

class StudyTask(BaseModel):
    subject: str
    topic: str
    duration_minutes: int
    mode: str
    priority: str
    time_slot: str

class StudyPlanDay(BaseModel):
    day: str
    date: str
    total_hours: float
    tasks: List[StudyTask]

class StudyPlanResponse(BaseModel):
    plan: List[StudyPlanDay]
    weekly_subject_split: dict
    ai_insight: str
    total_days_remaining: int

# ---------------------------------------------------------------------------
# PDF Helpers
# ---------------------------------------------------------------------------

CHUNK_WORDS = 500
OVERLAP_WORDS = 100

def extract_pages(pdf_bytes: bytes) -> List[dict]:
    pages = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i in range(len(doc)):
            text = doc[i].get_text("text")
            if text.strip():
                pages.append({"page_number": i + 1, "text": text})
        doc.close()
    except Exception as e:
        print(f"❌ PDF extract error: {e}")
        raise
    return pages

def split_into_chunks(pages: List[dict]) -> List[dict]:
    pairs = []
    for p in pages:
        for w in p["text"].split():
            pairs.append((w, p["page_number"]))
    if not pairs:
        return []
    chunks, step, start = [], CHUNK_WORDS - OVERLAP_WORDS, 0
    while start < len(pairs):
        end = min(start + CHUNK_WORDS, len(pairs))
        window = pairs[start:end]
        chunks.append({
            "chunk_text": " ".join(w for w, _ in window),
            "page_number": window[0][1]
        })
        if end == len(pairs):
            break
        start += step
    return chunks

def get_embedding(text: str) -> List[float]:
    try:
        return list(st_model.embed([text]))[0].tolist()
    except Exception as e:
        raise HTTPException(503, f"Embedding error: {e}")

async def download_pdf(url: str) -> bytes:
    async with httpx.AsyncClient(
        timeout=120.0, follow_redirects=True
    ) as c:
        r = await c.get(url)
        if r.status_code != 200:
            raise HTTPException(400, f"PDF download failed: {r.status_code}")
        return r.content

def clean_markdown(text: str) -> str:
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return '\n'.join(l.strip() for l in text.split('\n')).strip()

# ---------------------------------------------------------------------------
# Medical Books
# ---------------------------------------------------------------------------

MEDICAL_BOOKS = {
    "NEET": [
        "NCERT Biology Class 11 & 12",
        "Trueman's Objective Biology",
        "DC Pandey Physics",
        "Physical Chemistry - OP Tandon"
    ],
    "MBBS": [
        "Gray's Anatomy (Latest Edition)",
        "Guyton & Hall Physiology",
        "Robbins & Cotran Pathology",
        "BD Chaurasia's Human Anatomy",
        "Harrison's Principles of Internal Medicine",
        "KD Tripathi Pharmacology"
    ],
    "BDS": [
        "Shafer's Textbook of Oral Pathology",
        "Gray's Anatomy",
        "Dental Pharmacology",
        "Guyton & Hall Physiology"
    ],
    "BHMS": [
        "Boericke's Materia Medica",
        "Organon of Medicine",
        "Gray's Anatomy",
        "Guyton Physiology"
    ]
}

# ---------------------------------------------------------------------------
# Clinical Guidelines
# ---------------------------------------------------------------------------

CLINICAL_GUIDELINES = """
MANDATORY CLINICAL GUIDELINES:
- TB: WHO 2022 — HRZE 2mo + HR 4mo
  H=Isoniazid 5mg/kg, R=Rifampicin 10mg/kg
  Z=Pyrazinamide 25mg/kg, E=Ethambutol 15mg/kg
- DM Type 2: ICMR 2025 — Metformin first, HbA1c <7%
- HTN: JNC 8 — <140/90, <130/80 DM/CKD
- DKA: NS 15-20ml/kg/hr, Insulin 0.1U/kg/hr
- Malaria: P.vivax=Chloroquine+Primaquine 14d
  P.falciparum=ACT Artesunate-Lumefantrine
"""

# ---------------------------------------------------------------------------
# Mode Instructions
# ---------------------------------------------------------------------------

MODE_INSTRUCTIONS = {

    # ── Vault Answer Mode — NEW ─────────────────────────────────────────────
    "vault-answer": """
You are a MBBS topper sharing notes with a friend.
Be concise, exam-focused, natural language.
Max 300 words total. No paragraphs. No textbook language.

FORMAT EXACTLY — ALL sections required:

FINAL ANSWER:
[2-3 lines. Direct answer only. Natural language.]

HIGH-YIELD POINTS:
1. [Exam-relevant point]
2. [Exam-relevant point]
3. [Exam-relevant point]
4. [Exam-relevant point]
5. [Exam-relevant point]

CLINICAL PEARL:
[One practical clinical insight. 1-2 lines max.]

MNEMONIC:
[Easy mnemonic if applicable. Skip if not relevant.]

VIVA/PYQ POINT:
[Exactly how this is asked in exams. 1 line.]

ONE-LINE RECALL:
[Single sentence. Most important thing to remember.]

RULES:
- Max 300 words
- Topper notes style
- No academic wording
- Exam focused only
- Complete all sections
""",

    # ── AI Study Workspace Modes ────────────────────────────────────────────

    "deep-explanation": """
COMPLETE answer in ALL 6 sections. Never truncate.

1. DEFINITION: One precise line.

2. MECHANISM / PATHOPHYSIOLOGY:
Complete step-by-step mechanism.
Include all pathway components.

3. CLINICAL FEATURES:
Signs, symptoms, investigations with values.

4. TREATMENT PROTOCOL:
Drug names + doses + duration.
Follow WHO/ICMR/Indian guidelines.

5. CLINICAL PEARLS:
Key exam points, differentials, mnemonics.

6. REFERENCE: Textbook + chapter.

Write until ALL 6 sections complete.
""",

    "quick-summary": """
COMPLETE all sections. Max 200 words.

KEY FACTS:
- [Fact 1 - max 15 words]
- [Fact 2]
- [Fact 3]
- [Fact 4]
- [Fact 5]
- [Fact 6]
- [Fact 7]

MNEMONIC: [One easy mnemonic]

TREATMENT: [Key drug + dose in 1 line]

MUST REMEMBER: [#1 high yield fact]
""",

    "mcq-practice": """
Generate EXACTLY 5 MCQs. Complete ALL 5.
NEET/MBBS exam pattern.

Q1. [Question]
A. [Option]
B. [Option]
C. [Option]
D. [Option]
Answer: [Letter]
Explanation: [Correct + why others wrong]

Q2. [Question]
A. [Option]
B. [Option]
C. [Option]
D. [Option]
Answer: [Letter]
Explanation: [Correct + why others wrong]

Q3. [Question]
A. [Option]
B. [Option]
C. [Option]
D. [Option]
Answer: [Letter]
Explanation: [Correct + why others wrong]

Q4. [Question]
A. [Option]
B. [Option]
C. [Option]
D. [Option]
Answer: [Letter]
Explanation: [Correct + why others wrong]

Q5. [Question]
A. [Option]
B. [Option]
C. [Option]
D. [Option]
Answer: [Letter]
Explanation: [Correct + why others wrong]

High yield. Clinical scenarios included.
WHO/ICMR/NCERT based.
""",

    "rapid-recall": """
Complete ALL sections. No truncation.
If LIST asked — give COMPLETE list.

DEFINITION: [1 precise line]

COMPLETE LIST / CLASSIFICATION:
1. [Item 1]
2. [Item 2]
3. [Item 3]
4. [Item 4]
5. [Item 5]
(Continue until complete)

KEY POINTS:
- [Point 1]
- [Point 2]
- [Point 3]
- [Point 4]
- [Point 5]

MNEMONIC: [Easy mnemonic]

FLOWCHART:
[Step 1] → [Step 2] → [Step 3] → [Step 4]

TREATMENT: [Drug + Dose + Duration]

EXAM TIP: [How asked in NEET/MBBS]

HIGH YIELD FACT: [#1 to remember]
""",

    # ── Legacy ──────────────────────────────────────────────────────────────
    "explanation": "Thorough explanation: definition, mechanism, treatment.",
    "exam": "Concise exam answer with key points and treatment.",
    "revision": "Bullet point revision notes with mnemonics.",
    "notes": "Detailed notes with subtopics and clinical correlations.",
    "deep-dive": "Research-level: mechanisms, guidelines, landmark studies."
}

# ---------------------------------------------------------------------------
# Token limits per mode — Speed optimization
# ---------------------------------------------------------------------------

MODE_TOKENS = {
    "vault-answer": 800,      # Fast — short format
    "quick-summary": 1000,    # Medium
    "rapid-recall": 1200,     # Medium
    "mcq-practice": 2000,     # Needs 5 MCQs
    "deep-explanation": 3000, # Full explanation
    "explanation": 2000,
    "exam": 1000,
    "revision": 800,
    "notes": 2000,
    "deep-dive": 3000,
}

# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

def build_medical_prompt(
    query: str,
    mode: str = "deep-explanation",
    track: str = "NEET",
    context: str = ""
) -> str:

    instruction = MODE_INSTRUCTIONS.get(
        mode, MODE_INSTRUCTIONS["deep-explanation"]
    )
    books = MEDICAL_BOOKS.get(track, MEDICAL_BOOKS["MBBS"])
    books_text = "\n".join(f"- {b}" for b in books)

    # Vault context section
    context_section = ""
    if context:
        context_section = f"""
CONTEXT FROM STUDENT'S UPLOADED NOTES:
{context}

Answer based primarily on this context.
"""

    return f"""You are an expert medical educator for Indian {track} students.

RULES:
1. ONLY standard medical textbooks
2. EXACT medical terminology
3. NEVER truncate — complete ALL sections
4. Follow Indian medical curriculum
5. NO Markdown (no ##, **, __)
6. Plain text ONLY
{CLINICAL_GUIDELINES}
STUDENT LEVEL: {track}
MODE: {mode}
{context_section}
FORMAT:
{instruction}

TEXTBOOKS:
{books_text}

QUESTION: {query}

Write COMPLETE answer. Do NOT stop early."""

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    status = {
        "status": "ok",
        "service": "Medarro API",
        "version": "5.1.0",
        "apis": {
            "gemini_query": "checking",
            "gemini_study_plan": "checking",
            "embeddings": "checking",
            "supabase": "ok"
        },
        "improvements": {
            "vault_answer_mode": "added",
            "speed_optimization": "per_mode_tokens",
            "vault_format": "topper_notes_style",
            "truncation_fix": "enabled",
            "clinical_guidelines": "enabled"
        },
        "warnings": []
    }

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        m = genai.GenerativeModel("models/gemini-2.5-flash")
        r = m.generate_content("Say OK")
        status["apis"]["gemini_query"] = "ok" if r.text else "degraded"
    except Exception as e:
        status["apis"]["gemini_query"] = "down"
        status["warnings"].append(f"Primary: {str(e)[:60]}")

    try:
        genai.configure(api_key=GEMINI_STUDY_PLAN_KEY)
        m2 = genai.GenerativeModel("models/gemini-2.5-flash")
        r2 = m2.generate_content("Say OK")
        status["apis"]["gemini_study_plan"] = "ok" if r2.text else "degraded"
    except Exception as e:
        status["apis"]["gemini_study_plan"] = "down"
        status["warnings"].append(f"Study plan: {str(e)[:60]}")
    finally:
        genai.configure(api_key=GEMINI_API_KEY)

    try:
        emb = get_embedding("test")
        status["apis"]["embeddings"] = f"ok ({len(emb)} dims)"
    except Exception as e:
        status["apis"]["embeddings"] = "down"
        status["warnings"].append(f"Embedding: {str(e)[:60]}")

    return status

# ---------------------------------------------------------------------------
# PDF Upload
# ---------------------------------------------------------------------------

@app.post("/upload-pdf", response_model=UploadPDFResponse)
async def upload_pdf(request: UploadPDFRequest):
    try:
        pdf_bytes = await download_pdf(request.pdf_url)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Download error: {e}")

    pages = extract_pages(pdf_bytes)
    if not pages:
        raise HTTPException(422, "No text in PDF")

    chunks = split_into_chunks(pages)
    if not chunks:
        raise HTTPException(422, "No chunks generated")

    inserted = 0
    for chunk in chunks:
        emb = get_embedding(chunk["chunk_text"])
        result = supabase.table("personal_vault").insert({
            "user_id": request.user_id,
            "pdf_name": request.pdf_name,
            "page_number": chunk["page_number"],
            "chunk_text": chunk["chunk_text"],
            "embedding": emb,
        }).execute()
        if not result.data:
            raise HTTPException(500, "Supabase insert failed")
        inserted += 1

    return UploadPDFResponse(status="ready", chunks_count=inserted)

# ---------------------------------------------------------------------------
# Vault Search
# ---------------------------------------------------------------------------

@app.post("/search-vault", response_model=SearchVaultResponse)
async def search_vault(request: SearchVaultRequest):
    try:
        qe = get_embedding(request.query)
        rpc = supabase.rpc("match_vault_chunks", {
            "query_embedding": qe,
            "match_user_id": request.user_id,
            "match_count": 3,  # Reduced for speed
        }).execute()
        if rpc.data:
            return SearchVaultResponse(results=[
                VaultSearchResult(
                    chunk_text=r["chunk_text"],
                    pdf_name=r["pdf_name"],
                    page_number=r["page_number"],
                    similarity=r["similarity"]
                ) for r in rpc.data
            ])
    except Exception as e:
        print(f"⚠️ Vector search failed: {e}")

    # Fallback
    try:
        resp = (
            supabase.table("personal_vault")
            .select("chunk_text,pdf_name,page_number")
            .eq("user_id", request.user_id)
            .full_text_search("chunk_text", request.query)
            .limit(3)
            .execute()
        )
        if resp.data:
            return SearchVaultResponse(results=[
                VaultSearchResult(
                    chunk_text=r["chunk_text"],
                    pdf_name=r["pdf_name"],
                    page_number=r["page_number"],
                    similarity=0.85
                ) for r in resp.data
            ])
        return SearchVaultResponse(results=[])
    except Exception as e:
        raise HTTPException(503, f"Search unavailable: {str(e)[:100]}")

# ---------------------------------------------------------------------------
# AI Query — Speed optimized with per-mode tokens
# ---------------------------------------------------------------------------

@app.post("/query", response_model=AiQueryResponse)
async def gemini_query(request: QueryRequest):
    genai.configure(api_key=GEMINI_API_KEY)

    # Get token limit for this mode
    max_tokens = MODE_TOKENS.get(request.mode, 2000)

    prompt = build_medical_prompt(
        query=request.query,
        mode=request.mode,
        track=request.track,
        context=request.context
    )

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.5,
                "top_p": 0.9,
                "max_output_tokens": max_tokens,
            }
        )
        if not response.text:
            raise HTTPException(500, "Empty Gemini response")

        answer = clean_markdown(response.text)

        kws = {
            "gray": "Gray's Anatomy",
            "guyton": "Guyton & Hall Physiology",
            "robbins": "Robbins Pathology",
            "chaurasia": "BD Chaurasia Anatomy",
            "tripathi": "KD Tripathi Pharmacology",
            "harrison": "Harrison's Principles",
            "ncert": "NCERT Biology",
            "netter": "Netter's Anatomy",
            "who": "WHO Guidelines",
            "icmr": "ICMR Guidelines",
        }
        al = answer.lower()
        sources = list(dict.fromkeys(
            [v for k, v in kws.items() if k in al]
        )) or ["Standard Medical References"]

        return AiQueryResponse(
            answer=answer,
            sources=sources,
            confidence=0.95
        )

    except Exception as e:
        raise HTTPException(500, f"Gemini error: {e}")

@app.post("/search")
async def gemini_ai_search(request: AiQueryRequest):
    return await gemini_query(QueryRequest(
        query=request.query,
        mode=request.mode,
        track=request.track
    ))

# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

@app.post("/query-stream")
async def gemini_query_stream(request: QueryRequest):
    genai.configure(api_key=GEMINI_API_KEY)
    max_tokens = MODE_TOKENS.get(request.mode, 2000)
    prompt = build_medical_prompt(
        request.query, request.mode, request.track
    )

    async def generate():
        try:
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            for chunk in model.generate_content(
                prompt, stream=True,
                generation_config={
                    "temperature": 0.5,
                    "max_output_tokens": max_tokens
                }
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"Error: {e}"

    return StreamingResponse(generate(), media_type="text/plain")

# ---------------------------------------------------------------------------
# Study Plan — SECONDARY KEY
# ---------------------------------------------------------------------------

@app.post("/generate-study-plan", response_model=StudyPlanResponse)
async def generate_study_plan(request: StudyPlanRequest):

    genai.configure(api_key=GEMINI_STUDY_PLAN_KEY)

    try:
        days_remaining = (
            datetime.strptime(request.target_date, "%Y-%m-%d").date()
            - date.today()
        ).days
    except Exception:
        days_remaining = 90

    subjects_map = {
        "NEET": ["Biology", "Physics", "Chemistry"],
        "MBBS": ["Anatomy", "Physiology", "Biochemistry",
                 "Pharmacology", "Pathology", "Microbiology"],
        "BDS": ["Anatomy", "Physiology", "Biochemistry",
                "Dental Materials", "Oral Pathology"],
        "BHMS": ["Organon of Medicine", "Materia Medica",
                 "Anatomy", "Physiology"]
    }
    subjects = subjects_map.get(request.track, subjects_map["MBBS"])
    weak = ", ".join(request.weak_subjects) or "Not specified"

    # STEP 1: Gemini se sirf topics maango (plain text, no JSON)
    topic_prompt = f"""List 14 important medical topics for {request.track} student.
Prioritize these weak subjects: {weak}
From these subjects: {', '.join(subjects)}
Return ONLY comma-separated topic names.
Example format: Brachial Plexus, Starling Law, Beta Blockers MOA
No numbering. No explanation. No JSON. Just topic names."""

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        topic_resp = model.generate_content(
            topic_prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 300
            }
        )
        raw_topics = topic_resp.text.strip()
        topics = [t.strip() for t in raw_topics.split(',') if t.strip()]
        if len(topics) < 7:
            # Fallback topics
            topics = [
                f"{s} - Key Concepts" for s in subjects * 3
            ]
    except Exception:
        topics = [f"{s} - Key Concepts" for s in subjects * 3]

    # STEP 2: Python mein plan banao (no JSON from Gemini)
    today_date = date.today()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    modes_cycle = [
        "deep-explanation", "quick-summary",
        "mcq-practice", "rapid-recall"
    ]

    plan = []
    topic_idx = 0

    for i in range(7):
        d = today_date + timedelta(days=i)
        tasks_per_day = max(2, min(4, request.daily_hours // 2))
        tasks = []

        for j in range(tasks_per_day):
            subj = subjects[topic_idx % len(subjects)]
            topic = topics[topic_idx % len(topics)]
            mode = modes_cycle[j % len(modes_cycle)]
            duration = 90 if j == 0 else 60
            hour = 9 + (j * 2)
            time_slot = f"{hour}:00 AM" if hour < 12 else (
                "12:00 PM" if hour == 12 else f"{hour-12}:00 PM"
            )
            tasks.append({
                "subject": subj,
                "topic": topic,
                "duration_minutes": duration,
                "mode": mode,
                "priority": "high" if subj in weak else "medium",
                "time_slot": time_slot
            })
            topic_idx += 1

        plan.append({
            "day": day_names[d.weekday()],
            "date": d.strftime("%Y-%m-%d"),
            "total_hours": float(request.daily_hours),
            "tasks": tasks
        })

    # STEP 3: Subject split calculate karo
    n = len(subjects)
    base = 100 // n
    remainder = 100 - (base * n)
    weekly_split = {}
    for idx, s in enumerate(subjects):
        weekly_split[s] = base + (1 if idx < remainder else 0)

    # Weak subjects ko extra time
    for s in request.weak_subjects:
        if s in weekly_split:
            weekly_split[s] = min(35, weekly_split[s] + 5)

    return StudyPlanResponse(
        plan=plan,
        weekly_subject_split=weekly_split,
        ai_insight=(
            f"You have {days_remaining} days left for "
            f"{request.target_exam}. "
            f"Focus on {weak} — AI has prioritized these "
            f"in your daily schedule. "
            f"Study {request.daily_hours} hours daily to stay on track."
        ),
        total_days_remaining=days_remaining
    )

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

@app.get("/tracks")
async def get_tracks():
    return {
        "tracks": ["NEET", "MBBS", "BHMS", "BDS", "MD/MS"],
        "modes": list(MODE_INSTRUCTIONS.keys()),
        "lovable_modes": [
            "deep-explanation", "quick-summary",
            "mcq-practice", "rapid-recall"
        ],
        "vault_mode": "vault-answer"
    }

@app.get("/books")
async def get_books():
    return MEDICAL_BOOKS

@app.get("/status")
async def api_status():
    return {
        "version": "5.1.0",
        "vault_answer_mode": "enabled",
        "speed": "per_mode_tokens",
        "vault_tokens": 800,
        "deep_explanation_tokens": 3000,
        "mcq_tokens": 2000,
        "clinical_guidelines": "enabled",
        "dual_gemini_keys": "enabled"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=90)
