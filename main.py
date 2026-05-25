import os
import json
from typing import List
import re
from datetime import datetime, date, timedelta

import fitz
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
# KEYS
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_STUDY_PLAN_KEY = os.getenv("new_gemini_api_key")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY required")
if not GEMINI_STUDY_PLAN_KEY:
    raise ValueError("new_gemini_api_key required")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini configured")
except Exception as e:
    raise RuntimeError(f"Gemini error: {e}")

try:
    st_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    print("✅ FastEmbed loaded")
except Exception as e:
    raise RuntimeError(f"FastEmbed error: {e}")

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY"),
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Medarro API", version="6.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    context: str = ""

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
        print(f"PDF error: {e}")
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
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as c:
        r = await c.get(url)
        if r.status_code != 200:
            raise HTTPException(400, f"PDF download failed: {r.status_code}")
        return r.content

def clean_text(text: str) -> str:
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return '\n'.join(l.strip() for l in text.split('\n')).strip()

# ---------------------------------------------------------------------------
# Medical Data
# ---------------------------------------------------------------------------
MEDICAL_BOOKS = {
    "NEET": ["NCERT Biology 11&12", "Trueman Biology", "DC Pandey Physics", "OP Tandon Chemistry"],
    "MBBS": ["Gray's Anatomy", "Guyton Physiology", "Robbins Pathology", "BD Chaurasia", "Harrison's", "KD Tripathi"],
    "BDS": ["Shafer's Oral Pathology", "Gray's Anatomy", "Dental Pharmacology", "Guyton Physiology"],
    "BHMS": ["Boericke Materia Medica", "Organon of Medicine", "Gray's Anatomy", "Guyton Physiology"]
}

GUIDELINES = "TB:HRZE(H5R10Z25E15mg/kg)2+4mo|DM2:Metformin,HbA1c<7%|HTN:JNC8<140/90|DKA:NS15-20ml+Insulin0.1U/kg|Malaria:Vivax=CQ+Prima,Falci=ACT"

# ---------------------------------------------------------------------------
# PROMPTS
# Token saving strategy:
# - Remove all example placeholders
# - Remove verbose instructions
# - Keep only essential structure
# - More tokens left = longer complete answers
# ---------------------------------------------------------------------------

def build_prompt(query: str, mode: str, track: str, context: str = "") -> str:

    ctx = f"\nNOTES CONTEXT:\n{context[:600]}\n" if context else ""
    books = ", ".join(MEDICAL_BOOKS.get(track, MEDICAL_BOOKS["MBBS"])[:3])

    # ── VAULT ANSWER ────────────────────────────────────────────────────────
    if mode == "vault-answer":
        return (
            f"You are a {track} topper writing revision notes. Max 250 words."
            f"{ctx}\n"
            f"Q: {query}\n\n"
            "ANSWER:\nKEY POINTS:\n1.\n2.\n3.\n4.\n5.\n"
            "PEARL:\nMNEMONIC:\nEXAM TIP:\nRECALL:"
        )

    # ── QUICK SUMMARY ────────────────────────────────────────────────────────
    if mode == "quick-summary":
        return (
            f"You are a {track} expert. Concise revision. Max 180 words."
            f"{ctx}\n"
            f"Q: {query}\n\n"
            "KEY FACTS:\n-\n-\n-\n-\n-\n-\n-\n"
            "MNEMONIC:\nTREATMENT:\nMUST KNOW:"
        )

    # ── MCQ PRACTICE ─────────────────────────────────────────────────────────
    if mode == "mcq-practice":
        return (
            f"Senior {track} examiner. Generate 5 clinical MCQs on: {query}"
            f"{ctx}\n"
            "Return ONLY JSON array, no other text:\n"
            '[{"q":"<clinical scenario question>","options":{"A":"<opt>","B":"<opt>","C":"<opt>","D":"<opt>"},"correct":"<letter>","reason_correct":"<1-2 line why correct>","reason_wrong":{"<wrong1>":"<why>","<wrong2>":"<why>","<wrong3>":"<why>"}},{"q":"Q2..."},{"q":"Q3..."},{"q":"Q4..."},{"q":"Q5..."}]\n'
            f"Track: {track}. University/PG standard. One best answer. Plausible distractors."
        )

    # ── RAPID RECALL ─────────────────────────────────────────────────────────
    if mode == "rapid-recall":
        return (
            f"Rapid recall card for {track} student."
            f"{ctx}\n"
            f"Topic: {query}\n\n"
            "DEF:\nCLASSIFICATION/LIST:\n1.\n2.\n3.\n4.\n5.\n"
            "KEY POINTS:\n-\n-\n-\nMNEMONIC:\n"
            "FLOW: →→→\nTREATMENT:\nEXAM TIP:\nRECALL:"
        )

    # ── DEEP EXPLANATION (default) ───────────────────────────────────────────
    return (
        f"Medical educator for {track}. Answer completely — never stop mid-sentence.\n"
        f"Guidelines: {GUIDELINES}\n"
        f"Refs: {books}"
        f"{ctx}\n"
        f"Q: {query}\n\n"
        "1.DEFINITION:\n"
        "2.MECHANISM:\n"
        "3.CLINICAL FEATURES:\n"
        "4.TREATMENT:\n"
        "5.PEARLS:\n"
        "6.REFERENCE:"
    )

# ---------------------------------------------------------------------------
# Token limits
# Saving ~200-400 tokens on prompt = ~200-400 more tokens for answer
# ---------------------------------------------------------------------------
MODE_TOKENS = {
    "vault-answer":      900,
    "quick-summary":    1000,
    "rapid-recall":     1500,
    "mcq-practice":     3500,
    "deep-explanation": 2500,
    "explanation":      2000,
    "exam":             1000,
    "revision":          800,
    "notes":            2000,
    "deep-dive":        2500,
}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
FAST_MODEL = os.getenv("GEMINI_FAST_MODEL", "models/gemini-2.0-flash")

MODELS_FALLBACK = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.5-flash-lite-preview-06-17",
]

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    status = {
        "status": "ok",
        "service": "Medarro API",
        "version": "6.1.0",
        "apis": {
            "gemini_query": "checking",
            "gemini_study_plan": "checking",
            "embeddings": "checking",
            "supabase": "ok"
        },
        "warnings": []
    }

    try:
        for m in [PRIMARY_MODEL] + MODELS_FALLBACK:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                r = genai.GenerativeModel(m).generate_content(
                    "OK", generation_config={"max_output_tokens": 5}
                )
                if r.text:
                    status["apis"]["gemini_query"] = f"ok ({m})"
                    break
            except Exception as e:
                status["warnings"].append(f"{m}: {str(e)[:40]}")
    except Exception as e:
        status["apis"]["gemini_query"] = "down"

    try:
        for m in [FAST_MODEL] + MODELS_FALLBACK:
            try:
                genai.configure(api_key=GEMINI_STUDY_PLAN_KEY)
                r = genai.GenerativeModel(m).generate_content(
                    "OK", generation_config={"max_output_tokens": 5}
                )
                if r.text:
                    status["apis"]["gemini_study_plan"] = f"ok ({m})"
                    break
            except Exception as e:
                status["warnings"].append(f"SP {m}: {str(e)[:40]}")
    except Exception as e:
        status["apis"]["gemini_study_plan"] = "down"
    finally:
        genai.configure(api_key=GEMINI_API_KEY)

    try:
        emb = get_embedding("test")
        status["apis"]["embeddings"] = f"ok ({len(emb)} dims)"
    except Exception as e:
        status["apis"]["embeddings"] = "down"
        status["warnings"].append(f"Emb: {str(e)[:40]}")

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
            "match_count": 3,
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
        print(f"Vector search failed: {e}")

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
# AI Query — Smart model fallback
# ---------------------------------------------------------------------------
@app.post("/query", response_model=AiQueryResponse)
async def gemini_query(request: QueryRequest):
    genai.configure(api_key=GEMINI_API_KEY)

    max_tokens = MODE_TOKENS.get(request.mode, 2000)
    prompt = build_prompt(
        query=request.query,
        mode=request.mode,
        track=request.track,
        context=request.context
    )

    last_error = None
    for model_name in [PRIMARY_MODEL] + MODELS_FALLBACK:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": max_tokens,
                }
            )
            if not response.text:
                continue

            answer = clean_text(response.text)

            kws = {
                "gray": "Gray's Anatomy",
                "guyton": "Guyton & Hall",
                "robbins": "Robbins Pathology",
                "chaurasia": "BD Chaurasia",
                "tripathi": "KD Tripathi",
                "harrison": "Harrison's",
                "ncert": "NCERT Biology",
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
            last_error = e
            print(f"Model {model_name} failed: {str(e)[:80]}")
            continue

    raise HTTPException(500, f"All models failed: {last_error}")

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
    prompt = build_prompt(request.query, request.mode, request.track)

    async def generate():
        for model_name in [PRIMARY_MODEL] + MODELS_FALLBACK:
            try:
                model = genai.GenerativeModel(model_name)
                for chunk in model.generate_content(
                    prompt, stream=True,
                    generation_config={
                        "temperature": 0.3,
                        "max_output_tokens": max_tokens
                    }
                ):
                    if chunk.text:
                        yield chunk.text
                return
            except Exception as e:
                print(f"Stream {model_name} failed: {e}")
                continue
        yield "Error: All models unavailable"

    return StreamingResponse(generate(), media_type="text/plain")

# ---------------------------------------------------------------------------
# Study Plan
# ---------------------------------------------------------------------------
@app.post("/generate-study-plan", response_model=StudyPlanResponse)
async def generate_study_plan(request: StudyPlanRequest):

    try:
        days_remaining = (
            datetime.strptime(request.target_date, "%Y-%m-%d").date()
            - date.today()
        ).days
    except Exception:
        days_remaining = 90

    subjects_map = {
        "NEET": ["Biology", "Physics", "Chemistry"],
        "MBBS": ["Anatomy", "Physiology", "Biochemistry", "Pharmacology", "Pathology", "Microbiology"],
        "BDS": ["Anatomy", "Physiology", "Biochemistry", "Dental Materials", "Oral Pathology"],
        "BHMS": ["Organon of Medicine", "Materia Medica", "Anatomy", "Physiology"]
    }
    subjects = subjects_map.get(request.track, subjects_map["MBBS"])
    weak = ", ".join(request.weak_subjects) or "Not specified"

    topic_prompt = (
        f"List 14 {request.track} exam topics. Prioritize: {weak}. "
        f"Subjects: {', '.join(subjects)}. "
        "Return ONLY comma-separated names. No numbers. No explanation."
    )

    topics = []
    for model_name in [FAST_MODEL] + MODELS_FALLBACK:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(
                topic_prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 300}
            )
            topics = [t.strip() for t in resp.text.strip().split(',') if t.strip()]
            if len(topics) >= 7:
                break
        except Exception as e:
            print(f"Topics {model_name} failed: {e}")
            continue

    if len(topics) < 7:
        topics = [f"{s} Key Concepts" for s in subjects * 3]

    today_date = date.today()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    modes_cycle = ["deep-explanation", "quick-summary", "mcq-practice", "rapid-recall"]

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
            if hour < 12:
                time_slot = f"{hour}:00 AM"
            elif hour == 12:
                time_slot = "12:00 PM"
            else:
                time_slot = f"{hour - 12}:00 PM"

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

    n = len(subjects)
    base = 100 // n
    remainder = 100 - (base * n)
    weekly_split = {s: base + (1 if i < remainder else 0) for i, s in enumerate(subjects)}
    for s in request.weak_subjects:
        if s in weekly_split:
            weekly_split[s] = min(35, weekly_split[s] + 5)

    genai.configure(api_key=GEMINI_API_KEY)

    return StudyPlanResponse(
        plan=plan,
        weekly_subject_split=weekly_split,
        ai_insight=(
            f"{days_remaining} days left for {request.target_exam}. "
            f"Focus on {weak}. "
            f"Study {request.daily_hours}h daily."
        ),
        total_days_remaining=days_remaining
    )

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
@app.get("/health-models")
async def list_available_models():
    genai.configure(api_key=GEMINI_API_KEY)
    results = {}
    for m in MODELS_FALLBACK + ["models/gemini-2.5-flash"]:
        try:
            r = genai.GenerativeModel(m).generate_content(
                "OK", generation_config={"max_output_tokens": 5}
            )
            results[m] = "✅ available"
        except Exception as e:
            results[m] = f"❌ {str(e)[:60]}"
    return results

@app.get("/tracks")
async def get_tracks():
    return {
        "tracks": ["NEET", "MBBS", "BHMS", "BDS", "MD/MS"],
        "modes": list(MODE_TOKENS.keys()),
        "version": "6.1.0"
    }

@app.get("/status")
async def api_status():
    return {
        "version": "6.1.0",
        "primary_model": PRIMARY_MODEL,
        "model_fallback": "enabled",
        "prompt_optimization": "enabled",
        "token_savings": "~300 tokens saved per prompt",
        "mcq_fix": "fixed"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=120)
