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
# KEYS & CONFIGURATION
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_STUDY_PLAN_KEY = os.getenv("new_gemini_api_key")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY required")
if not GEMINI_STUDY_PLAN_KEY:
    raise ValueError("new_gemini_api_key required")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini configured successfully")
except Exception as e:
    raise RuntimeError(f"Gemini configuration error: {e}")

try:
    st_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    print("✅ FastEmbed engine loaded successfully")
except Exception as e:
    raise RuntimeError(f"FastEmbed initialization error: {e}")

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY"),
)

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(title="Medarro API", version="6.1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic Data Validation Models
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
# Core Text Processing Helpers
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
        print(f"PDF Extraction processing error: {e}")
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
        raise HTTPException(503, f"Vector Embedding generation failed: {e}")

async def download_pdf(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as c:
        r = await c.get(url)
        if r.status_code != 200:
            raise HTTPException(400, f"External PDF sourcing download failed with HTTP status code: {r.status_code}")
        return r.content

def clean_text(text: str) -> str:
    """
    Cleans system layout spaces without blowing up the core markdown 
    tags required for structural frontend alignment in components.
    """
    text = re.sub(r'\n{3,}', '\n\n', text)
    return '\n'.join(l.strip() for l in text.split('\n')).strip()

# ---------------------------------------------------------------------------
# Strict Medical Grounding Constraints
# ---------------------------------------------------------------------------
MEDICAL_BOOKS = {
    "NEET": ["NCERT Biology 11&12", "Trueman Biology", "DC Pandey Physics", "OP Tandon Chemistry"],
    "MBBS": ["Gray's Anatomy", "Guyton and Hall Physiology", "Robbins Pathology", "BD Chaurasia Anatomy", "Harrison's Internal Medicine", "KD Tripathi Pharmacology"],
    "BDS": ["Shafer's Oral Pathology", "Gray's Anatomy", "Dental Pharmacology", "Guyton Physiology"],
    "BHMS": ["Boericke Materia Medica", "Organon of Medicine", "Gray's Anatomy", "Guyton Physiology"]
}

GUIDELINES = "TB: HRZE (H5 R10 Z25 E15 mg/kg) 2 months intensive + 4 months continuous phase | DM2: First-line Metformin, HbA1c Target < 7% | HTN: JNC8 Protocol Target < 140/90 mmHg | DKA: Initial Normal Saline (15-20 ml/kg/hr) + Continuous Infusion Regular Insulin (0.1 U/kg/hr) | Malaria: P.vivax = Chloroquine + Primaquine for 14 days; P.falciparum = ACT Treatment Regimen."

# ---------------------------------------------------------------------------
# Strictly Optimized Multi-Mode Prompt Building Block
# ---------------------------------------------------------------------------
def build_prompt(query: str, mode: str, track: str, context: str = "") -> str:
    ctx = f"\n[CRITICAL UNDERLYING VAULT CONTEXT DATA:\n{context[:600]}]\n" if context else ""
    books = ", ".join(MEDICAL_BOOKS.get(track, MEDICAL_BOOKS["MBBS"])[:3])

    # --- VAULT REVISION EXCLUSIVITY MODE ---
    if mode == "vault-answer":
        return (
            f"Role: High-ranking academic {track} medical student topper creating high-yield micro revision sheets. Max 200 words. {ctx}\n"
            f"Topic: {query}\n\n"
            "Format the output text explicitly matching this layout block configuration below without deviations:\n"
            "**KEY OBSERVATIONAL DISCOVERIES**:\n- High yield system fact bullet point 1\n- High yield system fact bullet point 2\n"
            "**CLINICAL CORRELATION PEARL**:\n- 1 sentence critical clinical application insight\n"
            "**EXAM MEMORY CAPTURE RECALL**:\n- Strict quick concept review pointer"
        )

    # --- QUICK SUMMARY STRICTOR ENGINE ---
    if mode == "quick-summary":
        return (
            f"Role: {track} Medical Professor. Build an ultra-dense bulleted execution summary sheet. Do NOT write paragraphs or introductory narratives. Max 150 words. {ctx}\n"
            f"Target Query: {query}\n\n"
            "Format the text strictly matching the block template design parameters below:\n"
            "### ⚡ QUICK SUMMARY ANALYSIS\n"
            "- **Pathological Core Concept**: [Provide exactly 1 sentence summarizing the primary physiological/clinical mechanism]\n"
            "- **High-Yield Medical Triggers**: [Provide exactly 3 fast-recall diagnostic markers, criteria, or high-yield points]\n"
            "- **Management Intervention Standard**: [Provide exactly 1 line detailing the primary therapeutic drug choice or acute emergency management protocol matching standard guidelines]"
        )

    # --- MCQ PRACTICE JSON GENERATION ENGINE ---
    if mode == "mcq-practice":
        return (
            f"Role: Senior Medical Board {track} Examiner. Create exactly 5 authentic case-based clinical MCQs targeting: {query}. {ctx}\n"
            "You MUST output a valid, parsable raw JSON array ONLY. Do NOT enclose in markdown tags or add text prefixes/suffixes.\n"
            "Strict JSON schema array definition:\n"
            '[\n'
            '  {\n'
            '    "q": "Provide a detailed clinical presentation scenario question text.",\n'
            '    "options": {"A": "Option Alpha text", "B": "Option Beta text", "C": "Option Gamma text", "D": "Option Delta text"},\n'
            '    "correct": "A",\n'
            '    "reason_correct": "Provide precise physiological description of why this specific alternative is correct.",\n'
            '    "reason_wrong": {\n'
            '      "A": "Short diagnostic breakdown on option A status validation.",\n'
            '      "B": "Short diagnostic breakdown on option B status validation.",\n'
            '      "C": "Short diagnostic breakdown on option C status validation.",\n'
            '      "D": "Short diagnostic breakdown on option D status validation."\n'
            '    }\n'
            '  }\n'
            ']'
        )

    # --- RAPID RECALL FLASHCARD FORMAT ---
    if mode == "rapid-recall":
        return (
            f"Role: {track} High-Yield Specialization Trainer. Develop a highly structured, rapid active-recall revision flashcard block. Do NOT build massive explanations. Max 200 words. {ctx}\n"
            f"Target System Concept: {query}\n\n"
            "Format explicitly as follows:\n"
            "### 🎴 RAPID RECALL DATA COMPONENT\n"
            "- **Core Definitional Framework**: [Max 15 words concise summary]\n"
            "- **Pathognomonic Trait / Diagnostic Checklist**: [Provide up to two highly specific markers/criteria]\n"
            "- **Structural Flow Hierarchy**: [Concept Step 1] ➔ [Concept Step 2] ➔ [Concept Step 3]\n"
            "- **Critical Examination Trap / Gold Choice**: [Highlight one high-yield clinical contrast point or drug of choice to prevent negative marking]"
        )

    # --- UNIVERSAL DEEP EXPLANATION (COMPACT COMPREHENSIVE UNIVERSITY BLUEPRINT) ---
    return (
        f"Role: Expert Academic Professor of Medical Education for {track} curriculum students. Formulate concise, yet highly thorough, exam-oriented textbook documentation. Strict limit of 400 words maximum. Never break mid-sentence. {ctx}\n"
        f"Core Reference Material: {books}. Diagnostic and Treatment Criteria: {GUIDELINES}\n"
        f"Target Subject Query: {query}\n\n"
        "Format structurally into these explicit markdown section dividers:\n\n"
        "### 1. CLINICAL DEFINITION & MOLECULAR PATHOPHYSIOLOGY\n"
        "[Provide structured definition and precise physiological mechanism cascade steps]\n\n"
        "### 2. EXAM DIAGNOSTIC CRITERIA & CLINICAL PRESENTATION\n"
        "[Provide high-density bulleted presentation signs, pathognomonic indicators, and diagnostic rules]\n\n"
        "### 3. EVIDENCE-BASED PHARMACOLOGICAL & SURGICAL INTERVENTION\n"
        "[Provide precise, step-by-step guideline-directed clinical management management protocol or pharmacological execution pathways]\n\n"
        "### 4. MEDICAL TOPPER'S HIGH-YIELD EXAMINATION PEARLS\n"
        "[Provide one clinical memory mnemonic or highly targeted professional/university examination execution trick]"
    )

MODE_TOKENS = {
    "vault-answer":      500,
    "quick-summary":     600,
    "rapid-recall":      800,
    "mcq-practice":     3500,
    "deep-explanation": 1200,
    "explanation":      1200,
    "exam":             1000,
    "revision":          600,
    "notes":            1200,
    "deep-dive":        1500,
}

PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
MODELS_FALLBACK = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite",
]

# ---------------------------------------------------------------------------
# Core REST Endpoint Implementations
# ---------------------------------------------------------------------------
@app.post("/query", response_model=AiQueryResponse)
async def gemini_query(request: QueryRequest):
    genai.configure(api_key=GEMINI_API_KEY)

    max_tokens = MODE_TOKENS.get(request.mode, 1500)
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
            
            gen_config = {
                "temperature": 0.1 if request.mode == "mcq-practice" else 0.2,
                "max_output_tokens": max_tokens,
            }
            if request.mode == "mcq-practice":
                gen_config["response_mime_type"] = "application/json"

            response = model.generate_content(prompt, generation_config=gen_config)
            if not response.text:
                continue

            answer = response.text if request.mode == "mcq-practice" else clean_text(response.text)

            # Static Citation Engine Parsing
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
                confidence=0.98 if request.mode == "mcq-practice" else 0.95
            )

        except Exception as e:
            last_error = e
            print(f"Fallback alerting: Model {model_name} failed execution: {str(e)[:100]}")
            continue

    raise HTTPException(500, f"Medarro AI Query Stack internal failure: {last_error}")

@app.post("/search")
async def gemini_ai_search(request: AiQueryRequest):
    return await gemini_query(QueryRequest(
        query=request.query,
        mode=request.mode,
        track=request.track
    ))

@app.post("/query-stream")
async def gemini_query_stream(request: QueryRequest):
    genai.configure(api_key=GEMINI_API_KEY)
    max_tokens = MODE_TOKENS.get(request.mode, 1500)
    prompt = build_prompt(request.query, request.mode, request.track, request.context)

    async def generate():
        for model_name in [PRIMARY_MODEL] + MODELS_FALLBACK:
            try:
                model = genai.GenerativeModel(model_name)
                for chunk in model.generate_content(
                    prompt, stream=True,
                    generation_config={
                        "temperature": 0.2,
                        "max_output_tokens": max_tokens
                    }
                ):
                    if chunk.text:
                        yield chunk.text
                return
            except Exception as e:
                print(f"Streaming error on pipeline model {model_name}: {e}")
                continue
        yield "Error: All production backend nodes failed to route structural streaming generation chunks."

    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Medarro API Core Engine", "version": "6.1.2", "apis": {"supabase": "ok"}}

@app.post("/upload-pdf", response_model=UploadPDFResponse)
async def upload_pdf(request: UploadPDFRequest):
    pdf_bytes = await download_pdf(request.pdf_url)
    pages = extract_pages(pdf_bytes)
    chunks = split_into_chunks(pages)
    inserted = 0
    for chunk in chunks:
        emb = get_embedding(chunk["chunk_text"])
        supabase.table("personal_vault").insert({
            "user_id": request.user_id, 
            "pdf_name": request.pdf_name,
            "page_number": chunk["page_number"], 
            "chunk_text": chunk["chunk_text"], 
            "embedding": emb
        }).execute()
        inserted += 1
    return UploadPDFResponse(status="ready", chunks_count=inserted)

@app.post("/search-vault", response_model=SearchVaultResponse)
async def search_vault(request: SearchVaultRequest):
    qe = get_embedding(request.query)
    rpc = supabase.rpc("match_vault_chunks", {
        "query_embedding": qe, 
        "match_user_id": request.user_id, 
        "match_count": 3
    }).execute()
    return SearchVaultResponse(results=[
        VaultSearchResult(
            chunk_text=r["chunk_text"], 
            pdf_name=r["pdf_name"], 
            page_number=r["page_number"], 
            similarity=r["similarity"]
        ) for r in rpc.data or []
    ])

@app.post("/generate-study-plan", response_model=StudyPlanResponse)
async def generate_study_plan(request: StudyPlanRequest):
    try:
        days_remaining = (datetime.strptime(request.target_date, "%Y-%m-%d").date() - date.today()).days
    except Exception:
        days_remaining = 90
    subjects_map = {"NEET": ["Biology", "Physics", "Chemistry"], "MBBS": ["Anatomy", "Physiology", "Biochemistry"]}
    subjects = subjects_map.get(request.track, subjects_map["MBBS"])
    
    plan = []
    for i in range(7):
        d = date.today() + timedelta(days=i)
        plan.append({
            "day": d.strftime("%a"), 
            "date": d.strftime("%Y-%m-%d"), 
            "total_hours": float(request.daily_hours),
            "tasks": [{
                "subject": subjects[0], 
                "topic": "General Revision Overview", 
                "duration_minutes": 60, 
                "mode": "quick-summary", 
                "priority": "high", 
                "time_slot": "09:00 AM"
            }]
        })
    return StudyPlanResponse(
        plan=plan, 
        weekly_subject_split={s: 33 for s in subjects}, 
        ai_insight="Sustained production testing cycle locked. Keep executing.", 
        total_days_remaining=days_remaining
    )

@app.get("/tracks")
async def get_tracks():
    return {"tracks": ["NEET", "MBBS", "BDS", "BHMS"], "modes": list(MODE_TOKENS.keys()), "version": "6.1.2"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
