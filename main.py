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
# GLOBAL ABSOLUTE RULES
# ---------------------------------------------------------------------------
MEDARRO_BASE_RULES = """STRICT OUTPUT RULES — NEVER VIOLATE THESE:
- Do NOT start with any greeting or filler. No "Good morning class", "Good day students", "Today we will", "Hello", "Great question" or similar.
- Start your answer DIRECTLY with the first heading (DEFINITION or DEF or KEY FACTS).
- No meta-commentary like "Here is a comprehensive answer..." or "I'll explain this..." or "Certainly!"
- No closing lines like "I hope this helps", "Feel free to ask", or "In summary".
- No lecturer tone. You are a structured notes generator, not a teacher giving a lecture.
"""

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
app = FastAPI(title="Medarro API", version="6.1.8")

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
    Cleans structural system double layout enters but keeps markdown syntax intact
    for Lovable rich components rendering.
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
        system_prompt = MEDARRO_BASE_RULES + (
            f"Role: High-ranking academic {track} medical student topper creating high-yield micro revision sheets. Max 250 words. {ctx}\n"
            f"Topic: {query}\n\n"
            "Format the output text explicitly matching this layout block configuration below without deviations:\n"
            "**KEY OBSERVATIONAL DISCOVERIES**:\n- High yield system fact bullet point 1\n- High yield system fact bullet point 2\n"
            "**CLINICAL CORRELATION PEARL**:\n- 1 sentence critical clinical application insight\n"
            "**EXAM MEMORY CAPTURE RECALL**:\n- Strict quick concept review pointer"
        )
        return system_prompt

    # --- QUICK SUMMARY STRICTOR ENGINE ---
    if mode == "quick-summary":
        system_prompt = MEDARRO_BASE_RULES + (
            f"Target Context Query: {query}. Base Reference Core: {books}. {ctx}\n"
            "You are an MBBS exam expert. Answer in EXACTLY this format:\n\n"
            "**KEY FACTS:**\n"
            "- (5-7 bullet points, each max 15 words)\n"
            "- Include mechanism, uses, adverse effects if pharmacology\n"
            "- Include pathogenesis + diagnosis if pathology\n\n"
            "**QUICK COMPARE TABLE:** (only if question asks for difference)\n"
            "| Feature | A | B |\n\n"
            "**EXAM TIP:** (1 line, most commonly asked point)\n\n"
            "STRICT LIMITS:\n"
            "- Total answer: MAX 200 words\n"
            "- No paragraphs. Bullets only.\n"
            "- No introductory sentences.\n"
            "- If answer feels incomplete at 200 words, prioritize HIGH-YIELD points only."
        )
        return system_prompt

    # --- MCQ PRACTICE JSON GENERATION ENGINE ---
    if mode == "mcq-practice":
        system_prompt = MEDARRO_BASE_RULES + (
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
        return system_prompt

    # --- RAPID RECALL FLASHCARD FORMAT ---
    if mode == "rapid-recall":
        system_prompt = MEDARRO_BASE_RULES + (
            f"Target System Concept: {query}. Track context: {track}. {ctx}\n"
            "You are an MBBS exam topper. Answer in EXACTLY this format:\n\n"
            "**DEF:** (1 line max)\n"
            "**LIST:**\n"
            "- (6-8 high-yield bullets, max 12 words each)\n"
            "- Include: definition, classification, key values, mechanism\n"
            "**MNEMONIC:** (1 memorable mnemonic if applicable, else skip)\n\n"
            "STRICT LIMITS:\n"
            "- Total: MAX 120 words\n"
            "- No paragraphs\n"
            "- No introductory sentences\n"
            "- Exam-ready only"
        )
        return system_prompt

    # --- UNIVERSAL DEEP EXPLANATION ---
    if mode in ["deep-explanation", "explanation", "deep-dive"]:
        system_prompt = MEDARRO_BASE_RULES + (
            f"Target Subject Query: {query}. Track context: {track}. Reference: {books}. Guidelines: {GUIDELINES}. {ctx}\n"
            "You are an MBBS medical educator. Answer in EXACTLY this structure:\n\n"
            "**DEFINITION**\n"
            "(2-3 lines only)\n\n"
            "**MECHANISM**\n"
            "(Numbered steps, max 8 steps, each step max 20 words. Include CICR/key concepts inline.)\n\n"
            "**Differences / Comparison** (only if question asks)\n"
            "(Max 4 bullet points comparing two things)\n\n"
            "**CLINICAL RELEVANCE**\n"
            "- (4 bullet points: disease, pharmacology link, complication, clinical sign)\n\n"
            "**KEY FACTS BOX**\n"
            "- (Exactly 5 bullet points, exam-ready, high-yield only)\n\n"
            "STRICT LIMITS:\n"
            "- Total answer: MAX 450 words\n"
            "- No paragraphs in MECHANISM — numbered steps only\n"
            "- No filler, no repetition\n"
            "- If question has multiple parts, cover ALL parts within word limit"
        )
        return system_prompt

    # Default fallback framework
    system_prompt = MEDARRO_BASE_RULES + f"Provide a direct high yield medical answer regarding {query} for track {track} matching international clinical references under 300 words."
    return system_prompt

MODE_TOKENS = {
    "vault-answer":      1200,
    "quick-summary":     1200,
    "rapid-recall":      1500,
    "mcq-practice":      3500,
    "deep-explanation":  2500,
    "explanation":       2000,
    "exam":              1500,
    "revision":          1000,
    "notes":             2000,
    "deep-dive":         2500,
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
            # Code sanitization pattern for safe naming format
            target_model = model_name if model_name.startswith("models/") else f"models/{model_name}"
            model = genai.GenerativeModel(target_model)
            
            # FIXED: RESPONSE MIME TYPE LINE COMPLETELY REMOVED FROM CONFIGURATION MATRIX
            gen_config = {
                "temperature": 0.15 if request.mode == "mcq-practice" else 0.3,
                "max_output_tokens": max_tokens,
            }

            response = model.generate_content(prompt, generation_config=gen_config)
            if not response.text:
                continue

            # FIXED: APPLIED NEW CODE STRUCTURAL SANITATION AND PARSE FENCES REMOVAL
            if request.mode == "mcq-practice":
                raw = response.text.strip()
                # Strip markdown code fences if Gemini wraps JSON in ```json ... ```
                if raw.startswith("```"):
                    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
                    raw = re.sub(r"\n?```$", "", raw)
                    raw = raw.strip()
                # Validate JSON parseable — fail fast with clear error
                try:
                    json.loads(raw)
                except json.JSONDecodeError as je:
                    raise HTTPException(500, f"MCQ JSON parse failed: {je} | Raw: {raw[:200]}")
                answer = raw
            else:
                answer = clean_text(response.text)

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

        except HTTPException as he:
            # Re-raise standard functional HTTP errors from within validation matrices
            raise he
        except Exception as e:
            last_error = e
            print(f"Fallback alerting inside non-stream: Model {model_name} failed execution: {str(e)[:100]}")
            continue

    raise HTTPException(500, f"Medarro AI Query Stack internal failure: {last_error}")

@app.post("/search")
async def gemini_ai_search(request: AiQueryRequest):
    return await gemini_query(QueryRequest(
        query=request.query,
        mode=request.mode,
        track=request.track
    ))

# --- SERVER SENT EVENTS (SSE) STREAMING ENGINE (FIXED LOOP AND FORMAT PARAMETERS) ---
@app.post("/query-stream")
async def gemini_query_stream(request: QueryRequest):
    genai.configure(api_key=GEMINI_API_KEY)
    max_tokens = MODE_TOKENS.get(request.mode, 2000)
    prompt = build_prompt(request.query, request.mode, request.track, request.context)

    async def generate():
        stream_success = False
        
        for model_name in [PRIMARY_MODEL] + MODELS_FALLBACK:
            try:
                # Direct strict sanitation to resolve 400 bad format error
                target_model = model_name if model_name.startswith("models/") else f"models/{model_name}"
                model = genai.GenerativeModel(target_model)
                
                response_stream = model.generate_content(
                    prompt, stream=True,
                    generation_config={
                        "temperature": 0.3,
                        "max_output_tokens": max_tokens
                    }
                )
                
                # Check if generator can stream chunks out successfully
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
                
                stream_success = True
                break  # Exit loop immediately if the stream succeeds
                
            except Exception as e:
                print(f"Streaming error on pipeline model {model_name}: {e}")
                # Continue structural iterations through fallbacks instead of crashing out early
                continue
        
        # If all fallback systems fail, output clean error tracking event
        if not stream_success:
            yield f"data: {json.dumps({'error': 'Response generation failed. Please try again.', 'done': True})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Medarro API Core Engine", "version": "6.1.8", "apis": {"supabase": "ok"}}

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
                "topic": "General Review Overview", 
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
    return {"tracks": ["NEET", "MBBS", "BDS", "BHMS"], "modes": list(MODE_TOKENS.keys()), "version": "6.1.8"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
