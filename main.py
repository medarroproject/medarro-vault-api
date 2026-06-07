import os
import json
from typing import List
import re
from datetime import datetime, date, timedelta

import fitz
import httpx
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()

# ---------------------------------------------------------------------------
# RERANKER (optional — graceful fallback if not installed)
# ---------------------------------------------------------------------------
try:
    from fastembed import SparseTextEmbedding  # noqa: F401
    RERANKER_AVAILABLE = True
except Exception:
    RERANKER_AVAILABLE = False

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
app = FastAPI(title="Medarro API", version="7.0.0")

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

    if mode == "vault-answer":
        return MEDARRO_BASE_RULES + (
            f"Role: High-ranking academic {track} medical student topper creating high-yield micro revision sheets. Max 250 words. {ctx}\n"
            f"Topic: {query}\n\n"
            "Format the output text explicitly matching this layout block configuration below without deviations:\n"
            "**KEY OBSERVATIONAL DISCOVERIES**:\n- High yield system fact bullet point 1\n- High yield system fact bullet point 2\n"
            "**CLINICAL CORRELATION PEARL**:\n- 1 sentence critical clinical application insight\n"
            "**EXAM MEMORY CAPTURE RECALL**:\n- Strict quick concept review pointer"
        )

    if mode == "quick-summary":
        return MEDARRO_BASE_RULES + (
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

    if mode == "mcq-practice":
        return MEDARRO_BASE_RULES + (
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

    if mode == "rapid-recall":
        return MEDARRO_BASE_RULES + (
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

    if mode in ["deep-explanation", "explanation", "deep-dive"]:
        return MEDARRO_BASE_RULES + (
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

    return MEDARRO_BASE_RULES + f"Provide a direct high yield medical answer regarding {query} for track {track} matching international clinical references under 300 words."

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

def clean_model_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return "models/gemini-2.5-flash"
    cleaned = name.strip()
    if "yourgeminimodelname" in cleaned.lower() or not cleaned:
        return "models/gemini-2.5-flash"
    if not cleaned.startswith("models/"):
        cleaned = f"models/{cleaned}"
    return cleaned

PRIMARY_MODEL = clean_model_name(os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash"))
MODELS_FALLBACK = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite"
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

    pipeline = [PRIMARY_MODEL] + MODELS_FALLBACK
    sanitized_pipeline = []
    for m in pipeline:
        c = clean_model_name(m)
        if c not in sanitized_pipeline:
            sanitized_pipeline.append(c)

    last_error = None
    for model_name in sanitized_pipeline:
        try:
            model = genai.GenerativeModel(model_name)
            gen_config = {
                "temperature": 0.15 if request.mode == "mcq-practice" else 0.3,
                "max_output_tokens": max_tokens,
            }
            if request.mode == "mcq-practice":
                gen_config["response_mime_type"] = "application/json"

            response = model.generate_content(prompt, generation_config=gen_config)
            if not response or not response.text:
                continue

            if request.mode == "mcq-practice":
                raw = response.text.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
                    raw = re.sub(r"\n?```$", "", raw)
                    raw = raw.strip()
                try:
                    json.loads(raw)
                except json.JSONDecodeError as je:
                    raise HTTPException(500, f"MCQ JSON parse failed: {je} | Raw: {raw[:200]}")
                answer = raw
            else:
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
                confidence=0.98 if request.mode == "mcq-practice" else 0.95
            )

        except HTTPException as he:
            raise he
        except Exception as e:
            last_error = e
            print(f"Tracking error on: {model_name} | Trace: {str(e)[:100]}")
            continue

    raise HTTPException(500, f"Medarro AI Execution Engine failure: {last_error}")

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
    max_tokens = MODE_TOKENS.get(request.mode, 2000)
    prompt = build_prompt(request.query, request.mode, request.track, request.context)

    pipeline = [PRIMARY_MODEL] + MODELS_FALLBACK
    sanitized_pipeline = []
    for m in pipeline:
        c = clean_model_name(m)
        if c not in sanitized_pipeline:
            sanitized_pipeline.append(c)

    async def generate():
        stream_success = False
        for model_name in sanitized_pipeline:
            try:
                model = genai.GenerativeModel(model_name)
                gen_config = {
                    "temperature": 0.3,
                    "max_output_tokens": max_tokens
                }
                if request.mode == "mcq-practice":
                    gen_config["response_mime_type"] = "application/json"

                response_stream = model.generate_content(
                    prompt, stream=True,
                    generation_config=gen_config
                )
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text

                stream_success = True
                break

            except Exception as e:
                print(f"Streaming failed on {model_name}: {e}")
                continue

        if not stream_success:
            yield f"data: {json.dumps({'error': 'Response generation failed.', 'done': True})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "Medarro API",
        "version": "7.0.0",
        "reranker": RERANKER_AVAILABLE,
    }

@app.get("/usage")
async def get_usage(request: Request):
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "").strip()

    if not token:
        return {"queries_used": 0, "queries_limit": 5, "plan": "free"}

    try:
        user_resp = supabase.auth.get_user(token)
        user_id = user_resp.user.id if user_resp.user else None
    except Exception:
        user_id = None

    if not user_id:
        return {"queries_used": 0, "queries_limit": 5, "plan": "free"}

    try:
        profile = supabase.table("user_profiles") \
            .select("plan_type, ai_queries_used, last_query_date") \
            .eq("user_id", user_id) \
            .limit(1) \
            .execute()
        data = (profile.data or [{}])[0]
        plan = data.get("plan_type", "free")
        used = data.get("ai_queries_used", 0) or 0
        limit_map = {"free": 5, "pro": 100, "premium": 999, "beta": 999}
        limit = limit_map.get(plan, 5)
        return {"queries_used": used, "queries_limit": limit, "plan": plan}
    except Exception as e:
        print(f"/usage error: {e}")
        return {"queries_used": 0, "queries_limit": 5, "plan": "free"}

@app.get("/tracks")
async def get_tracks():
    return {
        "tracks": ["NEET", "MBBS", "BDS", "BHMS"],
        "modes": list(MODE_TOKENS.keys()),
        "version": "7.0.0",
    }

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

    subjects_map = {
        "NEET": ["Biology", "Physics", "Chemistry"],
        "MBBS": ["Anatomy", "Physiology", "Biochemistry", "Pharmacology", "Pathology", "Microbiology"],
        "BDS": ["Anatomy", "Physiology", "Biochemistry", "Dental Materials"],
        "BHMS": ["Organon", "Materia Medica", "Anatomy", "Physiology"],
    }
    subjects = subjects_map.get(request.track, subjects_map["MBBS"])
    weak = request.weak_subjects if request.weak_subjects else subjects[:2]

    prompt = (
        f"You are a medical education expert creating a 7-day study plan.\n"
        f"Student Profile:\n"
        f"- Exam: {request.target_exam}\n"
        f"- Track: {request.track}\n"
        f"- Daily study hours: {request.daily_hours}\n"
        f"- Days remaining: {days_remaining}\n"
        f"- Weak subjects: {', '.join(weak)}\n"
        f"- Subjects: {', '.join(subjects)}\n\n"
        f"Create a 7-day study plan. Return ONLY valid JSON, no markdown, no extra text.\n"
        f"JSON format:\n"
        f'{{\n'
        f'  "plan": [\n'
        f'    {{\n'
        f'      "day": "Mon",\n'
        f'      "date": "2026-05-27",\n'
        f'      "total_hours": {request.daily_hours},\n'
        f'      "tasks": [\n'
        f'        {{\n'
        f'          "subject": "Anatomy",\n'
        f'          "topic": "Brachial Plexus",\n'
        f'          "duration_minutes": 60,\n'
        f'          "mode": "deep-explanation",\n'
        f'          "priority": "high",\n'
        f'          "time_slot": "09:00 AM"\n'
        f'        }}\n'
        f'      ]\n'
        f'    }}\n'
        f'  ],\n'
        f'  "weekly_subject_split": {{"Anatomy": 35, "Physiology": 30, "Biochemistry": 35}},\n'
        f'  "ai_insight": "Focus more on weak areas this week.",\n'
        f'  "total_days_remaining": {days_remaining}\n'
        f'}}\n\n'
        f"Rules:\n"
        f"- Generate exactly 7 days starting from today\n"
        f"- Each day must have 3-5 tasks based on daily_hours ({request.daily_hours} hours total)\n"
        f"- Prioritize weak subjects: {', '.join(weak)}\n"
        f"- Use only these modes: deep-explanation, quick-summary, rapid-recall, mcq-practice\n"
        f"- Rotate subjects across the week, don't repeat same subject all 7 days\n"
        f"- weekly_subject_split values must add up to 100\n"
        f"- ai_insight must be personalized, mention weak subjects by name\n"
        f"- Return ONLY the JSON object, nothing else"
    )

    genai.configure(api_key=GEMINI_STUDY_PLAN_KEY)

    for model_name in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]:
        try:
            target_model = f"models/{model_name}" if not model_name.startswith("models/") else model_name
            model = genai.GenerativeModel(target_model)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 3000,
                    "response_mime_type": "application/json"
                }
            )
            if not response.text:
                continue

            raw = response.text.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
                raw = raw.strip()

            parsed = json.loads(raw)

            plan_days = []
            for i, day in enumerate(parsed.get("plan", [])[:7]):
                d = date.today() + timedelta(days=i)
                tasks = []
                for t in day.get("tasks", []):
                    tasks.append(StudyTask(
                        subject=t.get("subject", subjects[0]),
                        topic=t.get("topic", "General Revision"),
                        duration_minutes=int(t.get("duration_minutes", 60)),
                        mode=t.get("mode", "quick-summary"),
                        priority=t.get("priority", "medium"),
                        time_slot=t.get("time_slot", "09:00 AM"),
                    ))
                plan_days.append(StudyPlanDay(
                    day=d.strftime("%a"),
                    date=d.strftime("%Y-%m-%d"),
                    total_hours=float(request.daily_hours),
                    tasks=tasks,
                ))

            weekly_split = parsed.get("weekly_subject_split", {s: round(100 // len(subjects)) for s in subjects})
            ai_insight = parsed.get("ai_insight", f"Focus on {', '.join(weak)} this week.")

            return StudyPlanResponse(
                plan=plan_days,
                weekly_subject_split=weekly_split,
                ai_insight=ai_insight,
                total_days_remaining=days_remaining,
            )

        except json.JSONDecodeError as je:
            print(f"Study plan JSON parse failed on {model_name}: {je}")
            continue
        except Exception as e:
            print(f"Study plan generation failed on {model_name}: {e}")
            continue

    # Fallback
    plan = []
    for i in range(7):
        d = date.today() + timedelta(days=i)
        day_subjects = [subjects[i % len(subjects)], subjects[(i + 1) % len(subjects)]]
        tasks = []
        mins_per_task = (request.daily_hours * 60) // len(day_subjects)
        for j, subj in enumerate(day_subjects):
            tasks.append(StudyTask(
                subject=subj,
                topic=f"{subj} — High Yield Revision",
                duration_minutes=int(mins_per_task),
                mode="quick-summary" if j % 2 == 0 else "rapid-recall",
                priority="high" if subj in weak else "medium",
                time_slot=f"0{9+j}:00 AM",
            ))
        plan.append(StudyPlanDay(
            day=d.strftime("%a"),
            date=d.strftime("%Y-%m-%d"),
            total_hours=float(request.daily_hours),
            tasks=tasks,
        ))

    return StudyPlanResponse(
        plan=plan,
        weekly_subject_split={s: round(100 // len(subjects)) for s in subjects},
        ai_insight=f"Prioritize {', '.join(weak)} this week. Stay consistent!",
        total_days_remaining=days_remaining,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
