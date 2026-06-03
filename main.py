# main.py  —  Medarro API  v7.0.0
import os
import json
import hashlib
import re
from typing import List, Optional
from datetime import datetime, date, timedelta, timezone

import fitz
import httpx
import google.generativeai as genai
from fastapi import FastAPI, HTTPException,  Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from fastembed import TextEmbedding

from models import (
    QueryRequest, AiQueryRequest, AiQueryResponse,
    UploadPDFRequest, UploadPDFResponse,
    SearchVaultRequest, SearchVaultResponse, VaultSearchResult,
    StudyPlanRequest, StudyPlanResponse,
    MarkTaskRequest, DeviceRegisterRequest, UserPlanInfo,
    AnalyticsEventRequest,
)
from middleware import (
    get_current_user, check_rate_limit, check_and_consume_quota,
    register_device, make_cache_key, make_text_hash,
    get_cached_response, set_cached_response,
    get_cached_embedding, set_cached_embedding,
    track_event,
)
from study_plan_service import generate_study_plan

load_dotenv()

# ---------------------------------------------------------------------------
# GLOBAL OUTPUT RULES
# ---------------------------------------------------------------------------
MEDARRO_BASE_RULES = """STRICT OUTPUT RULES — NEVER VIOLATE THESE:
- Do NOT start with any greeting or filler. No "Good morning class", "Good day students", "Today we will", "Hello", "Great question" or similar.
- Start your answer DIRECTLY with the first heading (DEFINITION or DEF or KEY FACTS).
- No meta-commentary like "Here is a comprehensive answer..." or "I'll explain this..." or "Certainly!"
- No closing lines like "I hope this helps", "Feel free to ask", or "In summary".
- No lecturer tone. You are a structured notes generator, not a teacher giving a lecture.
"""

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_STUDY_PLAN_KEY = os.getenv("GEMINI_STUDY_PLAN_KEY", GEMINI_API_KEY)

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is required")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini configured")
except Exception as e:
    raise RuntimeError(f"Gemini configuration error: {e}")

try:
    st_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    print("✅ FastEmbed loaded")
except Exception as e:
    raise RuntimeError(f"FastEmbed initialization error: {e}")

try:
    from fastembed import TextCrossEncoder
    reranker = TextCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    RERANKER_AVAILABLE = True
    print("✅ Reranker loaded")
except Exception:
    reranker = None
    RERANKER_AVAILABLE = False
    print("⚠️  Reranker not available — falling back to similarity sort")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ---------------------------------------------------------------------------
# App
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
# Medical references
# ---------------------------------------------------------------------------
MEDICAL_BOOKS = {
    "NEET":  ["NCERT Biology 11&12", "Trueman Biology", "DC Pandey Physics", "OP Tandon Chemistry"],
    "MBBS":  ["Gray's Anatomy", "Guyton and Hall Physiology", "Robbins Pathology", "BD Chaurasia Anatomy", "Harrison's Internal Medicine", "KD Tripathi Pharmacology"],
    "BDS":   ["Shafer's Oral Pathology", "Gray's Anatomy", "Dental Pharmacology", "Guyton Physiology"],
    "BHMS":  ["Boericke Materia Medica", "Organon of Medicine", "Gray's Anatomy", "Guyton Physiology"],
}

GUIDELINES = (
    "TB: HRZE (H5 R10 Z25 E15 mg/kg) 2 months intensive + 4 months continuous | "
    "DM2: First-line Metformin, HbA1c < 7% | "
    "HTN: JNC8 Target < 140/90 mmHg | "
    "DKA: NS 15-20 ml/kg/hr + Regular Insulin 0.1 U/kg/hr | "
    "Malaria: P.vivax = Chloroquine + Primaquine 14d; P.falciparum = ACT"
)

# ---------------------------------------------------------------------------
# Prompt versions
# ---------------------------------------------------------------------------
PROMPT_VERSIONS = {
    "v1": {"temperature_multiplier": 1.0, "max_token_multiplier": 1.0},
    "v2": {"temperature_multiplier": 0.9, "max_token_multiplier": 1.1},
    "v3": {"temperature_multiplier": 0.85, "max_token_multiplier": 1.2},
}


def build_prompt(query: str, mode: str, track: str, context: str = "", version: str = "v3") -> str:
    ctx = f"\n[VAULT CONTEXT:\n{context[:600]}]\n" if context else ""
    books = ", ".join(MEDICAL_BOOKS.get(track, MEDICAL_BOOKS["MBBS"])[:3])

    if mode == "vault-answer":
        return MEDARRO_BASE_RULES + (
            f"Role: High-ranking academic {track} medical student topper. Max 250 words. {ctx}\n"
            f"Topic: {query}\n\n"
            "**KEY OBSERVATIONAL DISCOVERIES**:\n- High yield fact 1\n- High yield fact 2\n"
            "**CLINICAL CORRELATION PEARL**:\n- 1 sentence critical clinical insight\n"
            "**EXAM MEMORY RECALL**:\n- Quick concept pointer"
        )

    if mode == "quick-summary":
        return MEDARRO_BASE_RULES + (
            f"Query: {query}. Reference: {books}. {ctx}\n"
            "MBBS exam expert. EXACTLY this format:\n\n"
            "**KEY FACTS:**\n- (5-7 bullets, max 15 words each)\n"
            "- Include mechanism/uses/adverse effects if pharmacology\n"
            "- Include pathogenesis/diagnosis if pathology\n\n"
            "**QUICK COMPARE TABLE:** (only if differences asked)\n"
            "| Feature | A | B |\n\n"
            "**EXAM TIP:** (1 line, most commonly tested point)\n\n"
            "LIMITS: MAX 200 words. No paragraphs. No intros."
        )

    if mode == "mcq-practice":
        return MEDARRO_BASE_RULES + (
            f"Senior Medical {track} Examiner. Create 5 clinical case-based MCQs on: {query}. {ctx}\n"
            "Return ONLY valid raw JSON array — no markdown, no preamble.\n"
            "Schema:\n"
            '[\n'
            '  {\n'
            '    "q": "clinical scenario question",\n'
            '    "options": {"A": "text", "B": "text", "C": "text", "D": "text"},\n'
            '    "correct": "A",\n'
            '    "reason_correct": "why correct",\n'
            '    "reason_wrong": {"A": "why", "B": "why", "C": "why", "D": "why"}\n'
            '  }\n'
            ']'
        )

    if mode == "rapid-recall":
        return MEDARRO_BASE_RULES + (
            f"Concept: {query}. Track: {track}. {ctx}\n"
            "MBBS topper. EXACTLY:\n\n"
            "**DEF:** (1 line max)\n"
            "**LIST:**\n- (6-8 bullets, max 12 words each)\n"
            "**MNEMONIC:** (1 mnemonic if applicable)\n\n"
            "LIMITS: MAX 120 words. No paragraphs."
        )

    if mode in ["deep-explanation", "explanation", "deep-dive"]:
        return MEDARRO_BASE_RULES + (
            f"Query: {query}. Track: {track}. Reference: {books}. Guidelines: {GUIDELINES}. {ctx}\n"
            "MBBS educator. EXACTLY this structure:\n\n"
            "**DEFINITION**\n(2-3 lines)\n\n"
            "**MECHANISM**\n(Numbered steps, max 8, each max 20 words)\n\n"
            "**Differences / Comparison** (only if asked)\n(Max 4 bullets)\n\n"
            "**CLINICAL RELEVANCE**\n- (4 bullets: disease, pharmacology link, complication, clinical sign)\n\n"
            "**KEY FACTS BOX**\n- (5 bullets, exam-ready)\n\n"
            "LIMITS: MAX 450 words. No filler."
        )

    return MEDARRO_BASE_RULES + (
        f"Provide a direct high-yield medical answer on {query} for {track}. "
        f"Under 300 words. Structured. No filler."
    )


MODE_TOKENS = {
    "vault-answer":    1200,
    "quick-summary":   1200,
    "rapid-recall":    1500,
    "mcq-practice":    3500,
    "deep-explanation":2500,
    "explanation":     2000,
    "exam":            1500,
    "revision":        1000,
    "notes":           2000,
    "deep-dive":       2500,
}

PRIMARY_MODEL = "models/gemini-2.5-flash"
MODELS_FALLBACK = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite",
]


def _sanitize_pipeline() -> List[str]:
    seen = []
    for m in [PRIMARY_MODEL] + MODELS_FALLBACK:
        m = m.strip()
        if not m.startswith("models/"):
            m = f"models/{m}"
        if m not in seen:
            seen.append(m)
    return seen


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------
CHUNK_WORDS = 500
OVERLAP_WORDS = 100


def extract_pages(pdf_bytes: bytes) -> List[dict]:
    pages = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i in range(len(doc)):
        text = doc[i].get_text("text")
        if text.strip():
            pages.append({"page_number": i + 1, "text": text})
    doc.close()
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
            "page_number": window[0][1],
        })
        if end == len(pairs):
            break
        start += step
    return chunks


async def download_pdf(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as c:
        r = await c.get(url)
        if r.status_code != 200:
            raise HTTPException(400, f"PDF download failed: HTTP {r.status_code}")
        return r.content


def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    return '\n'.join(l.strip() for l in text.split('\n')).strip()


async def get_embedding_cached(text: str) -> List[float]:
    text_hash = make_text_hash(text)
    cached = await get_cached_embedding(text_hash, supabase)
    if cached:
        return cached
    try:
        embedding = list(st_model.embed([text]))[0].tolist()
        await set_cached_embedding(text_hash, embedding, supabase)
        return embedding
    except Exception as e:
        raise HTTPException(503, f"Embedding generation failed: {e}")


# ---------------------------------------------------------------------------
# Auth dependency factory
# ---------------------------------------------------------------------------

async def require_auth(request: Request) -> dict:
    return await get_current_user(request, supabase)


# ---------------------------------------------------------------------------
# Core AI execution
# ---------------------------------------------------------------------------

async def _run_gemini(prompt: str, mode: str, max_tokens: int, version: str = "v3") -> str:
    genai.configure(api_key=GEMINI_API_KEY)
    pv = PROMPT_VERSIONS.get(version, PROMPT_VERSIONS["v3"])
    token_limit = int(max_tokens * pv["max_token_multiplier"])
    temperature = round(0.3 * pv["temperature_multiplier"], 3)

    last_error = None
    for model_name in _sanitize_pipeline():
        try:
            model = genai.GenerativeModel(model_name)
            gen_config = {
                "temperature": 0.15 if mode == "mcq-practice" else temperature,
                "max_output_tokens": token_limit,
            }
            if mode == "mcq-practice":
                gen_config["response_mime_type"] = "application/json"

            response = model.generate_content(prompt, generation_config=gen_config)
            if not response or not response.text:
                continue

            if mode == "mcq-practice":
                raw = response.text.strip()
                raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw).strip()
                try:
                    json.loads(raw)
                except json.JSONDecodeError as je:
                    raise HTTPException(500, f"MCQ JSON parse failed: {je}")
                return raw
            else:
                return clean_text(response.text)

        except HTTPException:
            raise
        except Exception as e:
            last_error = e
            print(f"Model {model_name} failed: {str(e)[:100]}")
            continue

    raise HTTPException(500, f"All Gemini models failed: {last_error}")


def _extract_sources(answer: str) -> List[str]:
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
    sources = list(dict.fromkeys([v for k, v in kws.items() if k in al]))
    return sources or ["Standard Medical References"]


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

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
            .maybeSingle() \
            .execute()

        data = profile.data or {}
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


# ---------------------
# Auth: device registration
# ---------------------

@app.post("/auth/register-device")
async def auth_register_device(
    body: DeviceRegisterRequest,
    user: dict = Depends(require_auth),
):
    await register_device(
        user_id=user["id"],
        device_fingerprint=body.device_fingerprint,
        device_name=body.device_name,
        supabase=supabase,
    )
    return {"status": "registered", "device_fingerprint": body.device_fingerprint}


# ---------------------
# User plan info
# ---------------------

@app.get("/me/plan")
async def get_my_plan(user: dict = Depends(require_auth)) -> UserPlanInfo:
    try:
        profile_resp = supabase.table("user_profiles").select("*").eq("id", user["id"]).single().execute()
        profile = profile_resp.data
    except Exception:
        raise HTTPException(404, "User profile not found")

    plan = profile["plan"]
    now_utc = datetime.now(timezone.utc)
    today = date.today().isoformat()

    # Resolve effective plan
    if plan == "trial":
        trial_ends = datetime.fromisoformat(profile["trial_ends_at"].replace("Z", "+00:00"))
        if now_utc > trial_ends:
            plan = "free"
    if plan == "premium":
        premium_ends = profile.get("premium_ends_at")
        if premium_ends:
            pe = datetime.fromisoformat(premium_ends.replace("Z", "+00:00"))
            if now_utc > pe:
                plan = "free"

    try:
        usage_resp = supabase.table("daily_usage").select("queries_used").eq("user_id", user["id"]).eq("usage_date", today).single().execute()
        used_today = usage_resp.data["queries_used"]
    except Exception:
        used_today = 0

    limits = {"trial": 30, "free": 5, "premium": 70}
    limit = limits.get(plan, 5)

    return UserPlanInfo(
        user_id=user["id"],
        plan=plan,
        queries_used_today=used_today,
        queries_limit_today=limit,
        trial_ends_at=profile.get("trial_ends_at"),
        trial_queries_used=profile.get("trial_queries_used"),
        trial_queries_limit=30,
        is_trial_active=(profile["plan"] == "trial" and plan == "trial"),
        premium_ends_at=profile.get("premium_ends_at"),
    )


# ---------------------
# /query  (protected)
# ---------------------

@app.post("/query", response_model=AiQueryResponse)
async def gemini_query(
    request: QueryRequest,
    user: dict = Depends(require_auth),
):
    check_rate_limit(user["id"])
    await check_and_consume_quota(user["id"], supabase)

    cache_key = make_cache_key(request.query, request.mode, request.track)
    if request.mode != "mcq-practice":
        cached = await get_cached_response(cache_key, supabase)
        if cached:
            await track_event(user["id"], "query_cache_hit", {"mode": request.mode, "track": request.track}, supabase)
            return AiQueryResponse(
                answer=cached,
                sources=_extract_sources(cached),
                confidence=0.95,
                cached=True,
            )

    prompt = build_prompt(request.query, request.mode, request.track, request.context, request.prompt_version)
    max_tokens = MODE_TOKENS.get(request.mode, 2000)
    answer = await _run_gemini(prompt, request.mode, max_tokens, request.prompt_version)

    if request.mode != "mcq-practice":
        await set_cached_response(cache_key, request.query, request.mode, request.track, answer, supabase)

    await track_event(user["id"], "query", {"mode": request.mode, "track": request.track, "query_len": len(request.query)}, supabase)

    return AiQueryResponse(
        answer=answer,
        sources=_extract_sources(answer),
        confidence=0.98 if request.mode == "mcq-practice" else 0.95,
        cached=False,
    )


# ---------------------
# /query-stream  (protected)
# ---------------------

@app.post("/query-stream")
async def gemini_query_stream(
    request: QueryRequest,
    user: dict = Depends(require_auth),
):
    check_rate_limit(user["id"])
    await check_and_consume_quota(user["id"], supabase)

    genai.configure(api_key=GEMINI_API_KEY)
    prompt = build_prompt(request.query, request.mode, request.track, request.context, request.prompt_version)
    max_tokens = MODE_TOKENS.get(request.mode, 2000)

    async def generate():
        for model_name in _sanitize_pipeline():
            try:
                model = genai.GenerativeModel(model_name)
                gen_config = {"temperature": 0.3, "max_output_tokens": max_tokens}
                if request.mode == "mcq-practice":
                    gen_config["response_mime_type"] = "application/json"

                response_stream = model.generate_content(prompt, stream=True, generation_config=gen_config)
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
                return
            except Exception as e:
                print(f"Stream failed on {model_name}: {e}")
                continue

        yield json.dumps({"error": "Stream generation failed.", "done": True})

    await track_event(user["id"], "query_stream", {"mode": request.mode, "track": request.track}, supabase)
    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------
# /search  (public — backward compat)
# ---------------------

@app.post("/search")
async def gemini_ai_search(request: AiQueryRequest):
    """Public search endpoint (no auth) — for backward compatibility. No quota enforcement."""
    cache_key = make_cache_key(request.query, request.mode, request.track)
    if request.mode != "mcq-practice":
        cached = await get_cached_response(cache_key, supabase)
        if cached:
            return AiQueryResponse(answer=cached, sources=_extract_sources(cached), confidence=0.95, cached=True)

    prompt = build_prompt(request.query, request.mode, request.track)
    max_tokens = MODE_TOKENS.get(request.mode, 2000)
    answer = await _run_gemini(prompt, request.mode, max_tokens)

    if request.mode != "mcq-practice":
        await set_cached_response(cache_key, request.query, request.mode, request.track, answer, supabase)

    return AiQueryResponse(answer=answer, sources=_extract_sources(answer), confidence=0.95, cached=False)


# ---------------------
# /upload-pdf  (protected)
# ---------------------

@app.post("/upload-pdf", response_model=UploadPDFResponse)
async def upload_pdf(
    request: UploadPDFRequest,
    user: dict = Depends(require_auth),
):
    user_id = user["id"]

    # Plan-based PDF limits
    try:
        profile_resp = supabase.table("user_profiles").select("plan").eq("id", user_id).single().execute()
        plan = profile_resp.data.get("plan", "free")
    except Exception:
        plan = "free"

    pdf_limit = 50 if plan == "premium" else 1
    size_limit_mb = 50 if plan == "premium" else 10

    # Count existing PDFs
    existing_resp = supabase.table("personal_vault").select("pdf_name", count="exact").eq("user_id", user_id).execute()
    existing_pdf_names = set()
    if existing_resp.data:
        existing_pdf_names = {r["pdf_name"] for r in existing_resp.data}

    if len(existing_pdf_names) >= pdf_limit:
        raise HTTPException(403, f"PDF limit reached ({pdf_limit} PDFs). Upgrade to Premium.")

    # Download
    pdf_bytes = await download_pdf(request.pdf_url)

    # Size check
    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > size_limit_mb:
        raise HTTPException(400, f"PDF too large ({size_mb:.1f} MB). Limit: {size_limit_mb} MB.")

    # Deduplication
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    try:
        dup = supabase.table("personal_vault").select("id").eq("user_id", user_id).eq("pdf_hash", pdf_hash).limit(1).execute()
        if dup.data:
            return UploadPDFResponse(status="already_exists", chunks_count=0, pdf_hash=pdf_hash, skipped=True)
    except Exception:
        pass

    # Extract and chunk
    pages = extract_pages(pdf_bytes)
    chunks = split_into_chunks(pages)

    if not chunks:
        raise HTTPException(400, "PDF has no extractable text.")

    # Bulk embed + insert
    rows = []
    for chunk in chunks:
        emb = await get_embedding_cached(chunk["chunk_text"])
        rows.append({
            "user_id": user_id,
            "pdf_name": request.pdf_name,
            "page_number": chunk["page_number"],
            "chunk_text": chunk["chunk_text"],
            "embedding": emb,
            "pdf_hash": pdf_hash,
            "file_size_bytes": len(pdf_bytes),
        })

    # Bulk insert in batches of 50
    batch_size = 50
    for i in range(0, len(rows), batch_size):
        supabase.table("personal_vault").insert(rows[i:i + batch_size]).execute()

    await track_event(user_id, "pdf_upload", {"pdf_name": request.pdf_name, "chunks": len(rows)}, supabase)

    return UploadPDFResponse(status="ready", chunks_count=len(rows), pdf_hash=pdf_hash, skipped=False)


# ---------------------
# /search-vault  (protected)
# ---------------------

@app.post("/search-vault", response_model=SearchVaultResponse)
async def search_vault(
    request: SearchVaultRequest,
    user: dict = Depends(require_auth),
):
    check_rate_limit(user["id"])
    await check_and_consume_quota(user["id"], supabase)

    query_embedding = await get_embedding_cached(request.query)

    # Fetch top 10
    rpc = supabase.rpc("match_vault_chunks", {
        "query_embedding": query_embedding,
        "match_user_id": user["id"],
        "match_count": 10,
    }).execute()

    raw_results = rpc.data or []

    # Rerank if available
    if RERANKER_AVAILABLE and reranker and len(raw_results) > 3:
        try:
            pairs = [(request.query, r["chunk_text"]) for r in raw_results]
            scores = list(reranker.rerank(pairs))
            for i, r in enumerate(raw_results):
                r["rerank_score"] = float(scores[i])
            raw_results.sort(key=lambda x: x.get("rerank_score", x["similarity"]), reverse=True)
        except Exception as e:
            print(f"Reranker failed (non-fatal): {e}")
            raw_results.sort(key=lambda x: x["similarity"], reverse=True)
    else:
        raw_results.sort(key=lambda x: x["similarity"], reverse=True)

    top3 = raw_results[:3]

    await track_event(user["id"], "vault_search", {"query": request.query[:60]}, supabase)

    return SearchVaultResponse(results=[
        VaultSearchResult(
            chunk_text=r["chunk_text"],
            pdf_name=r["pdf_name"],
            page_number=r["page_number"],
            similarity=round(r["similarity"], 4),
        ) for r in top3
    ])


# ---------------------
# /generate-study-plan  (protected)
# ---------------------

@app.post("/generate-study-plan", response_model=StudyPlanResponse)
async def create_study_plan(
    request: StudyPlanRequest,
    user: dict = Depends(require_auth),
):
    plan = generate_study_plan(
        user_id=user["id"],
        request=request,
        gemini_key=GEMINI_STUDY_PLAN_KEY,
    )

    # Persist plan to Supabase
    try:
        plan_row = {
            "id": plan.plan_id,
            "user_id": user["id"],
            "target_exam": request.target_exam,
            "target_date": request.target_date,
            "track": request.track,
            "daily_hours": request.daily_hours,
            "study_days_per_week": request.study_days_per_week,
            "weak_subjects": request.weak_subjects,
            "strong_subjects": request.strong_subjects,
            "completed_topics": request.completed_topics,
            "pending_topics": request.pending_topics,
            "last_mock_score": request.last_mock_score,
            "target_score": request.target_score,
            "current_year": request.current_year,
            "days_remaining": plan.total_days_remaining,
            "syllabus_completion_percent": plan.syllabus_completion_percent,
            "weakness_score": plan.weakness_score,
            "ai_insight": plan.ai_insight,
            "motivation": plan.motivation,
            "is_active": True,
        }
        supabase.table("study_plans").upsert(plan_row).execute()

        # Persist tasks
        task_rows = []
        for day in plan.plan:
            for task in day.tasks:
                task_rows.append({
                    "plan_id": plan.plan_id,
                    "user_id": user["id"],
                    "scheduled_date": day.date,
                    "day_label": day.day,
                    "subject": task.subject,
                    "topic": task.topic,
                    "duration_minutes": task.duration_minutes,
                    "mode": task.mode,
                    "priority": task.priority,
                    "time_slot": task.time_slot,
                    "task_type": task.task_type,
                })
        if task_rows:
            supabase.table("study_tasks").insert(task_rows).execute()

        # Persist revision schedule
        rev_rows = []
        for rev in plan.revision_schedule:
            rev_rows.append({
                "user_id": user["id"],
                "plan_id": plan.plan_id,
                "topic": rev["topic"],
                "subject": rev["subject"],
                "first_learned_date": rev["first_learned_date"],
                "next_revision_date": rev["next_revision_date"],
                "revision_stage": rev["revision_stage"],
            })
        if rev_rows:
            # Batch insert
            for i in range(0, len(rev_rows), 50):
                supabase.table("revision_schedule").insert(rev_rows[i:i+50]).execute()

    except Exception as e:
        print(f"Study plan persistence failed (non-fatal): {e}")

    await track_event(user["id"], "study_plan_created", {"track": request.track, "exam": request.target_exam}, supabase)
    return plan


# ---------------------
# /study-plan/tasks/complete  (protected)
# ---------------------

@app.post("/study-plan/tasks/complete")
async def mark_task_complete(
    body: MarkTaskRequest,
    user: dict = Depends(require_auth),
):
    now = datetime.now(timezone.utc).isoformat()
    result = supabase.table("study_tasks").update({
        "is_completed": body.completed,
        "completed_at": now if body.completed else None,
    }).eq("id", body.task_id).eq("user_id", user["id"]).execute()

    if not result.data:
        raise HTTPException(404, "Task not found or not owned by user")

    return {"status": "updated", "task_id": body.task_id, "completed": body.completed}


# ---------------------
# /study-plan/backlog/reschedule  (protected)
# ---------------------

@app.post("/study-plan/backlog/reschedule")
async def reschedule_backlog(
    body: dict,
    user: dict = Depends(require_auth),
):
    plan_id = body.get("plan_id")
    if not plan_id:
        raise HTTPException(400, "plan_id required")

    today = date.today().isoformat()

    # Fetch all missed (not completed, date < today)
    missed_resp = supabase.table("study_tasks").select("*").eq("plan_id", plan_id).eq("user_id", user["id"]).eq("is_completed", False).lt("scheduled_date", today).execute()
    missed = missed_resp.data or []

    if not missed:
        return {"status": "no_backlog", "rescheduled_count": 0}

    rescheduled = 0
    for i, task in enumerate(missed):
        new_date = (date.today() + timedelta(days=i + 1)).isoformat()
        supabase.table("study_tasks").update({
            "scheduled_date": new_date,
            "is_missed": True,
            "rescheduled_from": task["scheduled_date"],
        }).eq("id", task["id"]).execute()
        rescheduled += 1

    return {"status": "rescheduled", "rescheduled_count": rescheduled}


# ---------------------
# /analytics/event  (protected)
# ---------------------

@app.post("/analytics/event")
async def log_analytics_event(
    body: AnalyticsEventRequest,
    user: dict = Depends(require_auth),
):
    await track_event(user["id"], body.event_type, body.event_data, supabase)
    return {"status": "logged"}


# ---------------------
# /referral/info  (protected)
# ---------------------

@app.get("/referral/info")
async def get_referral_info(user: dict = Depends(require_auth)):
    try:
        profile_resp = supabase.table("user_profiles").select("referral_code, referred_by").eq("id", user["id"]).single().execute()
        return profile_resp.data
    except Exception:
        raise HTTPException(404, "Profile not found")


# ---------------------
# Entry point
# ---------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
