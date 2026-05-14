import os
import io
import asyncio
import json
from typing import List
import hashlib
import re
from datetime import datetime, date

import fitz  # PyMuPDF
import httpx
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY environment variable is required")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API configured successfully")
except Exception as e:
    print(f"❌ Gemini configuration error: {e}")
    raise

try:
    st_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    print("✅ FastEmbed model loaded (384 dimensions)")
except Exception as e:
    print(f"❌ FastEmbed load error: {e}")
    raise

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY"),
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Medarro API",
    description="AI-powered medical education backend for Medarro platform.",
    version="4.0.0",
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
# Pydantic models
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


# Study Plan Models
class StudyPlanRequest(BaseModel):
    user_id: str
    target_exam: str = "NEET UG"
    target_date: str  # "2025-08-03"
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
# Helper functions for PDF Processing
# ---------------------------------------------------------------------------

CHUNK_WORDS = 500
OVERLAP_WORDS = 100


def extract_pages(pdf_bytes: bytes) -> List[dict]:
    pages = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_index in range(len(doc)):
            page = doc[page_index]
            text = page.get_text("text")
            if text.strip():
                pages.append({
                    "page_number": page_index + 1,
                    "text": text
                })
        doc.close()
    except Exception as e:
        print(f"❌ PDF extraction error: {e}")
        raise
    return pages


def split_into_chunks(pages: List[dict]) -> List[dict]:
    word_page_pairs = []
    for page in pages:
        words = page["text"].split()
        for word in words:
            word_page_pairs.append((word, page["page_number"]))

    if not word_page_pairs:
        return []

    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    start = 0

    while start < len(word_page_pairs):
        end = min(start + CHUNK_WORDS, len(word_page_pairs))
        window = word_page_pairs[start:end]
        chunk_text = " ".join(w for w, _ in window)
        page_number = window[0][1]
        chunks.append({
            "chunk_text": chunk_text,
            "page_number": page_number
        })
        if end == len(word_page_pairs):
            break
        start += step

    return chunks


def get_embedding(text: str) -> List[float]:
    try:
        embeddings = list(st_model.embed([text]))
        return embeddings[0].tolist()
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Embedding service error: {str(e)}"
        )


async def download_pdf(url: str) -> bytes:
    async with httpx.AsyncClient(
        timeout=60.0,
        follow_redirects=True
    ) as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download PDF. HTTP {response.status_code}",
            )
        return response.content


# ---------------------------------------------------------------------------
# MARKDOWN CLEANING
# ---------------------------------------------------------------------------

def clean_markdown(text: str) -> str:
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'\+\+(.+?)\+\+', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    return text.strip()


# ---------------------------------------------------------------------------
# MEDICAL BOOK REFERENCES
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
        "Netter's Anatomy",
        "Harrison's Principles of Internal Medicine"
    ],
    "BD Chaurasia": [
        "BD Chaurasia's Human Anatomy",
        "BD Chaurasia's Clinical Anatomy"
    ],
    "PHARMACOLOGY": [
        "KD Tripathi Essentials of Medical Pharmacology",
        "Goodman & Gilman's Pharmacological Basis"
    ],
    "SURGERY": [
        "SRB Manual of Surgery",
        "Bailey & Love Short Practice of Surgery",
        "Schwartz's Principles of Surgery"
    ],
    "PATHOLOGY": [
        "Robbins & Kumar Pathologic Basis of Disease",
        "Underwood's Pathology"
    ]
}

# ---------------------------------------------------------------------------
# UPDATED MODE INSTRUCTIONS — Lovable 4 modes + legacy modes
# ---------------------------------------------------------------------------

MODE_INSTRUCTIONS = {

    # ── Lovable Frontend Modes ──────────────────────────────────────────────

    "deep-explanation": """Explain the concept thoroughly
    in student-friendly language.
    - Start with a clear definition
    - Include mechanism / pathophysiology
    - Give clinical relevance
    - Use simple analogies where helpful
    - Include all important details
    - Mention landmark studies if relevant
    - Format: flowing paragraphs""",

    "quick-summary": """Provide a concise revision-ready summary.
    - 5-7 key bullet points only
    - No unnecessary details
    - Highlight key differentiators
    - Memory-friendly format
    - Include 1 mnemonic if applicable
    - Suitable for quick revision""",

    "mcq-practice": """Generate exactly 5 high-quality MCQs
    on this topic at NEET/MBBS exam level.

    Use this EXACT format for each question:

    Q1. [Question text]
    A. [Option A]
    B. [Option B]
    C. [Option C]
    D. [Option D]
    Answer: [Correct letter e.g. B]
    Explanation: [Clear explanation of why this answer is correct]

    Q2. [Question text]
    A. [Option A]
    B. [Option B]
    C. [Option C]
    D. [Option D]
    Answer: [Correct letter]
    Explanation: [Explanation]

    Continue for all 5 questions.
    Make questions exam-relevant and clinically important.""",

    "rapid-recall": """Rapid recall format for last-minute exam prep.
    Include ALL of these sections:

    DEFINITION: [1 crisp line]

    KEY POINTS:
    - [Point 1]
    - [Point 2]
    - [Point 3 - max 5 points]

    MNEMONIC: [Easy to remember mnemonic]

    FLOWCHART:
    [Step/Component 1] → [Step/Component 2] → [Step/Component 3]

    EXAM TIP: [Exactly how this topic is asked in NEET/MBBS exams]

    HIGH YIELD FACT: [Single most important thing to remember]""",

    # ── Legacy Modes (kept for backward compatibility) ──────────────────────

    "explanation": """Explain the concept thoroughly
    but in student-friendly language.
    - Start with definition
    - Include mechanism/pathophysiology
    - Give clinical relevance
    - Use simple analogies where helpful""",

    "exam": """Provide concise exam-ready answer.
    - Short, accurate points
    - No unnecessary details
    - Highlight key differentiators
    - Suitable for MCQs and short answer""",

    "revision": """Format as quick revision notes.
    - Bullet points only
    - Key facts highlighted
    - Mnemonics if applicable
    - Memory-friendly format""",

    "notes": """Provide detailed study notes.
    - Numbered points
    - Include subtopics
    - Clinical correlations
    - Exam-relevant details""",

    "deep-dive": """Comprehensive, research-level answer.
    - Detailed mechanisms
    - Recent advances
    - Controversial aspects
    - References to landmark studies"""
}


def build_medical_prompt(
    query: str,
    mode: str = "deep-explanation",
    track: str = "NEET"
) -> str:

    mode_instruction = MODE_INSTRUCTIONS.get(
        mode,
        MODE_INSTRUCTIONS["deep-explanation"]
    )

    if track == "NEET":
        relevant_books = MEDICAL_BOOKS["NEET"]
    elif track == "MBBS":
        relevant_books = (
            MEDICAL_BOOKS["MBBS"]
            + [MEDICAL_BOOKS["BD Chaurasia"][0]]
        )
    elif track == "BHMS":
        relevant_books = [
            "Boericke's Materia Medica",
            "Organon of Medicine"
        ]
    elif track == "BDS":
        relevant_books = [
            "Shafer's Textbook of Oral Pathology",
            "Dental Pharmacology"
        ]
    else:
        relevant_books = MEDICAL_BOOKS["MBBS"]

    books_text = "\n".join([f"- {book}" for book in relevant_books])

    prompt = f"""You are an expert medical educator \
for Indian {track} students.

STRICT GUIDELINES:
1. ONLY answer based on standard medical textbooks
   and established medical knowledge
2. Reference specific textbooks when relevant
3. Use EXACT medical terminology — no approximations
4. If uncertain, clearly state:
   "This topic is not covered in standard {track} curriculum"
5. Prioritize ICMR, WHO, and Indian medical standards
6. For drugs: include generic name, class,
   mechanism, important side effects
7. For anatomy: describe structures, relations,
   nerve/blood supply accurately
8. For physiology: explain mechanisms at
   cellular/system level with accuracy
9. For pathology: describe pathogenesis, pathology,
   clinical features accurately

STUDENT LEVEL: {track} track medical student
ANSWER MODE: {mode}

ANSWER FORMAT:
{mode_instruction}

RELEVANT TEXTBOOKS FOR THIS QUERY:
{books_text}

STUDENT QUESTION: {query}

RULES FOR YOUR RESPONSE:
- Be factually accurate — this is for medical education
- Include source references naturally in the answer
- Highlight key points the student MUST remember
- Include clinical correlations where relevant
- For NEET: use curriculum-level language
- For MBBS: can be more detailed and comprehensive
- If there is any ambiguity, clarify it
- DO NOT use Markdown formatting
  (no ##, **, __, bullet dashes, etc.)
- Return plain text only

Now provide the accurate medical answer:"""

    return prompt


# ---------------------------------------------------------------------------
# Endpoints — Health Check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    health_status = {
        "status": "ok",
        "service": "Medarro API",
        "version": "4.0.0",
        "apis": {
            "gemini_query": "ok",
            "embeddings": "fastembed_local",
            "supabase": "ok"
        },
        "embedding_info": {
            "model": "BAAI/bge-small-en-v1.5",
            "dimensions": 384,
            "type": "local_free",
            "size": "~50MB",
            "cost": "zero"
        },
        "modes_available": list(MODE_INSTRUCTIONS.keys()),
        "warnings": []
    }

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(
            contents=[{
                "role": "user",
                "parts": [{"text": "Say OK"}]
            }]
        )
        response_text = None
        try:
            response_text = response.text
        except:
            pass
        if not response_text and hasattr(response, "candidates"):
            try:
                response_text = (
                    response.candidates[0]
                    .content.parts[0].text
                )
            except:
                pass
        if response_text and len(response_text.strip()) > 0:
            health_status["apis"]["gemini_query"] = "ok"
        else:
            health_status["apis"]["gemini_query"] = "degraded"
            health_status["warnings"].append(
                "Gemini returned empty response"
            )
    except Exception as e:
        health_status["apis"]["gemini_query"] = "down"
        health_status["warnings"].append(
            f"Gemini error: {str(e)[:80]}"
        )

    try:
        test_emb = get_embedding("test")
        if len(test_emb) == 384:
            health_status["apis"]["embeddings"] = (
                "ok (fastembed, 384 dims)"
            )
        else:
            health_status["warnings"].append(
                f"Unexpected dims: {len(test_emb)}"
            )
    except Exception as e:
        health_status["apis"]["embeddings"] = "down"
        health_status["warnings"].append(
            f"Embedding error: {str(e)[:80]}"
        )

    return health_status


# ---------------------------------------------------------------------------
# Endpoints — PDF Upload & Vault Search
# ---------------------------------------------------------------------------

@app.post("/upload-pdf", response_model=UploadPDFResponse)
async def upload_pdf(request: UploadPDFRequest):
    try:
        pdf_bytes = await download_pdf(request.pdf_url)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading PDF: {str(e)}"
        )

    try:
        pages = extract_pages(pdf_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Error extracting PDF text: {str(e)}"
        )

    if not pages:
        raise HTTPException(
            status_code=422,
            detail="PDF contains no extractable text"
        )

    chunks = split_into_chunks(pages)
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No text chunks could be generated"
        )

    inserted_count = 0
    try:
        for chunk in chunks:
            embedding = get_embedding(chunk["chunk_text"])
            row = {
                "user_id": request.user_id,
                "pdf_name": request.pdf_name,
                "page_number": chunk["page_number"],
                "chunk_text": chunk["chunk_text"],
                "embedding": embedding,
            }
            result = (
                supabase.table("personal_vault")
                .insert(row)
                .execute()
            )
            if not result.data:
                raise HTTPException(
                    status_code=500,
                    detail="Supabase insert failed. Check RLS policies.",
                )
            inserted_count += 1
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during storage: {str(e)}"
        )

    return UploadPDFResponse(
        status="ready",
        chunks_count=inserted_count
    )


@app.post("/search-vault", response_model=SearchVaultResponse)
async def search_vault(request: SearchVaultRequest):
    query_embedding = None
    embedding_failed = False

    try:
        query_embedding = get_embedding(request.query)
    except Exception as e:
        print(f"⚠️ Vector embedding failed: {str(e)}")
        embedding_failed = True

    if query_embedding and not embedding_failed:
        try:
            rpc_response = supabase.rpc(
                "match_vault_chunks",
                {
                    "query_embedding": query_embedding,
                    "match_user_id": request.user_id,
                    "match_count": 5,
                },
            ).execute()

            if rpc_response.data:
                results = [
                    VaultSearchResult(
                        chunk_text=row["chunk_text"],
                        pdf_name=row["pdf_name"],
                        page_number=row["page_number"],
                        similarity=row["similarity"],
                    )
                    for row in rpc_response.data
                ]
                return SearchVaultResponse(results=results)
        except Exception as e:
            print(f"⚠️ Vector search RPC failed: {str(e)}")
            embedding_failed = True

    print("📌 Falling back to full-text search")
    try:
        response = (
            supabase.table("personal_vault")
            .select("chunk_text, pdf_name, page_number")
            .eq("user_id", request.user_id)
            .full_text_search("chunk_text", request.query)
            .limit(5)
            .execute()
        )

        if response.data:
            results = [
                VaultSearchResult(
                    chunk_text=row["chunk_text"],
                    pdf_name=row["pdf_name"],
                    page_number=row["page_number"],
                    similarity=0.85,
                )
                for row in response.data
            ]
            return SearchVaultResponse(results=results)
        else:
            return SearchVaultResponse(results=[])

    except Exception as e:
        print(f"❌ Full-text search failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Search unavailable: {str(e)[:100]}"
        )


# ---------------------------------------------------------------------------
# Endpoints — Gemini Medical Q&A
# ---------------------------------------------------------------------------

@app.post("/query", response_model=AiQueryResponse)
async def gemini_query(request: QueryRequest):
    prompt = build_medical_prompt(
        query=request.query,
        mode=request.mode,
        track=request.track
    )

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_output_tokens": 2000,
            }
        )

        if not response.text:
            raise HTTPException(
                status_code=500,
                detail="Gemini returned empty response"
            )

        clean_answer = clean_markdown(response.text)

        sources = []
        book_keywords = {
            "Gray": "Gray's Anatomy",
            "Guyton": "Guyton & Hall Physiology",
            "Robbins": "Robbins Pathology",
            "Chaurasia": "BD Chaurasia Anatomy",
            "Tripathi": "KD Tripathi Pharmacology",
            "SRB": "SRB Manual of Surgery",
            "Harrison": "Harrison's Principles",
            "NCERT": "NCERT Biology",
            "Netter": "Netter's Anatomy",
        }

        answer_lower = clean_answer.lower()
        for keyword, full_name in book_keywords.items():
            if keyword.lower() in answer_lower:
                sources.append(full_name)

        if not sources:
            sources = ["Standard Medical References"]

        sources = list(dict.fromkeys(sources))

        return AiQueryResponse(
            answer=clean_answer,
            sources=sources,
            confidence=0.95
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calling Gemini API: {str(e)}"
        )


@app.post("/search")
async def gemini_ai_search(request: AiQueryRequest):
    """Backward compatibility endpoint."""
    query_req = QueryRequest(
        query=request.query,
        mode=request.mode,
        track=request.track
    )
    return await gemini_query(query_req)


@app.post("/query-stream")
async def gemini_query_stream(request: QueryRequest):
    from fastapi.responses import StreamingResponse

    prompt = build_medical_prompt(
        query=request.query,
        mode=request.mode,
        track=request.track
    )

    async def generate():
        try:
            model = genai.GenerativeModel(
                "models/gemini-2.5-flash"
            )
            response = model.generate_content(
                prompt,
                stream=True,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_output_tokens": 2000,
                }
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )


# ---------------------------------------------------------------------------
# Endpoints — Study Plan
# ---------------------------------------------------------------------------

@app.post(
    "/generate-study-plan",
    response_model=StudyPlanResponse
)
async def generate_study_plan(request: StudyPlanRequest):
    """Generate AI personalized 7-day study plan."""

    try:
        target = datetime.strptime(
            request.target_date, "%Y-%m-%d"
        ).date()
        today = date.today()
        days_remaining = (target - today).days
    except Exception:
        days_remaining = 90

    all_subjects = {
        "NEET": ["Biology", "Physics", "Chemistry"],
        "MBBS": [
            "Anatomy", "Physiology",
            "Biochemistry", "Pharmacology",
            "Pathology", "Microbiology"
        ],
        "BDS": [
            "Anatomy", "Physiology",
            "Biochemistry", "Dental Materials",
            "Oral Pathology"
        ],
        "BHMS": [
            "Organon of Medicine",
            "Materia Medica",
            "Anatomy", "Physiology"
        ]
    }

    subjects = all_subjects.get(
        request.track,
        all_subjects["MBBS"]
    )

    weak_str = (
        ", ".join(request.weak_subjects)
        if request.weak_subjects
        else "Not specified"
    )

    # Generate 7 dates from today
    from datetime import timedelta
    days_list = []
    day_names = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]
    for i in range(7):
        d = today + timedelta(days=i)
        days_list.append({
            "day": day_names[d.weekday()],
            "date": d.strftime("%Y-%m-%d")
        })

    prompt = f"""You are an expert medical education 
planner for Indian students.

Create a personalized 7-day study plan.

STUDENT INFO:
- Target Exam: {request.target_exam}
- Track: {request.track}
- Days Remaining: {days_remaining}
- Daily Study Hours: {request.daily_hours}
- Weak Subjects: {weak_str}

SUBJECTS: {', '.join(subjects)}

DAYS TO PLAN:
{json.dumps(days_list, indent=2)}

RULES:
1. Allocate MORE time to weak subjects
2. Each task: 45-90 minutes max
3. Mix subjects across each day
4. Include revision sessions
5. Realistic and achievable
6. Match Indian medical curriculum

RESPOND IN THIS EXACT JSON FORMAT ONLY:
{{
  "plan": [
    {{
      "day": "Monday",
      "date": "2025-05-13",
      "total_hours": 6.0,
      "tasks": [
        {{
          "subject": "Anatomy",
          "topic": "Brachial Plexus Formation",
          "duration_minutes": 90,
          "mode": "deep-explanation",
          "priority": "high",
          "time_slot": "9:00 AM"
        }},
        {{
          "subject": "Pharmacology",
          "topic": "ANS — Cholinergic Drugs",
          "duration_minutes": 60,
          "mode": "rapid-recall",
          "priority": "high",
          "time_slot": "11:00 AM"
        }}
      ]
    }}
  ],
  "weekly_subject_split": {{
    "Anatomy": 22,
    "Physiology": 18,
    "Pharmacology": 25,
    "Pathology": 20,
    "Biochemistry": 15
  }},
  "ai_insight": "Your Pharmacology score is lowest. AI has allocated 25% of weekly time to Pharmacology. Focus on KD Tripathi ANS and CVS chapters this week.",
  "total_days_remaining": {days_remaining}
}}

Generate plan for all 7 days with 
{request.daily_hours} hours each day.
Return ONLY valid JSON. No extra text."""

    try:
        model = genai.GenerativeModel(
            "models/gemini-2.5-flash"
        )
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 4000,
            }
        )

        response_text = response.text.strip()

        # Strip markdown if present
        if "```json" in response_text:
            response_text = (
                response_text
                .split("```json")[1]
                .split("```")[0]
                .strip()
            )
        elif "```" in response_text:
            response_text = (
                response_text
                .split("```")[1]
                .split("```")[0]
                .strip()
            )

        plan_data = json.loads(response_text)

        return StudyPlanResponse(
            plan=plan_data.get("plan", []),
            weekly_subject_split=plan_data.get(
                "weekly_subject_split", {}
            ),
            ai_insight=plan_data.get(
                "ai_insight",
                "Stay consistent and focus on weak subjects!"
            ),
            total_days_remaining=days_remaining
        )

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI response parse error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Study plan error: {str(e)}"
        )


# ---------------------------------------------------------------------------
# Utility Endpoints
# ---------------------------------------------------------------------------

@app.get("/tracks")
async def get_tracks():
    return {
        "tracks": ["NEET", "MBBS", "BHMS", "BDS", "MD/MS"],
        "modes": list(MODE_INSTRUCTIONS.keys()),
        "lovable_modes": [
            "deep-explanation",
            "quick-summary",
            "mcq-practice",
            "rapid-recall"
        ]
    }


@app.get("/books")
async def get_books():
    return MEDICAL_BOOKS


@app.get("/status")
async def api_status():
    status = {
        "vector_search_available": True,
        "full_text_search_fallback": False,
        "gemini_query_api": "ok",
        "embedding_type": "fastembed_local",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimensions": 384,
        "embedding_cost": "free",
        "study_plan_endpoint": "available",
        "recommendations": []
    }

    try:
        test_emb = get_embedding("test medical query")
        if len(test_emb) == 384:
            status["vector_search_available"] = True
        else:
            status["recommendations"].append(
                f"Unexpected embedding size: {len(test_emb)}"
            )
    except Exception as e:
        status["vector_search_available"] = False
        status["full_text_search_fallback"] = True
        status["recommendations"].append(
            f"Embedding error: {str(e)[:80]}"
        )

    return status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
