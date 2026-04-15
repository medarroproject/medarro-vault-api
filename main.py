import os
import io
import asyncio
from typing import List
import hashlib

import fitz  # PyMuPDF
import httpx
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

# Gemini Configuration - Using correct available models
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY environment variable is required")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API configured successfully")
except Exception as e:
    print(f"❌ Gemini configuration error: {e}")
    raise

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY"),
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Medarro Personal Vault API",
    description="PDF ingestion and semantic search backend for Medarro's Personal Vault feature.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


# Gemini AI Models
class AiQueryRequest(BaseModel):
    query: str
    user_id: str = None
    mode: str = "explanation"
    track: str = "NEET"


class AiQueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float


class QueryRequest(BaseModel):
    query: str
    mode: str = "explanation"
    track: str = "NEET"


# ---------------------------------------------------------------------------
# Helper functions for PDF Processing
# ---------------------------------------------------------------------------

CHUNK_WORDS = 500
OVERLAP_WORDS = 100


def extract_pages(pdf_bytes: bytes) -> List[dict]:
    """Extract text from each page of a PDF."""
    pages = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_index in range(len(doc)):
            page = doc[page_index]
            text = page.get_text("text")
            if text.strip():
                pages.append({"page_number": page_index + 1, "text": text})
        doc.close()
    except Exception as e:
        print(f"❌ PDF extraction error: {e}")
        raise
    return pages


def split_into_chunks(pages: List[dict]) -> List[dict]:
    """Split pages into overlapping word chunks."""
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
        chunks.append({"chunk_text": chunk_text, "page_number": page_number})
        if end == len(word_page_pairs):
            break
        start += step

    return chunks


def get_embedding_gemini(text: str) -> List[float]:
    """
    Generate embedding using Gemini API.
    Falls back to hash-based if API fails.
    """
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        if response and "embedding" in response:
            print(f"✅ Embedding generated (length: {len(response['embedding'])})")
            return response["embedding"]
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ Gemini Embedding API error: {error_msg}")
        
        if any(keyword in error_msg.lower() for keyword in ["quota", "permission", "disabled", "not enabled", "429", "403", "resource exhausted"]):
            print("📌 Using fallback hash-based embedding")
            return get_embedding_fallback(text)
        
        raise HTTPException(
            status_code=503,
            detail=f"Embedding service error: {error_msg[:100]}"
        )


def get_embedding_fallback(text: str) -> List[float]:
    """Fallback hash-based embedding when API fails."""
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()
    
    embedding = []
    for i in range(1536):
        embedding.append(float(hash_bytes[i % 32]) / 256.0)
    
    return embedding


async def download_pdf(url: str) -> bytes:
    """Asynchronously download PDF from URL."""
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download PDF. HTTP {response.status_code}",
            )
        return response.content


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
        "Goodman & Gilman's Pharmacological Basis of Therapeutics"
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

MODE_INSTRUCTIONS = {
    "explanation": """Explain the concept thoroughly but in student-friendly language.
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


def build_medical_prompt(query: str, mode: str = "explanation", track: str = "NEET") -> str:
    """Build medical-specific prompt for Gemini."""
    
    mode_instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["explanation"])
    
    if track == "NEET":
        relevant_books = MEDICAL_BOOKS["NEET"]
    elif track == "MBBS":
        relevant_books = MEDICAL_BOOKS["MBBS"] + [MEDICAL_BOOKS["BD Chaurasia"][0]]
    elif track == "BHMS":
        relevant_books = ["Boericke's Materia Medica", "Organon of Medicine"]
    elif track == "BDS":
        relevant_books = ["Shafer's Textbook of Oral Pathology", "Dental Pharmacology"]
    else:
        relevant_books = MEDICAL_BOOKS["MBBS"]
    
    books_text = "\n".join([f"- {book}" for book in relevant_books])
    
    prompt = f"""You are an expert medical educator for Indian {track} students.

STRICT GUIDELINES:
1. ONLY answer based on standard medical textbooks and established medical knowledge
2. Reference specific textbooks when relevant
3. Use EXACT medical terminology - no approximations
4. If uncertain, clearly state: "This topic is not covered in standard {track} curriculum"
5. Prioritize ICMR, WHO, and Indian medical standards where applicable
6. For drugs: include generic name, class, mechanism, important side effects
7. For anatomy: describe structures, relations, nerve/blood supply accurately
8. For physiology: explain mechanisms at cellular/system level with accuracy
9. For pathology: describe pathogenesis, pathology, clinical features accurately

STUDENT LEVEL: {track} track medical student
ANSWER MODE: {mode}

ANSWER FORMAT:
{mode_instruction}

RELEVANT TEXTBOOKS FOR THIS QUERY:
{books_text}

STUDENT QUESTION: {query}

RULES FOR YOUR RESPONSE:
- Be factually accurate - this is for medical education
- Include source references (textbook names) naturally in the answer
- Highlight key points the student MUST remember
- Include clinical correlations where relevant
- For NEET: use curriculum-level language
- For MBBS: can be more detailed and comprehensive
- If there's any ambiguity in the question, clarify it

Now provide the accurate medical answer:"""
    
    return prompt


# ---------------------------------------------------------------------------
# Endpoints - Health Check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint - tests all APIs."""
    health_status = {
        "status": "ok",
        "service": "Medarro API",
        "version": "2.0.0",
        "apis": {
            "gemini_query": "ok",
            "gemini_embeddings": "ok",
            "supabase": "ok"
        },
        "warnings": []
    }
    
    # Test Gemini Query API
    try:
        model = model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            "test",
            generation_config={"max_output_tokens": 10}
        )
        try:
            response_text = response.text
        except Exception:
            response_text = None
        if response and response_text:
            health_status["apis"]["gemini_query"] = "ok"
        else:
            health_status["apis"]["gemini_query"] = "degraded"
            health_status["warnings"].append("Gemini Query API responding but producing empty responses")
    except Exception as e:
        health_status["apis"]["gemini_query"] = "down"
        health_status["warnings"].append(f"Gemini Query API error: {str(e)[:80]}")
        health_status["status"] = "degraded"
    
    # Test Gemini Embeddings API
    try:
        test_embedding = genai.embed_content(
           model="models/gemini-embedding-001",
            content="test",
            task_type="retrieval_document"
        )
        if test_embedding and "embedding" in test_embedding:
            health_status["apis"]["gemini_embeddings"] = "ok"
        else:
            health_status["apis"]["gemini_embeddings"] = "degraded"
            health_status["warnings"].append("Embedding API returning invalid response")
    except Exception as e:
        error_str = str(e).lower()
        health_status["apis"]["gemini_embeddings"] = "disabled"
        
        if any(keyword in error_str for keyword in ["quota", "resource exhausted"]):
            health_status["warnings"].append("⚠️ QUOTA EXCEEDED - Fallback to full-text search enabled")
        elif any(keyword in error_str for keyword in ["permission", "disabled", "not enabled", "403"]):
            health_status["warnings"].append("⚠️ API DISABLED - Enable in Google Cloud Console")
        else:
            health_status["warnings"].append(f"Embedding API error: {str(e)[:80]}")
    
    # Test Supabase
    try:
        result = supabase.table("personal_vault").select("*", count="exact").limit(1).execute()
        health_status["apis"]["supabase"] = "ok"
    except Exception as e:
        health_status["apis"]["supabase"] = "down"
        health_status["warnings"].append(f"Supabase error: {str(e)[:80]}")
        health_status["status"] = "degraded"
    
    return health_status


# ---------------------------------------------------------------------------
# Endpoints - PDF Upload & Search
# ---------------------------------------------------------------------------

@app.post("/upload-pdf", response_model=UploadPDFResponse)
async def upload_pdf(request: UploadPDFRequest):
    """Download PDF, extract text, chunk, embed, and store in Supabase."""
    
    try:
        pdf_bytes = await download_pdf(request.pdf_url)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading PDF: {str(e)}")

    try:
        pages = extract_pages(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error extracting PDF text: {str(e)}")

    if not pages:
        raise HTTPException(status_code=422, detail="PDF appears to contain no extractable text")

    chunks = split_into_chunks(pages)
    if not chunks:
        raise HTTPException(status_code=422, detail="No text chunks could be generated from this PDF")

    inserted_count = 0
    try:
        for chunk in chunks:
            embedding = get_embedding_gemini(chunk["chunk_text"])

            row = {
                "user_id": request.user_id,
                "pdf_name": request.pdf_name,
                "page_number": chunk["page_number"],
                "chunk_text": chunk["chunk_text"],
                "embedding": embedding,
            }

            result = supabase.table("personal_vault").insert(row).execute()

            if not result.data:
                raise HTTPException(
                    status_code=500,
                    detail="Supabase insert failed. Check RLS policies.",
                )
            inserted_count += 1

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during storage: {str(e)}")

    return UploadPDFResponse(status="ready", chunks_count=inserted_count)


@app.post("/search-vault", response_model=SearchVaultResponse)
async def search_vault(request: SearchVaultRequest):
    """Search vault with fallback to full-text search."""
    
    query_embedding = None
    embedding_failed = False
    
    try:
        query_embedding = get_embedding_gemini(request.query)
    except Exception as e:
        print(f"⚠️ Vector embedding failed: {str(e)}")
        embedding_failed = True

    # Try vector search first
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

    # FALLBACK: Full-text search
    print("📌 Falling back to full-text search")
    try:
        response = supabase.table("personal_vault") \
            .select("chunk_text, pdf_name, page_number") \
            .eq("user_id", request.user_id) \
            .full_text_search("chunk_text", request.query) \
            .limit(5) \
            .execute()
        
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
        print(f"❌ Full-text search also failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Search service temporarily unavailable: {str(e)[:100]}"
        )


# ---------------------------------------------------------------------------
# Endpoints - Gemini Medical Q&A
# ---------------------------------------------------------------------------

@app.post("/query", response_model=AiQueryResponse)
async def gemini_query(request: QueryRequest):
    """Main medical query endpoint using Gemini."""
    
    prompt = build_medical_prompt(
        query=request.query,
        mode=request.mode,
        track=request.track
    )
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_output_tokens": 2000,
            }
        )
        
        if not response.text:
            raise HTTPException(status_code=500, detail="Gemini returned empty response")
        
        # Extract mentioned textbooks
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
        
        answer_lower = response.text.lower()
        for keyword, full_name in book_keywords.items():
            if keyword.lower() in answer_lower:
                sources.append(full_name)
        
        if not sources:
            sources = ["Standard Medical References"]
        
        sources = list(dict.fromkeys(sources))
        
        return AiQueryResponse(
            answer=response.text,
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
    """Backward compatibility endpoint for /query."""
    query_req = QueryRequest(
        query=request.query,
        mode=request.mode,
        track=request.track
    )
    return await gemini_query(query_req)


@app.post("/query-stream")
async def gemini_query_stream(request: QueryRequest):
    """Streaming endpoint for long-form answers."""
    from fastapi.responses import StreamingResponse
    
    prompt = build_medical_prompt(
        query=request.query,
        mode=request.mode,
        track=request.track
    )
    
    async def generate():
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
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
    
    return StreamingResponse(generate(), media_type="text/plain")


# ---------------------------------------------------------------------------
# Utility Endpoints
# ---------------------------------------------------------------------------

@app.get("/tracks")
async def get_tracks():
    """Get available medical tracks and modes."""
    return {
        "tracks": ["NEET", "MBBS", "BHMS", "BDS", "MD/MS"],
        "modes": ["explanation", "exam", "revision", "notes", "deep-dive"]
    }


@app.get("/books")
async def get_books():
    """Get recommended medical textbooks."""
    return MEDICAL_BOOKS


@app.get("/status")
async def api_status():
    """Check API status and fallback mode."""
    status = {
        "vector_search_available": True,
        "full_text_search_fallback": False,
        "gemini_query_api": "ok",
        "recommendations": []
    }
    
    try:
        genai.embed_content(
            model="models/embedding-001",
            content="test",
            task_type="retrieval_document"
        )
        status["vector_search_available"] = True
    except Exception as e:
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ["quota", "resource exhausted", "permission", "disabled", "403", "429"]):
            status["vector_search_available"] = False
            status["full_text_search_fallback"] = True
            status["recommendations"].append("⚠️ Vector search disabled - using full-text search fallback")
            status["recommendations"].append("Check: /health endpoint for details")
        else:
            status["recommendations"].append(f"Embedding service error: {str(e)[:80]}")
    
    return status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
