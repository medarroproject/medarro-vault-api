import os
import io
import asyncio
from typing import List

import fitz  # PyMuPDF
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from openai import OpenAI
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    version="1.0.0",
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


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

CHUNK_WORDS = 500
OVERLAP_WORDS = 100


def extract_pages(pdf_bytes: bytes) -> List[dict]:
    """
    Extract text from each page of a PDF.
    Returns a list of dicts: {"page_number": int, "text": str}
    """
    pages = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")
        if text.strip():
            pages.append({"page_number": page_index + 1, "text": text})
    doc.close()
    return pages


def split_into_chunks(pages: List[dict]) -> List[dict]:
    """
    Slide a window of CHUNK_WORDS words over the full document text,
    advancing by (CHUNK_WORDS - OVERLAP_WORDS) words each step.
    Each chunk records the page_number where its first word originates.
    Returns a list of dicts: {"chunk_text": str, "page_number": int}
    """
    # Build a flat list of (word, page_number) tuples
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


def get_embedding(text: str) -> List[float]:
    """
    Generate an embedding for a single text string using OpenAI.
    """
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
    )
    return response.data[0].embedding


async def download_pdf(url: str) -> bytes:
    """
    Asynchronously download a PDF from a public URL.
    """
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download PDF. HTTP {response.status_code} from {url}",
            )
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type and not url.lower().endswith(".pdf"):
            # Warn but continue — some storage buckets don't set content-type correctly
            pass
        return response.content


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/upload-pdf", response_model=UploadPDFResponse)
async def upload_pdf(request: UploadPDFRequest):
    """
    Download a PDF from a public URL, extract text, chunk it,
    embed each chunk with OpenAI, and store everything in Supabase.
    """
    # 1. Download PDF
    try:
        pdf_bytes = await download_pdf(request.pdf_url)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading PDF: {str(e)}")

    # 2. Extract text page by page
    try:
        pages = extract_pages(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error extracting PDF text: {str(e)}")

    if not pages:
        raise HTTPException(status_code=422, detail="PDF appears to contain no extractable text.")

    # 3. Split into overlapping chunks
    chunks = split_into_chunks(pages)
    if not chunks:
        raise HTTPException(status_code=422, detail="No text chunks could be generated from this PDF.")

    # 4. Embed each chunk and insert into Supabase
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

            result = supabase.table("personal_vault").insert(row).execute()

            if not result.data:
                raise HTTPException(
                    status_code=500,
                    detail="Supabase insert returned no data. Check RLS policies and service key.",
                )
            inserted_count += 1

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during embedding/storage: {str(e)}")

    return UploadPDFResponse(status="ready", chunks_count=inserted_count)


@app.post("/search-vault", response_model=SearchVaultResponse)
async def search_vault(request: SearchVaultRequest):
    """
    Embed a user query and run cosine similarity search against
    the user's stored vault chunks via a Supabase RPC function.
    """
    # 1. Embed the query
    try:
        query_embedding = get_embedding(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating query embedding: {str(e)}")

    # 2. Call Supabase RPC for similarity search
    try:
        rpc_response = supabase.rpc(
            "match_vault_chunks",
            {
                "query_embedding": query_embedding,
                "match_user_id": request.user_id,
                "match_count": 5,
            },
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Supabase RPC: {str(e)}")

    if rpc_response.data is None:
        raise HTTPException(status_code=500, detail="Supabase RPC returned no data.")

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
