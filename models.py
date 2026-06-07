# models.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import date


# ---------------------------------------------------------------------------
# Auth & User
# ---------------------------------------------------------------------------

class UserPlanInfo(BaseModel):
    user_id: str
    plan: str  # trial | free | premium
    queries_used_today: int
    queries_limit_today: int
    trial_ends_at: Optional[str] = None
    trial_queries_used: Optional[int] = None
    trial_queries_limit: int = 30
    is_trial_active: bool = False
    premium_ends_at: Optional[str] = None


class DeviceRegisterRequest(BaseModel):
    device_fingerprint: str
    device_name: Optional[str] = None


class SessionTokenRequest(BaseModel):
    device_fingerprint: str


# ---------------------------------------------------------------------------
# AI Query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    mode: str = "deep-explanation"
    track: str = "NEET"
    context: str = ""
    prompt_version: str = "v3"
    user_id: Optional[str] = None


class AiQueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    mode: str = "deep-explanation"
    track: str = "NEET"


class AiQueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    cached: bool = False


# ---------------------------------------------------------------------------
# PDF Vault
# ---------------------------------------------------------------------------

class UploadPDFRequest(BaseModel):
    pdf_url: str
    pdf_name: str


class UploadPDFResponse(BaseModel):
    status: str
    chunks_count: int
    pdf_hash: str
    skipped: bool = False


class SearchVaultRequest(BaseModel):
    query: str


class VaultSearchResult(BaseModel):
    chunk_text: str
    pdf_name: str
    page_number: int
    similarity: float


class SearchVaultResponse(BaseModel):
    results: List[VaultSearchResult]


# ---------------------------------------------------------------------------
# Study Plan
# ---------------------------------------------------------------------------

class StudyPlanRequest(BaseModel):
    target_exam: str = "NEET UG"
    target_date: str
    daily_hours: int = Field(default=6, ge=1, le=16)
    study_days_per_week: int = Field(default=6, ge=1, le=7)
    weak_subjects: List[str] = []
    strong_subjects: List[str] = []
    completed_topics: List[str] = []
    pending_topics: List[str] = []
    last_mock_score: Optional[int] = None
    target_score: Optional[int] = None
    current_year: Optional[int] = None
    track: str = "NEET"


class StudyTask(BaseModel):
    subject: str
    topic: str
    duration_minutes: int
    mode: str
    priority: str
    time_slot: str
    task_type: str = "new_learning"


class StudyPlanDay(BaseModel):
    day: str
    date: str
    total_hours: float
    tasks: List[StudyTask]


class StudyPlanResponse(BaseModel):
    plan_id: str
    plan: List[StudyPlanDay]
    revision_schedule: List[Dict[str, Any]]
    weekly_subject_split: Dict[str, int]
    ai_insight: str
    motivation: str
    total_days_remaining: int
    syllabus_completion_percent: float
    weakness_score: float
    profile_analysis: Dict[str, Any]


class MarkTaskRequest(BaseModel):
    task_id: str
    completed: bool


class RescheduleBacklogRequest(BaseModel):
    plan_id: str


# ---------------------------------------------------------------------------
# Analytics & Admin
# ---------------------------------------------------------------------------

class AnalyticsEventRequest(BaseModel):
    event_type: str
    event_data: Dict[str, Any] = {}
