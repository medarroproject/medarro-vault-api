# middleware.py
import time
import hashlib
from typing import Optional, Tuple
from datetime import datetime, date, timezone

from fastapi import HTTPException, Request
from supabase import Client


# ---------------------------------------------------------------------------
# JWT Auth helper
# ---------------------------------------------------------------------------

async def get_current_user(request: Request, supabase: Client) -> dict:
    """
    Validates Supabase JWT from Authorization header.
    Returns user dict with id, email.
    Raises 401 on failure.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty bearer token")
    
    try:
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return {"id": user_response.user.id, "email": user_response.user.email}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token validation failed: {str(e)[:80]}")


# ---------------------------------------------------------------------------
# Rate limiter (in-memory, per user, 1 req/3 sec)
# ---------------------------------------------------------------------------

_rate_limit_store: dict = {}

def check_rate_limit(user_id: str) -> None:
    """Raises 429 if user is calling faster than 1 req/3 seconds."""
    now = time.time()
    last = _rate_limit_store.get(user_id, 0)
    if now - last < 3.0:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait 3 seconds between requests."
        )
    _rate_limit_store[user_id] = now


# ---------------------------------------------------------------------------
# Quota engine
# ---------------------------------------------------------------------------

QUOTA_CONFIG = {
    "premium": {"daily": 70, "is_total": False},
    "trial":   {"daily": 30, "is_total": True},
    "free":    {"daily": 5,  "is_total": False},
}


async def check_and_consume_quota(user_id: str, supabase: Client) -> dict:
    """
    Fetches user profile, checks quota, increments usage.
    Returns plan info dict.
    Raises 403 if quota exceeded or trial expired.
    """
    # Fetch profile
    try:
        profile_resp = supabase.table("user_profiles").select("*").eq("user_id", user_id).single().execute()
        profile = profile_resp.data
    except Exception:
        raise HTTPException(status_code=403, detail="User profile not found. Please re-login.")

    plan: str = profile["plan"]
    now_utc = datetime.now(timezone.utc)

    # --- Trial expiry auto-downgrade ---
    if plan == "trial":
        trial_ends = datetime.fromisoformat(profile["trial_ends_at"].replace("Z", "+00:00"))
        if now_utc > trial_ends:
            # Auto-downgrade to free
            supabase.table("user_profiles").update({"plan": "free"}).eq("id", user_id).execute()
            plan = "free"
        else:
            # Check total trial queries
            trial_used = profile.get("trial_queries_used", 0)
            if trial_used >= 30:
                raise HTTPException(
                    status_code=403,
                    detail="Trial query limit (30) reached. Upgrade to Premium for ₹499/month."
                )
            # Consume trial quota
            supabase.table("user_profiles").update(
                {"trial_queries_used": trial_used + 1, "updated_at": now_utc.isoformat()}
            ).eq("id", user_id).execute()
            return {
                "plan": "trial",
                "queries_used": trial_used + 1,
                "queries_limit": 30,
                "is_trial": True,
                "trial_ends_at": profile["trial_ends_at"],
            }

    # --- Premium expiry auto-downgrade ---
    if plan == "premium":
        premium_ends = profile.get("premium_ends_at")
        if premium_ends:
            premium_ends_dt = datetime.fromisoformat(premium_ends.replace("Z", "+00:00"))
            if now_utc > premium_ends_dt:
                supabase.table("user_profiles").update({"plan": "free"}).eq("id", user_id).execute()
                plan = "free"

    # --- Daily quota for free and premium ---
    config = QUOTA_CONFIG.get(plan, QUOTA_CONFIG["free"])
    daily_limit = config["daily"]
    today = date.today().isoformat()

    try:
        usage_resp = supabase.table("daily_usage").select("queries_used").eq("user_id", user_id).eq("usage_date", today).single().execute()
        current_used = usage_resp.data["queries_used"]
    except Exception:
        current_used = 0

    if current_used >= daily_limit:
        upgrade_msg = " Upgrade to Premium for 70 queries/day." if plan == "free" else ""
        raise HTTPException(
            status_code=403,
            detail=f"Daily query limit ({daily_limit}) reached.{upgrade_msg}"
        )

    # Upsert daily usage
    supabase.table("daily_usage").upsert(
        {"user_id": user_id, "usage_date": today, "queries_used": current_used + 1},
        on_conflict="user_id,usage_date"
    ).execute()

    return {
        "plan": plan,
        "queries_used": current_used + 1,
        "queries_limit": daily_limit,
        "is_trial": False,
    }


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

MAX_DEVICES = 2


async def register_device(user_id: str, device_fingerprint: str, device_name: Optional[str], supabase: Client) -> None:
    """
    Registers device. If already registered, updates last_seen.
    If 3rd new device, removes oldest before inserting.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Check if device already registered
    try:
        existing = supabase.table("user_devices").select("id").eq("user_id", user_id).eq("device_fingerprint", device_fingerprint).single().execute()
        if existing.data:
            supabase.table("user_devices").update({"last_seen_at": now}).eq("id", existing.data["id"]).execute()
            return
    except Exception:
        pass

    # Count current devices
    devices_resp = supabase.table("user_devices").select("id, registered_at").eq("user_id", user_id).order("registered_at").execute()
    devices = devices_resp.data or []

    if len(devices) >= MAX_DEVICES:
        # Remove oldest device
        oldest_id = devices[0]["id"]
        supabase.table("user_devices").delete().eq("id", oldest_id).execute()

    # Insert new device
    supabase.table("user_devices").insert({
        "user_id": user_id,
        "device_fingerprint": device_fingerprint,
        "device_name": device_name or "Unknown Device",
        "last_seen_at": now,
        "registered_at": now,
    }).execute()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def make_cache_key(query: str, mode: str, track: str) -> str:
    raw = f"{query.lower().strip()}|{mode}|{track}"
    return hashlib.sha256(raw.encode()).hexdigest()


def make_text_hash(text: str) -> str:
    return hashlib.sha256(text.lower().strip().encode()).hexdigest()


async def get_cached_response(cache_key: str, supabase: Client) -> Optional[str]:
    try:
        now = datetime.now(timezone.utc).isoformat()
        resp = supabase.table("response_cache").select("response_text").eq("cache_key", cache_key).gt("expires_at", now).single().execute()
        if resp.data:
            return resp.data["response_text"]
    except Exception:
        pass
    return None


async def set_cached_response(cache_key: str, query: str, mode: str, track: str, response_text: str, supabase: Client) -> None:
    from datetime import timedelta
    try:
        now = datetime.now(timezone.utc)
        expires = (now + timedelta(hours=24)).isoformat()
        query_hash = make_text_hash(query)
        supabase.table("response_cache").upsert({
            "cache_key": cache_key,
            "query_hash": query_hash,
            "mode": mode,
            "track": track,
            "response_text": response_text,
            "created_at": now.isoformat(),
            "expires_at": expires,
        }, on_conflict="cache_key").execute()
    except Exception as e:
        print(f"Cache write failed (non-fatal): {e}")


async def get_cached_embedding(text_hash: str, supabase: Client) -> Optional[list]:
    try:
        resp = supabase.table("embedding_cache").select("embedding").eq("text_hash", text_hash).single().execute()
        if resp.data:
            return resp.data["embedding"]
    except Exception:
        pass
    return None


async def set_cached_embedding(text_hash: str, embedding: list, supabase: Client) -> None:
    try:
        supabase.table("embedding_cache").upsert(
            {"text_hash": text_hash, "embedding": embedding},
            on_conflict="text_hash"
        ).execute()
    except Exception as e:
        print(f"Embedding cache write failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Analytics tracker
# ---------------------------------------------------------------------------

async def track_event(user_id: Optional[str], event_type: str, event_data: dict, supabase: Client) -> None:
    try:
        supabase.table("analytics_events").insert({
            "user_id": user_id,
            "event_type": event_type,
            "event_data": event_data,
        }).execute()
    except Exception as e:
        print(f"Analytics track failed (non-fatal): {e}")
