# study_plan_service.py
"""
Deterministic study plan engine.
Gemini is called ONLY for ai_insight and motivation text.
All scheduling logic is pure Python.
"""

import uuid
import math
from datetime import date, timedelta
from typing import List, Dict, Tuple, Optional, Any

import google.generativeai as genai

from models import StudyPlanRequest, StudyPlanDay, StudyTask, StudyPlanResponse


# ---------------------------------------------------------------------------
# Subject configuration per track
# ---------------------------------------------------------------------------

TRACK_SUBJECTS: Dict[str, List[Dict]] = {
    "NEET": [
        {"name": "Biology",   "weight": 50, "daily_sessions": 2},
        {"name": "Physics",   "weight": 25, "daily_sessions": 1},
        {"name": "Chemistry", "weight": 25, "daily_sessions": 1},
    ],
    "MBBS": [
        {"name": "Anatomy",       "weight": 20, "daily_sessions": 1},
        {"name": "Physiology",    "weight": 20, "daily_sessions": 1},
        {"name": "Biochemistry",  "weight": 15, "daily_sessions": 1},
        {"name": "Pharmacology",  "weight": 15, "daily_sessions": 1},
        {"name": "Pathology",     "weight": 20, "daily_sessions": 1},
        {"name": "Microbiology",  "weight": 10, "daily_sessions": 1},
    ],
    "BDS": [
        {"name": "Anatomy",         "weight": 25, "daily_sessions": 1},
        {"name": "Physiology",      "weight": 20, "daily_sessions": 1},
        {"name": "Biochemistry",    "weight": 15, "daily_sessions": 1},
        {"name": "Dental Materials","weight": 20, "daily_sessions": 1},
        {"name": "Oral Pathology",  "weight": 20, "daily_sessions": 1},
    ],
    "BHMS": [
        {"name": "Organon",        "weight": 25, "daily_sessions": 1},
        {"name": "Materia Medica", "weight": 30, "daily_sessions": 1},
        {"name": "Anatomy",        "weight": 25, "daily_sessions": 1},
        {"name": "Physiology",     "weight": 20, "daily_sessions": 1},
    ],
}

HIGH_YIELD_TOPICS: Dict[str, List[str]] = {
    "Biology": [
        "Cell Division", "Genetics & Heredity", "Human Physiology",
        "Biotechnology", "Ecology", "Plant Physiology", "Biomolecules",
        "Animal Kingdom", "Evolution", "Reproductive Health"
    ],
    "Physics": [
        "Laws of Motion", "Electrostatics", "Current Electricity",
        "Optics", "Modern Physics", "Thermodynamics", "Work Energy Power",
        "Fluid Mechanics", "Magnetism", "Oscillations"
    ],
    "Chemistry": [
        "Chemical Bonding", "Organic Reactions", "Electrochemistry",
        "Coordination Compounds", "Equilibrium", "Thermochemistry",
        "Biomolecules Chemistry", "Polymers", "d-Block Elements", "Aldehydes Ketones"
    ],
    "Anatomy": [
        "Brachial Plexus", "Femoral Triangle", "Heart Anatomy",
        "Cranial Nerves", "Lung Anatomy", "Histology", "Embryology",
        "Upper Limb", "Lower Limb", "Abdomen"
    ],
    "Physiology": [
        "Cardiac Cycle", "Nerve Muscle", "Renal Physiology",
        "GI Physiology", "Respiratory", "Endocrinology", "Blood",
        "CNS", "Special Senses", "Reproductive Physiology"
    ],
    "Biochemistry": [
        "Enzymes", "Carbohydrate Metabolism", "Lipid Metabolism",
        "Protein Structure", "DNA Replication", "TCA Cycle",
        "Vitamins", "Minerals", "Hemoglobin", "Nucleotides"
    ],
    "Pharmacology": [
        "Autonomic Pharmacology", "Cardiovascular Drugs", "Antibiotics",
        "CNS Drugs", "Antiepileptics", "Anti-TB Drugs", "Analgesics",
        "Chemotherapy", "Antidiabetics", "Drug Interactions"
    ],
    "Pathology": [
        "Cell Injury", "Inflammation", "Neoplasia", "Cardiovascular Pathology",
        "Respiratory Pathology", "GI Pathology", "Renal Pathology",
        "Blood Disorders", "Liver Pathology", "Neuropathology"
    ],
    "Microbiology": [
        "Bacteriology", "Virology", "Mycology", "Parasitology",
        "Immunology", "Hospital Infections", "TB Microbiology",
        "Serology", "Culture Media", "Sterilization"
    ],
    "Organon": [
        "Aphorisms 1-30", "Aphorisms 31-70", "Miasms",
        "Individualization", "Case Taking", "Potency Selection",
        "Second Prescription", "Hering's Law", "Drug Proving", "Suppression"
    ],
    "Materia Medica": [
        "Sulphur", "Nux Vomica", "Pulsatilla", "Calcarea Carb",
        "Lycopodium", "Phosphorus", "Arsenicum Album", "Belladonna",
        "Bryonia", "Apis Mellifica"
    ],
    "Dental Materials": [
        "Impression Materials", "Gypsum Products", "Cements",
        "Amalgam", "Composite Resins", "Waxes", "Casting Alloys",
        "Ceramics", "Acrylic Resins", "Abrasives"
    ],
    "Oral Pathology": [
        "Caries", "Pulpitis", "Periodontitis", "Cysts",
        "Tumors of Jaw", "Salivary Gland Pathology", "Oral Cancer",
        "White Lesions", "Infections", "Developmental Anomalies"
    ],
}

DAILY_TIME_SLOTS = {
    "morning":   ["07:00 AM", "08:00 AM", "09:00 AM", "10:00 AM"],
    "afternoon": ["12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM"],
    "evening":   ["05:00 PM", "06:00 PM"],
    "night":     ["08:00 PM", "09:00 PM", "10:00 PM"],
}

SLOT_MODES = {
    "morning":   "deep-explanation",
    "afternoon": "mcq-practice",
    "evening":   "quick-summary",
    "night":     "rapid-recall",
}

SPACED_REVISION_DAYS = [1, 3, 7, 15, 30]


# ---------------------------------------------------------------------------
# Profile analysis
# ---------------------------------------------------------------------------

def analyse_profile(request: StudyPlanRequest, days_remaining: int) -> Dict[str, Any]:
    subjects = TRACK_SUBJECTS.get(request.track, TRACK_SUBJECTS["MBBS"])
    total_subjects = len(subjects)

    completed_count = len(request.completed_topics)
    total_topics = max(len(request.pending_topics) + completed_count, 10)
    syllabus_completion = round((completed_count / total_topics) * 100, 1)

    weak_count = len(request.weak_subjects)
    weakness_score = round(min((weak_count / max(total_subjects, 1)) * 100, 100), 1)

    backlog_load = len(request.pending_topics)
    revision_load = max(0, completed_count - 5)

    score_gap = 0
    if request.last_mock_score and request.target_score:
        score_gap = max(0, request.target_score - request.last_mock_score)

    urgency = "low"
    if days_remaining < 30:
        urgency = "critical"
    elif days_remaining < 60:
        urgency = "high"
    elif days_remaining < 90:
        urgency = "medium"

    return {
        "days_remaining": days_remaining,
        "syllabus_completion_percent": syllabus_completion,
        "weakness_score": weakness_score,
        "backlog_load": backlog_load,
        "revision_load": revision_load,
        "score_gap": score_gap,
        "urgency": urgency,
        "total_subjects": total_subjects,
    }


# ---------------------------------------------------------------------------
# Subject weight calculation (adjusts for weak subjects)
# ---------------------------------------------------------------------------

def compute_subject_weights(request: StudyPlanRequest) -> Dict[str, float]:
    base_subjects = TRACK_SUBJECTS.get(request.track, TRACK_SUBJECTS["MBBS"])
    weights: Dict[str, float] = {s["name"]: s["weight"] for s in base_subjects}

    # Boost weak subjects by 10% each, pull from strong subjects
    for weak in request.weak_subjects:
        if weak in weights:
            weights[weak] = min(weights[weak] + 10, 60)

    for strong in request.strong_subjects:
        if strong in weights and strong not in request.weak_subjects:
            weights[strong] = max(weights[strong] - 5, 10)

    # Normalize to 100
    total = sum(weights.values())
    normalized = {k: round((v / total) * 100) for k, v in weights.items()}

    # Fix rounding drift
    diff = 100 - sum(normalized.values())
    if diff != 0:
        first_key = next(iter(normalized))
        normalized[first_key] += diff

    return normalized


# ---------------------------------------------------------------------------
# Topic list builder
# ---------------------------------------------------------------------------

def get_topic_pool(subject: str, completed_topics: List[str]) -> List[str]:
    all_topics = HIGH_YIELD_TOPICS.get(subject, [f"{subject} Core Concepts", f"{subject} Clinical Applications"])
    pending = [t for t in all_topics if t not in completed_topics]
    return pending if pending else all_topics


# ---------------------------------------------------------------------------
# Daily task builder
# ---------------------------------------------------------------------------

def build_day_tasks(
    day_index: int,
    daily_hours: int,
    subjects_for_day: List[str],
    weak_subjects: List[str],
    completed_topics: List[str],
    is_test_day: bool,
) -> List[StudyTask]:
    tasks: List[StudyTask] = []
    total_minutes = daily_hours * 60

    if is_test_day:
        # Weekly MCQ test — single block
        tasks.append(StudyTask(
            subject="All Subjects",
            topic="Weekly Mock Test — Full Syllabus MCQs",
            duration_minutes=min(total_minutes, 180),
            mode="mcq-practice",
            priority="high",
            time_slot="09:00 AM",
            task_type="mock_test",
        ))
        return tasks

    # Distribute minutes across subjects proportionally
    per_subject_minutes = total_minutes // len(subjects_for_day)
    remainder = total_minutes % len(subjects_for_day)

    # Map sessions to time-of-day slots
    slot_keys = list(DAILY_TIME_SLOTS.keys())
    slot_idx = 0

    for i, subject in enumerate(subjects_for_day):
        topic_pool = get_topic_pool(subject, completed_topics)
        topic = topic_pool[day_index % len(topic_pool)]

        duration = per_subject_minutes + (remainder if i == 0 else 0)
        priority = "high" if subject in weak_subjects else "medium"

        slot_key = slot_keys[slot_idx % len(slot_keys)]
        time_slot_list = DAILY_TIME_SLOTS[slot_key]
        time_slot = time_slot_list[i % len(time_slot_list)]
        mode = SLOT_MODES[slot_key]

        # Morning = new learning, Afternoon = MCQs, Evening = revision, Night = recall
        task_type_map = {
            "morning":   "new_learning",
            "afternoon": "mcq_practice",
            "evening":   "revision",
            "night":     "rapid_recall",
        }

        tasks.append(StudyTask(
            subject=subject,
            topic=topic,
            duration_minutes=max(30, duration),
            mode=mode,
            priority=priority,
            time_slot=time_slot,
            task_type=task_type_map[slot_key],
        ))
        slot_idx += 1

    return tasks


# ---------------------------------------------------------------------------
# Spaced revision schedule builder
# ---------------------------------------------------------------------------

def build_revision_schedule(
    plan_start: date,
    plan_days: List[StudyPlanDay],
) -> List[Dict[str, Any]]:
    revision_entries: List[Dict[str, Any]] = []

    for day in plan_days:
        learned_date = date.fromisoformat(day.date)
        for task in day.tasks:
            if task.task_type == "new_learning":
                for stage, offset in enumerate(SPACED_REVISION_DAYS, start=1):
                    revision_date = learned_date + timedelta(days=offset)
                    revision_entries.append({
                        "topic": task.topic,
                        "subject": task.subject,
                        "first_learned_date": day.date,
                        "next_revision_date": revision_date.isoformat(),
                        "revision_stage": stage,
                    })

    return revision_entries


# ---------------------------------------------------------------------------
# Gemini call for ai_insight + motivation only
# ---------------------------------------------------------------------------

def generate_ai_insight(
    request: StudyPlanRequest,
    profile: Dict[str, Any],
    gemini_key: str,
    model_name: str = "models/gemini-2.5-flash",
) -> Tuple[str, str]:
    try:
        genai.configure(api_key=gemini_key)
        weak_str = ", ".join(request.weak_subjects) if request.weak_subjects else "none specified"
        score_info = ""
        if request.last_mock_score and request.target_score:
            score_info = f"Last mock: {request.last_mock_score}%, Target: {request.target_score}%."

        prompt = (
            f"You are a medical exam coach. A student is preparing for {request.target_exam} "
            f"({request.track} track) with {profile['days_remaining']} days left. "
            f"Weak subjects: {weak_str}. {score_info} "
            f"Urgency level: {profile['urgency']}. Completion: {profile['syllabus_completion_percent']}%.\n\n"
            "Write TWO things only:\n"
            "1. AI_INSIGHT: 2-3 actionable sentences on what to prioritize this week.\n"
            "2. MOTIVATION: 1 powerful motivational sentence for this student.\n\n"
            "Format:\nAI_INSIGHT: <text>\nMOTIVATION: <text>"
        )

        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0.7, "max_output_tokens": 300}
        )
        raw = resp.text.strip() if resp and resp.text else ""

        ai_insight = "Focus on your weak subjects daily. Consistency beats cramming."
        motivation = "Every expert was once a beginner — keep going."

        for line in raw.split("\n"):
            if line.startswith("AI_INSIGHT:"):
                ai_insight = line.replace("AI_INSIGHT:", "").strip()
            elif line.startswith("MOTIVATION:"):
                motivation = line.replace("MOTIVATION:", "").strip()

        return ai_insight, motivation

    except Exception as e:
        print(f"Gemini insight generation failed (non-fatal): {e}")
        weak_label = ", ".join(request.weak_subjects[:2]) if request.weak_subjects else "your weak areas"
        return (
            f"Prioritize {weak_label} with daily MCQ practice and rapid recall sessions.",
            "Your consistency today builds your confidence tomorrow.",
        )


# ---------------------------------------------------------------------------
# Main plan generator
# ---------------------------------------------------------------------------

def generate_study_plan(
    user_id: str,
    request: StudyPlanRequest,
    gemini_key: str,
) -> StudyPlanResponse:
    from datetime import datetime as dt

    # Days remaining
    try:
        target = date.fromisoformat(request.target_date)
        days_remaining = max((target - date.today()).days, 1)
    except Exception:
        days_remaining = 90

    profile = analyse_profile(request, days_remaining)
    subject_weights = compute_subject_weights(request)
    subjects_list = list(subject_weights.keys())

    plan_days: List[StudyPlanDay] = []
    plan_start = date.today()

    for i in range(7):
        current_date = plan_start + timedelta(days=i)

        # Every 7th day = MCQ test
        is_test_day = (i + 1) % 7 == 0

        # Grand test every 14th day (when you call it for a 14-day plan; for 7-day we skip)
        # Subject rotation: distribute subjects across days
        num_subjects_per_day = min(3, len(subjects_list))
        start_idx = (i * num_subjects_per_day) % len(subjects_list)
        day_subjects = [subjects_list[(start_idx + j) % len(subjects_list)] for j in range(num_subjects_per_day)]

        tasks = build_day_tasks(
            day_index=i,
            daily_hours=request.daily_hours,
            subjects_for_day=day_subjects,
            weak_subjects=request.weak_subjects,
            completed_topics=request.completed_topics,
            is_test_day=is_test_day,
        )

        plan_days.append(StudyPlanDay(
            day=current_date.strftime("%a"),
            date=current_date.isoformat(),
            total_hours=float(request.daily_hours),
            tasks=tasks,
        ))

    revision_entries = build_revision_schedule(plan_start, plan_days)
    ai_insight, motivation = generate_ai_insight(request, profile, gemini_key)

    plan_id = str(uuid.uuid4())

    return StudyPlanResponse(
        plan_id=plan_id,
        plan=plan_days,
        revision_schedule=revision_entries,
        weekly_subject_split={k: int(v) for k, v in subject_weights.items()},
        ai_insight=ai_insight,
        motivation=motivation,
        total_days_remaining=days_remaining,
        syllabus_completion_percent=profile["syllabus_completion_percent"],
        weakness_score=profile["weakness_score"],
        profile_analysis=profile,
    )
