"""
═══════════════════════════════════════════════════════════════════════════════
  Agent Memory System — Persistent, Multi-Layer Memory
  Tracks conversations, student profiles, recommendations, and learning history.
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import PROJECT_ROOT

logger = logging.getLogger(__name__)

MEMORY_DIR = PROJECT_ROOT / "data" / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


class ConversationMemory:
    """
    Tracks conversation history per student session.
    Supports rolling window to manage context length.
    """

    def __init__(self, max_turns: int = 50):
        self.max_turns = max_turns
        self._history: dict[int, list[dict]] = defaultdict(list)

    def add_message(self, student_id: int, role: str, content: str):
        """Add a message to conversation history."""
        self._history[student_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        # Trim to max_turns (keep system message + last N)
        if len(self._history[student_id]) > self.max_turns:
            self._history[student_id] = self._history[student_id][-self.max_turns:]

    def get_history(self, student_id: int, last_n: Optional[int] = None) -> list[dict]:
        """Get conversation history for a student."""
        history = self._history.get(student_id, [])
        if last_n:
            return history[-last_n:]
        return history

    def get_context_messages(self, student_id: int) -> list[dict]:
        """Get messages formatted for LLM context."""
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.get_history(student_id)
        ]

    def clear(self, student_id: int):
        """Clear conversation history for a student."""
        self._history[student_id] = []

    def to_dict(self) -> dict:
        return dict(self._history)


class StudentProfileMemory:
    """
    Persistent student profiles with learning preferences,
    strengths, weaknesses, and metadata.
    """

    def __init__(self):
        self._profiles: dict[int, dict] = {}
        self._load()

    def get_profile(self, student_id: int) -> dict:
        """Get or create a student profile."""
        if student_id not in self._profiles:
            self._profiles[student_id] = {
                "student_id": student_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "demographics": {},
                "learning_style": None,
                "strengths": [],
                "weaknesses": [],
                "goals": [],
                "preferred_resource_types": [],
                "study_schedule_preference": None,
                "mastery_snapshot": {},
                "engagement_score": 0.0,
                "risk_level": "Unknown",
                "notes": [],
            }
        return self._profiles[student_id]

    def update_profile(self, student_id: int, updates: dict):
        """Update fields in a student's profile."""
        profile = self.get_profile(student_id)
        for key, value in updates.items():
            if key in profile:
                if isinstance(profile[key], list) and isinstance(value, list):
                    # Merge lists, deduplicate
                    profile[key] = list(set(profile[key] + value))
                elif isinstance(profile[key], dict) and isinstance(value, dict):
                    profile[key].update(value)
                else:
                    profile[key] = value
        profile["updated_at"] = datetime.now().isoformat()
        self._save()

    def add_simulated_boost(self, student_id: int, concept: str, boost_amount: float):
        profile = self.get_profile(student_id)
        if "simulated_boosts" not in profile:
            profile["simulated_boosts"] = {}
        
        current = profile["simulated_boosts"].get(concept, 0.0)
        profile["simulated_boosts"][concept] = min(0.99, current + boost_amount)
        self._save()

    def get_simulated_boosts(self, student_id: int) -> dict:
        return self.get_profile(student_id).get("simulated_boosts", {})

    def update_mastery(self, student_id: int, mastery_scores: dict):
        """Update mastery snapshot for a student."""
        profile = self.get_profile(student_id)
        profile["mastery_snapshot"] = mastery_scores
        profile["updated_at"] = datetime.now().isoformat()

        # Auto-detect strengths and weaknesses
        strengths = [c for c, s in mastery_scores.items() if s >= 0.7]
        weaknesses = [c for c, s in mastery_scores.items() if s < 0.4]
        profile["strengths"] = strengths
        profile["weaknesses"] = weaknesses

        self._save()

    def get_summary(self, student_id: int) -> str:
        """Get a natural language summary of the student profile."""
        profile = self.get_profile(student_id)

        summary_parts = [f"Student ID: {student_id}"]

        if profile.get("demographics"):
            demo = profile["demographics"]
            if "gender" in demo:
                summary_parts.append(f"Gender: {demo['gender']}")
            if "age_band" in demo:
                summary_parts.append(f"Age: {demo['age_band']}")
            if "highest_education" in demo:
                summary_parts.append(f"Education: {demo['highest_education']}")

        if profile.get("strengths"):
            summary_parts.append(f"Strengths: {', '.join(profile['strengths'][:5])}")
        if profile.get("weaknesses"):
            summary_parts.append(f"Weaknesses: {', '.join(profile['weaknesses'][:5])}")
        if profile.get("risk_level") != "Unknown":
            summary_parts.append(f"Risk Level: {profile['risk_level']}")
        if profile.get("engagement_score"):
            summary_parts.append(f"Engagement: {profile['engagement_score']:.1%}")

        return "\n".join(summary_parts)

    def _save(self):
        """Persist profiles to disk."""
        path = MEMORY_DIR / "student_profiles.json"
        serializable = {}
        for sid, profile in self._profiles.items():
            serializable[str(sid)] = profile
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

    def _load(self):
        """Load profiles from disk."""
        path = MEMORY_DIR / "student_profiles.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                self._profiles = {int(k): v for k, v in data.items()}
                logger.info(f"  Loaded {len(self._profiles)} student profiles from memory")
            except (json.JSONDecodeError, ValueError):
                self._profiles = {}


class RecommendationMemory:
    """
    Tracks past recommendations per student.
    Prevents repeated suggestions and enables feedback tracking.
    """

    def __init__(self):
        self._recommendations: dict[int, list[dict]] = defaultdict(list)
        self._load()

    def add_recommendation(
        self,
        student_id: int,
        concept: str,
        resources: list[dict],
        reason: str = "",
    ):
        """Record a recommendation event."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "concept": concept,
            "resources": [
                {"id": r.get("id", ""), "title": r.get("title", ""), "type": r.get("type", "")}
                for r in resources
            ],
            "reason": reason,
            "feedback": None,  # Will be updated later
            "completed": False,
        }
        self._recommendations[student_id].append(entry)
        self._save()

    def get_past_recommendations(
        self, student_id: int, last_n: Optional[int] = None
    ) -> list[dict]:
        """Get past recommendations for a student."""
        recs = self._recommendations.get(student_id, [])
        if last_n:
            return recs[-last_n:]
        return recs

    def get_recommended_resource_ids(self, student_id: int) -> set:
        """Get set of all previously recommended resource IDs."""
        ids = set()
        for rec in self._recommendations.get(student_id, []):
            for r in rec.get("resources", []):
                ids.add(r.get("id", ""))
        return ids

    def add_feedback(self, student_id: int, resource_id: str, rating: int, comment: str = ""):
        """Record student feedback on a recommendation."""
        for rec in reversed(self._recommendations.get(student_id, [])):
            for r in rec.get("resources", []):
                if r.get("id") == resource_id:
                    rec["feedback"] = {
                        "rating": rating,
                        "comment": comment,
                        "timestamp": datetime.now().isoformat(),
                    }
                    self._save()
                    return

    def mark_completed(self, student_id: int, resource_id: str):
        """Mark a recommended resource as completed."""
        for rec in reversed(self._recommendations.get(student_id, [])):
            for r in rec.get("resources", []):
                if r.get("id") == resource_id:
                    rec["completed"] = True
                    self._save()
                    return

    def get_summary(self, student_id: int) -> str:
        """Get summary of past recommendations."""
        recs = self._recommendations.get(student_id, [])
        if not recs:
            return "No past recommendations."

        total = len(recs)
        completed = sum(1 for r in recs if r.get("completed"))
        concepts = list(set(r["concept"] for r in recs))

        return (
            f"Past recommendations: {total} total, {completed} completed.\n"
            f"Concepts covered: {', '.join(concepts[:10])}"
        )

    def _save(self):
        path = MEMORY_DIR / "recommendations.json"
        serializable = {str(k): v for k, v in self._recommendations.items()}
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

    def _load(self):
        path = MEMORY_DIR / "recommendations.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                self._recommendations = defaultdict(list, {int(k): v for k, v in data.items()})
            except (json.JSONDecodeError, ValueError):
                self._recommendations = defaultdict(list)


class LearningHistory:
    """
    Tracks student learning events over time — scores, mastery changes,
    study sessions, milestones, and engagement patterns.
    """

    def __init__(self):
        self._events: dict[int, list[dict]] = defaultdict(list)
        self._load()

    def log_event(
        self,
        student_id: int,
        event_type: str,
        data: dict,
    ):
        """
        Log a learning event.

        Event types: 'assessment', 'mastery_update', 'study_session',
                     'resource_completed', 'milestone', 'gap_detected'
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data,
        }
        self._events[student_id].append(event)

        # Keep last 500 events per student
        if len(self._events[student_id]) > 500:
            self._events[student_id] = self._events[student_id][-500:]

        self._save()

    def get_events(
        self,
        student_id: int,
        event_type: Optional[str] = None,
        last_n: Optional[int] = None,
    ) -> list[dict]:
        """Get learning events, optionally filtered by type."""
        events = self._events.get(student_id, [])
        if event_type:
            events = [e for e in events if e["event_type"] == event_type]
        if last_n:
            events = events[-last_n:]
        return events

    def get_mastery_trajectory(self, student_id: int) -> list[dict]:
        """Get mastery score changes over time."""
        return self.get_events(student_id, event_type="mastery_update")

    def get_progress_summary(self, student_id: int) -> dict:
        """Calculate progress metrics from learning history."""
        events = self._events.get(student_id, [])
        if not events:
            return {"total_events": 0, "summary": "No learning history available."}

        mastery_events = [e for e in events if e["event_type"] == "mastery_update"]
        assessment_events = [e for e in events if e["event_type"] == "assessment"]
        resource_events = [e for e in events if e["event_type"] == "resource_completed"]

        # Calculate improvement
        improvement = 0.0
        if len(mastery_events) >= 2:
            first_mastery = mastery_events[0]["data"].get("avg_mastery", 0)
            last_mastery = mastery_events[-1]["data"].get("avg_mastery", 0)
            improvement = last_mastery - first_mastery

        return {
            "total_events": len(events),
            "total_assessments": len(assessment_events),
            "total_resources_completed": len(resource_events),
            "mastery_updates": len(mastery_events),
            "overall_improvement": round(improvement, 4),
            "improvement_pct": round(improvement * 100, 2),
            "first_event": events[0]["timestamp"] if events else None,
            "last_event": events[-1]["timestamp"] if events else None,
        }

    def get_weekly_summary(self, student_id: int) -> str:
        """Generate a natural language weekly summary."""
        progress = self.get_progress_summary(student_id)
        if progress["total_events"] == 0:
            return "No activity recorded this week."

        parts = [
            f"📊 Weekly Progress Report for Student {student_id}:",
            f"  • Total learning events: {progress['total_events']}",
            f"  • Assessments completed: {progress['total_assessments']}",
            f"  • Resources completed: {progress['total_resources_completed']}",
        ]

        if progress["overall_improvement"] > 0:
            parts.append(f"  • Overall improvement: +{progress['improvement_pct']}% 📈")
        elif progress["overall_improvement"] < 0:
            parts.append(f"  • Overall change: {progress['improvement_pct']}% 📉")
        else:
            parts.append("  • Mastery level: Stable ➡️")

        return "\n".join(parts)

    def _save(self):
        path = MEMORY_DIR / "learning_history.json"
        serializable = {str(k): v for k, v in self._events.items()}
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

    def _load(self):
        path = MEMORY_DIR / "learning_history.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                self._events = defaultdict(list, {int(k): v for k, v in data.items()})
            except (json.JSONDecodeError, ValueError):
                self._events = defaultdict(list)


class AgentMemory:
    """
    Unified memory interface that combines all memory layers.
    Single entry point for the agent to access all memory subsystems.
    """

    def __init__(self):
        self.conversation = ConversationMemory()
        self.profiles = StudentProfileMemory()
        self.recommendations = RecommendationMemory()
        self.history = LearningHistory()
        logger.info("  ✓ Agent memory system initialized (4 layers)")

    def get_full_context(self, student_id: int) -> str:
        """
        Build comprehensive context string for the agent,
        combining all memory layers for a specific student.
        """
        parts = []

        # Student profile
        profile_summary = self.profiles.get_summary(student_id)
        parts.append(f"=== STUDENT PROFILE ===\n{profile_summary}")

        # Past recommendations
        rec_summary = self.recommendations.get_summary(student_id)
        parts.append(f"\n=== PAST RECOMMENDATIONS ===\n{rec_summary}")

        # Learning history
        progress = self.history.get_progress_summary(student_id)
        if progress["total_events"] > 0:
            history_text = (
                f"Total events: {progress['total_events']}, "
                f"Assessments: {progress['total_assessments']}, "
                f"Resources completed: {progress['total_resources_completed']}, "
                f"Improvement: {progress['improvement_pct']}%"
            )
        else:
            history_text = "No learning history available."
        parts.append(f"\n=== LEARNING HISTORY ===\n{history_text}")

        # Recent conversation
        recent = self.conversation.get_history(student_id, last_n=5)
        if recent:
            conv_text = "\n".join([
                f"  {m['role']}: {m['content'][:200]}" for m in recent
            ])
            parts.append(f"\n=== RECENT CONVERSATION ===\n{conv_text}")

        return "\n".join(parts)

    def clear_student(self, student_id: int):
        """Clear all memory for a student."""
        self.conversation.clear(student_id)
        logger.info(f"  Cleared conversation memory for student {student_id}")
