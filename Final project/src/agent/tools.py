"""
═══════════════════════════════════════════════════════════════════════════════
  Agent Tools — the 5 tools used by the learning agent
  Each tool wraps a model/service and returns structured results.
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import MODELS_DIR, PROCESSED_DIR, MASTERY_LEVELS, format_module_name

logger = logging.getLogger(__name__)


class AgentTools:
    """
    Wrapper class that provides the 5 agent tools.
    Each tool loads models lazily and caches them.
    """

    def __init__(self, llm=None):
        self.llm = llm
        self._dkt_index = None
        self._dkt_trainer = None
        self._gap_detector = None
        self._recommender = None
        self._gap_features: Optional[pd.DataFrame] = None
        self._dkt_data: Optional[dict] = None
        self._student_profiles: Optional[pd.DataFrame] = None
        self._interaction_matrix: Optional[pd.DataFrame] = None

    # ── Lazy Loaders ──────────────────────────────────────────────────────

    def _load_dkt(self):
        if self._dkt_trainer is None:
            try:
                from src.models.dkt_trainer import DKTTrainer
                self._dkt_trainer = DKTTrainer.load_trained_model()
            except Exception as e:
                logger.warning(f"Could not load DKT model: {e}")
        return self._dkt_trainer

    def _load_gap_detector(self):
        if self._gap_detector is None:
            try:
                from src.models.gap_detector import GapDetector
                self._gap_detector = GapDetector.load()
            except Exception as e:
                logger.warning(f"Could not load Gap Detector: {e}")
        return self._gap_detector

    def _load_recommender(self):
        if self._recommender is None:
            try:
                from src.models.recommender import Recommender
                self._recommender = Recommender(llm=self.llm)
            except Exception as e:
                logger.warning(f"Could not load Recommender: {e}")
        return self._recommender

    def _load_gap_features(self) -> pd.DataFrame:
        if self._gap_features is None:
            path = PROCESSED_DIR / "gap_features.csv"
            if path.exists():
                self._gap_features = pd.read_csv(path)
            else:
                self._gap_features = pd.DataFrame()
        return self._gap_features

    def _load_dkt_data(self) -> dict:
        if self._dkt_data is None:
            path = PROCESSED_DIR / "dkt_sequences.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    self._dkt_data = pickle.load(f)
            else:
                self._dkt_data = {"sequences": [], "student_ids": [], "num_concepts": 0, "concept_map": {}}
        return self._dkt_data

    def _load_student_profiles(self) -> pd.DataFrame:
        if self._student_profiles is None:
            path = PROCESSED_DIR / "student_profiles.csv"
            if path.exists():
                self._student_profiles = pd.read_csv(path)
            else:
                self._student_profiles = pd.DataFrame()
        return self._student_profiles

    def _load_interaction_matrix(self) -> pd.DataFrame:
        if self._interaction_matrix is None:
            path = PROCESSED_DIR / "interaction_matrix.csv"
            if path.exists():
                self._interaction_matrix = pd.read_csv(path, index_col=0)
            else:
                self._interaction_matrix = pd.DataFrame()
        return self._interaction_matrix

    def _dkt_student_index(self, dkt_data: dict) -> dict:
        if self._dkt_index is None:
            self._dkt_index = {int(sid): i for i, sid in enumerate(dkt_data.get("student_ids", []))}
        return self._dkt_index

    # ═══════════════════════════════════════════════════════════════════════
    #  Tool 1: Knowledge Tracer
    # ═══════════════════════════════════════════════════════════════════════

    def knowledge_tracer(self, student_id: int, memory=None) -> dict:
        """
        Analyze a student's mastery level per concept using Deep Knowledge Tracing.
        Returns mastery scores (0-1) for each concept and overall knowledge state.
        """
        logger.info(f"🔍 Knowledge Tracer: Analyzing student {student_id}")

        # Try to load trained DKT model
        trainer = self._load_dkt()
        dkt_data = self._load_dkt_data()
        concept_map = dkt_data.get("concept_map", {})
        student_index = self._dkt_student_index(dkt_data)
        gap_features = self._load_gap_features()
        student_rows = (gap_features[gap_features["id_student"] == student_id]
                        if not gap_features.empty else pd.DataFrame())

        mastery_values: dict[str, float] = {}
        attempted: set[str] = set()
        source = "none"

        if trainer and student_id in student_index:
            seq = dkt_data["sequences"][student_index[student_id]]
            concept_ids = torch.tensor([[int(x[0]) for x in seq]], dtype=torch.long)
            responses = torch.tensor([[int(x[1]) for x in seq]], dtype=torch.long)
            lengths = torch.tensor([len(seq)], dtype=torch.long)
            state = trainer.model.get_knowledge_state(concept_ids, responses, lengths)
            predicted = state["mastery"]
            attempted = {concept_map.get(int(x[0]), f"concept_{int(x[0])}") for x in seq}
            # Only report concepts in the modules this student actually takes
            modules = {c.split("_")[0] for c in attempted}
            for i, name in concept_map.items():
                if name.split("_")[0] in modules:
                    mastery_values[name] = float(predicted[i])
            source = "dkt"
        elif not student_rows.empty:
            # Not enough history for DKT (fewer than 3 assessments): use observed averages
            for _, row in student_rows.iterrows():
                mastery_values[row["concept"]] = float(row["avg_score"])
                attempted.add(row["concept"])
            source = "observed_scores"

        concept_names = list(mastery_values.keys())
        mastery = list(mastery_values.values())

        # Build result
        mastery_by_concept = {}
        
        # Get simulated boosts
        boosts = memory.profiles.get_simulated_boosts(student_id) if memory else {}

        for concept_name, score in zip(concept_names, mastery):
            
            # Apply boost
            boost = boosts.get(concept_name, 0.0)
            final_score = min(0.99, float(score) + boost)

            level = self._get_mastery_level(final_score)
            mastery_by_concept[concept_name] = {
                "score": round(final_score, 4),
                "level": level["label"],
                "level_num": [k for k, v in MASTERY_LEVELS.items()
                              if v["label"] == level["label"]][0],
                "attempted": concept_name in attempted,
            }

        avg_mastery = np.mean([v["score"] for v in mastery_by_concept.values()]) if mastery_by_concept else 0.0
        strong = [c for c, v in mastery_by_concept.items() if v["score"] >= 0.7]
        weak = [c for c, v in mastery_by_concept.items() if v["score"] < 0.4]

        result = {
            "student_id": student_id,
            "timestamp": datetime.now().isoformat(),
            "mastery_by_concept": mastery_by_concept,
            "average_mastery": round(avg_mastery, 4),
            "strong_concepts": strong,
            "weak_concepts": weak,
            "num_concepts": len(mastery_by_concept),
            "overall_level": self._get_mastery_level(avg_mastery)["label"],
            "source": source,
        }

        logger.info(
            f"  ✓ Student {student_id}: avg mastery={avg_mastery:.2%}, "
            f"{len(strong)} strong, {len(weak)} weak concepts"
        )
        return result

    # ═══════════════════════════════════════════════════════════════════════
    #  Tool 2: Gap Detector
    # ═══════════════════════════════════════════════════════════════════════

    def gap_detector(self, student_id: int, top_k: int = 5) -> dict:
        """
        Identify the student's weakest topics with confidence scores.
        Uses XGBoost to classify and rank weak concepts by severity.
        """
        logger.info(f"🎯 Gap Detector: Finding weak topics for student {student_id}")

        detector = self._load_gap_detector()
        gap_features = self._load_gap_features()

        ranked = []
        if detector and not gap_features.empty and student_id in gap_features["id_student"].values:
            ranked = detector.rank_student_concepts(student_id, gap_features)
            gaps = [g for g in ranked if g["is_weak"]][:top_k]
        else:
            # Fallback: derive from mastery scores
            mastery = self.knowledge_tracer(student_id)
            gaps = []
            for concept, info in mastery["mastery_by_concept"].items():
                if info["score"] < 0.5:
                    severity = "Critical" if info["score"] < 0.2 else \
                               "High" if info["score"] < 0.3 else \
                               "Moderate" if info["score"] < 0.4 else "Low"
                    gaps.append({
                        "concept": concept,
                        "confidence": round(1.0 - info["score"], 3),
                        "severity": severity,
                        "priority": len(gaps) + 1,
                    })
            gaps = sorted(gaps, key=lambda x: x["confidence"], reverse=True)[:top_k]

        # Focus topics: the flagged gaps, or the highest-risk concepts when none are
        # flagged, so the recommender and planner always have something concrete.
        focus = gaps if gaps else [dict(g, watch=True) for g in ranked[:2]]

        result = {
            "student_id": student_id,
            "timestamp": datetime.now().isoformat(),
            "weak_topics": gaps,
            "focus_topics": focus,
            "all_concepts": ranked,
            "total_gaps": len(gaps),
            "critical_count": sum(1 for g in gaps if g.get("severity") == "Critical"),
            "high_count": sum(1 for g in gaps if g.get("severity") == "High"),
        }

        logger.info(
            f"  ✓ Found {len(gaps)} weak topics "
            f"({result['critical_count']} critical, {result['high_count']} high)"
        )
        return result

    # ═══════════════════════════════════════════════════════════════════════
    #  Tool 3: Resource Recommender
    # ═══════════════════════════════════════════════════════════════════════

    def resource_recommender(
        self,
        student_id: int,
        weak_concepts: Optional[list] = None,
        top_k: int = 5,
    ) -> dict:
        """
        Recommend learning resources.
        Uses the OpenRouter LLM and falls back to a curated offline catalog.
        """
        logger.info(f"📚 Resource Recommender: Finding resources for student {student_id}")

        # Get focus concepts if not provided
        if not weak_concepts:
            gap_result = self.gap_detector(student_id)
            weak_concepts = gap_result["focus_topics"]

        recommender = self._load_recommender()
        profiles = self._load_student_profiles()

        # Get student profile
        student_profile = None
        if not profiles.empty and student_id in profiles["id_student"].values:
            profile_row = profiles[profiles["id_student"] == student_id].iloc[0]
            student_profile = profile_row.to_dict()

        if recommender:
            resources = recommender.recommend(
                student_id=student_id,
                weak_concepts=weak_concepts,
                student_profile=student_profile,
                top_k=top_k,
            )
        else:
            resources = []

        result = {
            "student_id": student_id,
            "timestamp": datetime.now().isoformat(),
            "recommendations": resources[:top_k],
            "mode": getattr(recommender, "last_mode", "heuristic"),
            "total_resources": len(resources),
            "concepts_covered": list(set(r.get("concept", "") for r in resources)),
            "resource_types": {
                "videos": sum(1 for r in resources if r.get("type") == "video"),
                "pdfs": sum(1 for r in resources if r.get("type") == "pdf"),
                "practice": sum(1 for r in resources if r.get("type") == "practice"),
            },
        }

        logger.info(f"  ✓ Recommended {len(resources)} resources for {len(result['concepts_covered'])} concepts")
        return result

    # ═══════════════════════════════════════════════════════════════════════
    #  Tool 4: Study Planner
    # ═══════════════════════════════════════════════════════════════════════

    def study_planner(
        self,
        student_id: int,
        daily_hours: float = 2.0,
    ) -> dict:
        """
        Generate a personalized 7-day study plan.
        Combines rule-based scheduling with resource-aware content planning.
        GUARANTEED to produce meaningful activities even without LLM resources.
        """
        logger.info(f"📅 Study Planner: Creating 7-day plan for student {student_id}")

        from config import format_module_name

        # Get student context
        mastery = self.knowledge_tracer(student_id)
        gaps = self.gap_detector(student_id)
        # Topics for the week: focus topics first, then the next highest-risk concepts (max 3)
        topics = list(gaps["focus_topics"])
        for c in gaps.get("all_concepts", []):
            if len(topics) >= 3:
                break
            if c["concept"] not in {t["concept"] for t in topics}:
                topics.append(c)
        resources_result = self.resource_recommender(student_id, topics, top_k=12) if topics else \
            {"recommendations": []}
        resources = resources_result["recommendations"]

        # Highest-risk topic gets the most days: e.g. [A, B, A, C, A, B]
        patterns = {1: [0, 0, 0, 0, 0, 0], 2: [0, 1, 0, 1, 0, 1], 3: [0, 1, 0, 2, 0, 1]}
        pattern = patterns.get(len(topics), [])
        daily_minutes = int(daily_hours * 60)
        used: set = set()
        days = []

        for day_num in range(1, 8):
            day_plan = {"day": day_num, "day_label": f"Day {day_num}", "focus_topic": "",
                        "activities": [], "total_minutes": 0, "review": False}

            if day_num == 7:
                day_plan["focus_topic"] = "Weekly Review & Self-Assessment"
                day_plan["review"] = True
                review_minutes = max(daily_minutes, 90)
                chunk = review_minutes // 4
                day_plan["activities"] = [
                    {"time": f"{chunk} min", "activity": "Review notes from the entire week", "type": "review", "resource": None},
                    {"time": f"{chunk} min", "activity": "Practice questions on the week's highest-risk topic", "type": "practice", "resource": None},
                    {"time": f"{chunk} min", "activity": "Self-assessment quiz — test yourself without notes", "type": "assessment", "resource": None},
                    {"time": f"{chunk} min", "activity": "Reflect on progress and plan next week's goals", "type": "reflection", "resource": None},
                ]
                day_plan["total_minutes"] = review_minutes
                days.append(day_plan)
                continue

            if pattern:
                raw_topic = topics[pattern[day_num - 1]]["concept"]
            else:
                raw_topic = "General Review"
            day_plan["focus_topic"] = format_module_name(raw_topic)
            day_plan["raw_topic"] = raw_topic
            a_type = raw_topic.split("_")[1] if "_" in raw_topic else ""
            visit = sum(1 for d in days if d.get("raw_topic") == raw_topic)   # nth day on this topic

            topic_resources = [r for r in resources if r.get("concept") == raw_topic]
            time_remaining = daily_minutes

            def take(kind):
                for r in topic_resources:
                    if r.get("type") == kind and (r["id"], raw_topic) not in used:
                        used.add((r["id"], raw_topic))
                        return r
                return None

            # 1) Learn: a video or reading resource not used yet this week
            r = take("video") or take("pdf")
            if r and time_remaining >= 25:
                duration = min(int(r.get("duration_min", 30)), 45, time_remaining)
                verb = "Watch" if r["type"] == "video" else "Read"
                day_plan["activities"].append({"time": f"{duration} min", "activity": f"{verb}: {r['title']}",
                                               "type": "video" if r["type"] == "video" else "reading", "resource": r})
                time_remaining -= duration
            elif time_remaining >= 25:
                duration = min(30, time_remaining)
                day_plan["activities"].append({"time": f"{duration} min",
                                               "activity": _LEARN_ACTIVITY.get(a_type, _LEARN_ACTIVITY[""])[visit % 3],
                                               "type": "reading", "resource": None})
                time_remaining -= duration

            # 2) Practise: a practice resource, else an assessment-specific exercise
            r = take("practice")
            if r and time_remaining >= 20:
                duration = min(int(r.get("duration_min", 30)), 40, time_remaining)
                day_plan["activities"].append({"time": f"{duration} min", "activity": f"Practice: {r['title']}",
                                               "type": "practice", "resource": r})
                time_remaining -= duration
            elif time_remaining >= 20:
                duration = min(40, time_remaining)
                day_plan["activities"].append({"time": f"{duration} min",
                                               "activity": _PRACTICE_ACTIVITY.get(a_type, _PRACTICE_ACTIVITY[""])[visit % 3],
                                               "type": "practice", "resource": None})
                time_remaining -= duration

            # 3) Consolidate
            if time_remaining >= 10:
                day_plan["activities"].append({"time": f"{time_remaining} min",
                                               "activity": _REVIEW_ACTIVITY[visit % 3],
                                               "type": "review", "resource": None})
                time_remaining = 0

            day_plan["total_minutes"] = daily_minutes - time_remaining
            if day_plan["total_minutes"] > 45 and len(day_plan["activities"]) >= 2:
                day_plan["activities"].insert(1, {"time": "10 min", "activity": "🧘 Break — stretch and hydrate",
                                                  "type": "break", "resource": None})
            days.append(day_plan)

        result = {
            "student_id": student_id,
            "timestamp": datetime.now().isoformat(),
            "plan_duration": "7 days",
            "daily_hours": daily_hours,
            "days": days,
            "total_study_minutes": sum(d["total_minutes"] for d in days),
            "topics_covered": list(set(d["focus_topic"] for d in days)),
            "motivation": self._get_motivation_tip(mastery["average_mastery"]),
        }

        logger.info(
            f"  ✓ Plan created: {result['total_study_minutes']} min total, "
            f"{len(result['topics_covered'])} topics"
        )
        return result

    # ═══════════════════════════════════════════════════════════════════════
    #  Tool 5: Progress Reporter
    # ═══════════════════════════════════════════════════════════════════════

    def progress_reporter(self, student_id: int, memory=None) -> dict:
        """
        Calculate weekly improvement and generate a progress report.
        Compares current mastery with historical data from memory.
        """
        logger.info(f"📊 Progress Reporter: Generating report for student {student_id}")

        # Current mastery
        current_mastery = self.knowledge_tracer(student_id, memory=memory)

        # Historical data from memory
        improvement_data = {}
        if memory:
            trajectory = memory.history.get_mastery_trajectory(student_id)
            if len(trajectory) > 1:
                first_event = trajectory[0]["data"]
                for concept, current_info in current_mastery["mastery_by_concept"].items():
                    prev_score = first_event.get(concept, current_info["score"])
                    change = current_info["score"] - prev_score
                    improvement_data[concept] = {
                        "previous": round(prev_score, 4),
                        "current": current_info["score"],
                        "change": round(change, 4),
                        "improved": change > 0,
                    }

        # Progress summary from memory
        progress_summary = {}
        if memory:
            progress_summary = memory.history.get_progress_summary(student_id)

        # Build report
        improving = [c for c, v in improvement_data.items() if v.get("change", 0) > 0.05]
        declining = [c for c, v in improvement_data.items() if v.get("change", 0) < -0.05]
        stable = [c for c, v in improvement_data.items()
                   if -0.05 <= v.get("change", 0) <= 0.05]

        overall_improvement = np.mean([v["change"] for v in improvement_data.values()]) \
            if improvement_data else 0.0

        result = {
            "student_id": student_id,
            "timestamp": datetime.now().isoformat(),
            "current_mastery": current_mastery,
            "improvement_by_concept": improvement_data,
            "overall_improvement": round(float(overall_improvement), 4),
            "overall_improvement_pct": round(float(overall_improvement * 100), 2),
            "improving_concepts": improving,
            "declining_concepts": declining,
            "stable_concepts": stable,
            "progress_summary": progress_summary,
            "report_text": self._format_report(
                student_id, current_mastery, improvement_data,
                improving, declining, overall_improvement
            ),
        }

        logger.info(
            f"  ✓ Report: {overall_improvement:+.2%} overall, "
            f"{len(improving)} improving, {len(declining)} declining"
        )
        return result

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _get_mastery_level(score: float) -> dict:
        """Map a score (0-1) to a mastery level."""
        for level_num, level_info in MASTERY_LEVELS.items():
            low, high = level_info["range"]
            if low <= score < high:
                return level_info
        return MASTERY_LEVELS[4]  # Default to highest

    @staticmethod
    def _get_motivation_tip(avg_mastery: float) -> str:
        if avg_mastery >= 0.8:
            return "🌟 You're doing amazingly well! Keep pushing for excellence!"
        elif avg_mastery >= 0.6:
            return "💪 Great progress! Focus on your weak areas to level up."
        elif avg_mastery >= 0.4:
            return "📈 You're on the right track! Consistent daily practice will make a big difference."
        else:
            return "🚀 Every expert was once a beginner. Let's tackle one topic at a time!"

    @staticmethod
    def _format_report(
        student_id, mastery, improvements, improving, declining, overall
    ) -> str:
        from config import format_module_name

        strong = ', '.join([format_module_name(c) for c in mastery['strong_concepts'][:5]]) or 'Building up!'
        weak = ', '.join([format_module_name(c) for c in mastery['weak_concepts'][:5]]) or 'Looking good!'

        if not improvements:
            lines = [
                f"📊 **Baseline Mastery Report — Student {student_id}**",
                f"",
                f"**Overall Mastery:** {mastery['average_mastery']:.1%} ({mastery['overall_level']})",
                f"",
                f"**Strong Areas:** {strong}",
                f"**Focus Areas:** {weak}",
                f"",
                f"Complete activities to track your progress over time!"
            ]
            return "\n".join(lines)

        lines = [
            f"📊 **Weekly Progress Report — Student {student_id}**",
            f"",
            f"**Overall Mastery:** {mastery['average_mastery']:.1%} ({mastery['overall_level']})",
            f"**Overall Change:** {overall:+.1%}",
            f"",
        ]

        if improving:
            lines.append("✅ **Improving Concepts:**")
            for c in improving[:5]:
                info = improvements[c]
                lines.append(f"  • {format_module_name(c)}: {info['previous']:.0%} → {info['current']:.0%} (+{info['change']:.0%})")
            lines.append("")

        if declining:
            lines.append("⚠️ **Needs Attention:**")
            for c in declining[:5]:
                info = improvements[c]
                lines.append(f"  • {format_module_name(c)}: {info['previous']:.0%} → {info['current']:.0%} ({info['change']:.0%})")
            lines.append("")

        lines.append(f"**Strong Areas:** {strong}")
        lines.append(f"**Focus Areas:** {weak}")

        return "\n".join(lines)

# Assessment-specific activities used when the catalog has no unused resource left
_LEARN_ACTIVITY = {
    "TMA": ["Re-read the assignment brief and highlight the task words",
            "Study a model answer / tutor feedback from a previous assignment",
            "Read the module notes for the next assignment's topic"],
    "CMA": ["Review the module notes behind recent quiz questions",
            "Re-learn the definitions you got wrong in the last quiz",
            "Work through the worked examples for this unit"],
    "Exam": ["Make a one-page summary sheet of the key topics",
             "Re-watch / re-read the hardest unit of the module",
             "Map out likely exam questions from past papers"],
    "": ["Read the module notes for this topic",
         "Summarise the key ideas in your own words",
         "Review your recent feedback"],
}
_PRACTICE_ACTIVITY = {
    "TMA": ["Draft an outline for the next assignment question",
            "Write one timed practice paragraph and self-mark it against the rubric",
            "Revise a previous answer using the tutor's feedback"],
    "CMA": ["Do a timed set of practice questions",
            "Redo the questions you got wrong — without notes",
            "Make flashcards for the facts you keep forgetting"],
    "Exam": ["Answer one past-paper question under timed conditions",
             "Active-recall session: write everything you remember, then check",
             "Explain the hardest concept out loud as if teaching it"],
    "": ["Solve practice problems on this topic",
         "Test yourself with flashcards",
         "Explain the topic to someone else"],
}
_REVIEW_ACTIVITY = ["Summarise today's key points in 5 bullet notes",
                    "Spaced-repetition review of earlier notes",
                    "Write 3 questions you still have and look up the answers"]
