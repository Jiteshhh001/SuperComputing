"""
═══════════════════════════════════════════════════════════════════════════════
  Agent Tools — 5 LangChain @tool definitions for the learning agent
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
from config import MODELS_DIR, PROCESSED_DIR, MASTERY_LEVELS

logger = logging.getLogger(__name__)


class AgentTools:
    """
    Wrapper class that provides the 5 agent tools.
    Each tool loads models lazily and caches them.
    """

    def __init__(self):
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
                from src.models.recommender import DynamicLLMRecommender
                self._recommender = DynamicLLMRecommender()
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

        # Check if student has sequences
        student_ids = dkt_data.get("student_ids", [])

        if trainer and student_id in student_ids:
            idx = student_ids.index(student_id)
            seq = dkt_data["sequences"][idx]

            # Prepare tensors
            concept_ids = torch.tensor([[s[0] for s in seq]], dtype=torch.long)
            responses = torch.tensor([[s[1] for s in seq]], dtype=torch.long)
            lengths = torch.tensor([len(seq)], dtype=torch.long)

            # Get knowledge state
            state = trainer.model.get_knowledge_state(concept_ids, responses, lengths)
            mastery = state["mastery"]
        else:
            # Fallback: derive mastery from gap features
            gap_features = self._load_gap_features()
            if not gap_features.empty and student_id in gap_features["id_student"].values:
                student_data = gap_features[gap_features["id_student"] == student_id]
                mastery_dict = {}
                for _, row in student_data.iterrows():
                    mastery_dict[row["concept"]] = row["avg_score"]

                # Build mastery array
                mastery = np.array([mastery_dict.get(concept_map.get(i, f"concept_{i}"), 0.5)
                                     for i in range(len(concept_map))])
            else:
                mastery = np.random.uniform(0.3, 0.8, max(len(concept_map), 5))

        # Build result
        mastery_by_concept = {}
        
        # Get simulated boosts
        boosts = memory.profiles.get_simulated_boosts(student_id) if memory else {}

        for i, score in enumerate(mastery):
            concept_name = concept_map.get(i, f"concept_{i}")
            
            # Apply boost
            boost = boosts.get(concept_name, 0.0)
            final_score = min(0.99, float(score) + boost)

            level = self._get_mastery_level(final_score)
            mastery_by_concept[concept_name] = {
                "score": round(final_score, 4),
                "level": level["label"],
                "level_num": [k for k, v in MASTERY_LEVELS.items()
                              if v["label"] == level["label"]][0],
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

        if detector and not gap_features.empty and student_id in gap_features["id_student"].values:
            gaps = detector.get_student_gaps(student_id, gap_features, top_k=top_k)
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

        result = {
            "student_id": student_id,
            "timestamp": datetime.now().isoformat(),
            "weak_topics": gaps,
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
        Recommend learning resources using the hybrid engine.
        Combines collaborative filtering, content-based matching, and LLM ranking.
        """
        logger.info(f"📚 Resource Recommender: Finding resources for student {student_id}")

        # Get weak concepts if not provided
        if not weak_concepts:
            gap_result = self.gap_detector(student_id)
            weak_concepts = gap_result["weak_topics"]

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
        resources_result = self.resource_recommender(student_id, gaps["weak_topics"])
        resources = resources_result["recommendations"]

        # Rule-based plan structure
        days = []
        weak_topics = gaps["weak_topics"]
        daily_minutes = int(daily_hours * 60)

        for day_num in range(1, 8):
            day_plan = {
                "day": day_num,
                "day_label": f"Day {day_num}",
                "focus_topic": "",
                "activities": [],
                "total_minutes": 0,
                "review": False,
            }

            if day_num == 7:
                # Day 7: Review & Self-Assessment
                day_plan["focus_topic"] = "Weekly Review & Self-Assessment"
                day_plan["review"] = True
                review_minutes = max(daily_minutes, 90)
                chunk = review_minutes // 4
                day_plan["activities"] = [
                    {"time": f"{chunk} min", "activity": "Review notes from the entire week",
                     "type": "review", "resource": None},
                    {"time": f"{chunk} min", "activity": "Practice problems on weakest topics",
                     "type": "practice", "resource": None},
                    {"time": f"{chunk} min", "activity": "Self-assessment quiz — test yourself without notes",
                     "type": "assessment", "resource": None},
                    {"time": f"{chunk} min", "activity": "Reflect on progress and plan next week's goals",
                     "type": "reflection", "resource": None},
                ]
                day_plan["total_minutes"] = review_minutes
            else:
                # Assign a weak topic to each day (cycle if more days than topics)
                raw_topic = ""
                if weak_topics:
                    topic_idx = (day_num - 1) % len(weak_topics)
                    topic = weak_topics[topic_idx]
                    raw_topic = topic.get("concept", f"Topic {day_num}")
                else:
                    raw_topic = f"General Review"

                # Store human-readable name for display
                day_plan["focus_topic"] = format_module_name(raw_topic)
                day_plan["raw_topic"] = raw_topic

                # Find resources matching this topic
                topic_resources = [r for r in resources if r.get("concept") == raw_topic]

                # ── Build activities ──
                time_remaining = daily_minutes

                # Video session
                videos = [r for r in topic_resources if r.get("type") == "video"]
                if videos and time_remaining >= 30:
                    v = videos[0]
                    duration = min(v.get("duration_min", 30), 45)
                    day_plan["activities"].append({
                        "time": f"{duration} min",
                        "activity": f"Watch: {v.get('title', 'Video lesson')}",
                        "type": "video",
                        "resource": v,
                    })
                    time_remaining -= duration
                elif time_remaining >= 30:
                    # No video resource — generate a practical default
                    duration = min(30, time_remaining)
                    day_plan["activities"].append({
                        "time": f"{duration} min",
                        "activity": f"Watch a lecture or tutorial on {day_plan['focus_topic']}",
                        "type": "video",
                        "resource": None,
                    })
                    time_remaining -= duration

                # Reading / PDF session
                pdfs = [r for r in topic_resources if r.get("type") == "pdf"]
                if pdfs and time_remaining >= 25:
                    p = pdfs[0]
                    duration = min(p.get("duration_min", 30), 40)
                    day_plan["activities"].append({
                        "time": f"{duration} min",
                        "activity": f"Read: {p.get('title', 'Study material')}",
                        "type": "reading",
                        "resource": p,
                    })
                    time_remaining -= duration
                elif time_remaining >= 25:
                    duration = min(30, time_remaining)
                    day_plan["activities"].append({
                        "time": f"{duration} min",
                        "activity": f"Read textbook chapter on {day_plan['focus_topic']}",
                        "type": "reading",
                        "resource": None,
                    })
                    time_remaining -= duration

                # Practice session
                practice = [r for r in topic_resources if r.get("type") == "practice"]
                if practice and time_remaining >= 20:
                    pr = practice[0]
                    duration = min(pr.get("duration_min", 45), time_remaining)
                    day_plan["activities"].append({
                        "time": f"{duration} min",
                        "activity": f"Practice: {pr.get('title', 'Exercises')}",
                        "type": "practice",
                        "resource": pr,
                    })
                    time_remaining -= duration
                elif time_remaining >= 20:
                    duration = min(30, time_remaining)
                    day_plan["activities"].append({
                        "time": f"{duration} min",
                        "activity": f"Solve practice problems on {day_plan['focus_topic']}",
                        "type": "practice",
                        "resource": None,
                    })
                    time_remaining -= duration

                # Fill remaining time with review
                if time_remaining >= 10:
                    day_plan["activities"].append({
                        "time": f"{time_remaining} min",
                        "activity": "Review notes and summarize key concepts",
                        "type": "review",
                        "resource": None,
                    })
                    time_remaining = 0

                day_plan["total_minutes"] = daily_minutes - time_remaining

                # Add break reminder for long sessions
                if day_plan["total_minutes"] > 45 and len(day_plan["activities"]) >= 2:
                    day_plan["activities"].insert(
                        len(day_plan["activities"]) // 2,
                        {"time": "10 min", "activity": "🧘 Break — Stretch and hydrate",
                         "type": "break", "resource": None}
                    )

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
