"""
═══════════════════════════════════════════════════════════════════════════════
  Learning Agent — tool-using orchestrator with an OpenRouter LLM
  Coordinates the 5 tools with persistent multi-layer memory. Chat answers are
  grounded in the tools' outputs; without an LLM the agent answers from the
  same data using templates.
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import re
from typing import Optional

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import PROCESSED_DIR, format_module_name
from src.agent.llm import LLMClient
from src.agent.memory import AgentMemory
from src.agent.tools import AgentTools
from src.agent.prompts import CHAT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

AGENT_VERSION = 2  # bump when the agent's interface changes (the dashboard re-imports stale code)

# Keyword intents → which tools to run before answering
INTENTS = {
    "mastery": ["mastery", "master", "level", "score", "know", "strong", "strength", "perform", "grade", "good at"],
    "gaps": ["weak", "gap", "struggl", "difficult", "hard", "risk", "focus", "worst", "problem", "fail"],
    "recommend": ["recommend", "resource", "video", "material", "read", "book", "course", "link", "learn",
                  "practice", "watch", "suggest"],
    "plan": ["plan", "schedule", "timetable", "week", "routine", "daily", "day"],
    "progress": ["progress", "report", "how am i", "how i am", "doing", "trend"],
    "tips": ["tip", "advice", "how do i", "how can i", "how should i", "improve", "better", "revise",
             "revision", "prepare", "exam", "assignment", "quiz", "essay", "motivat", "procrastinat", "focus on"],
    "profile": ["profile", "about me", "who am i", "module", "enrolled", "my course", "my data"],
}


def detect_intents(message: str) -> list[str]:
    text = message.lower()
    return [name for name, words in INTENTS.items() if any(w in text for w in words)]


class LearningAgent:
    VERSION = AGENT_VERSION
    """
    Orchestrates the 5 learning tools with multi-layer memory.

    Chat flow:
        1. Record the message in conversation memory
        2. Detect intents and run the matching tools
        3. Build a grounded context (profile, mastery, gaps, resources, plan, progress)
        4. Answer with the LLM (model fallback chain); if unavailable, answer from templates
        5. Store the answer in memory
    """

    def __init__(self):
        self.llm = LLMClient()
        self.tools = AgentTools(llm=self.llm)
        self.memory = AgentMemory()
        self.last_meta: dict = {}
        self._profiles: Optional[pd.DataFrame] = None
        logger.info(f"LearnFlow AI agent ready — LLM {'configured' if self.llm.available else 'not configured'}")

    # ═══════════════════════════════════════════════════════════════════════
    #  Core Agent Actions
    # ═══════════════════════════════════════════════════════════════════════

    def analyze_student(self, student_id: int) -> dict:
        """
        Run the full analysis pipeline for a student.
        Executes: Knowledge Tracing → Gap Detection → Recommendations → Study Plan → Progress Report
        """
        results = {"mastery": self.get_mastery(student_id)}

        results["gaps"] = self.tools.gap_detector(student_id)
        for gap in results["gaps"]["weak_topics"]:
            self.memory.history.log_event_once(student_id, "gap_detected", gap, key=gap["concept"])

        results["recommendations"] = self.tools.resource_recommender(
            student_id, results["gaps"]["focus_topics"]
        )
        self._remember_recommendations(student_id, results["recommendations"])

        results["study_plan"] = self.tools.study_planner(student_id)
        results["progress"] = self.tools.progress_reporter(student_id, self.memory)
        results["summary"] = self._generate_analysis_summary(student_id, results)
        logger.info(f"  ✓ Full analysis complete for student {student_id}")
        return results

    def chat(self, student_id: int, message: str) -> str:
        """
        Handle a conversational message from a student.
        Uses conversation memory and tool results for personalised, data-backed answers.
        """
        self.memory.conversation.add_message(student_id, "user", message)
        intents = detect_intents(message)
        data = self._gather(student_id, intents)

        response, mode = None, "offline"
        if self.llm.available:
            response = self._llm_chat(student_id, message, data)
            if response:
                mode = "llm"
        if not response:
            response = self._rule_based_chat(student_id, message, intents, data)

        self.last_meta = {"mode": mode, "model": self.llm.last_model if mode == "llm" else None,
                          "tools": sorted(k for k in data if k != "profile"),
                          "llm_error": self.llm.last_error if mode != "llm" else None}
        self.memory.conversation.add_message(student_id, "assistant", response)
        return response

    def get_mastery(self, student_id: int) -> dict:
        """Mastery check (DKT); logs a history event only when mastery changed."""
        result = self.tools.knowledge_tracer(student_id, memory=self.memory)
        scores = {c: v["score"] for c, v in result["mastery_by_concept"].items()}
        self.memory.profiles.update_mastery(student_id, scores)
        self._log_mastery(student_id, result)
        return result

    def get_gaps(self, student_id: int) -> dict:
        return self.tools.gap_detector(student_id)

    def get_recommendations(self, student_id: int, top_k: int = 5) -> dict:
        gaps = self.tools.gap_detector(student_id)
        recs = self.tools.resource_recommender(student_id, gaps["focus_topics"], top_k=top_k)
        self._remember_recommendations(student_id, recs)
        return recs

    def get_study_plan(self, student_id: int, daily_hours: float = 2.0) -> dict:
        return self.tools.study_planner(student_id, daily_hours)

    def get_progress(self, student_id: int) -> dict:
        return self.tools.progress_reporter(student_id, self.memory)

    def simulate_study_plan_completion(self, student_id: int, plan: dict) -> dict:
        """Simulate completing a study plan to demonstrate progress tracking."""
        for day in plan.get("days", []):
            topic = day.get("raw_topic")
            if topic and "Review" not in topic:
                self.memory.profiles.add_simulated_boost(student_id, topic, 0.15)

        self.memory.history.log_event(student_id, "study_session", {"duration_days": 7})
        new_mastery = self.tools.knowledge_tracer(student_id, memory=self.memory)
        self._log_mastery(student_id, new_mastery)
        return new_mastery

    def get_student_ids(self) -> list[int]:
        """Students that have assessment data (so every page has something to show)."""
        try:
            gap = self.tools._load_gap_features()
            profiles = self._load_profiles()
            ids = set(gap["id_student"].astype(int)) if not gap.empty else set()
            if not profiles.empty:
                ids = ids & set(profiles["id_student"].astype(int)) if ids else set(profiles["id_student"].astype(int))
            return sorted(ids)
        except Exception:
            return []

    def get_profile(self, student_id: int) -> dict:
        profiles = self._load_profiles()
        rows = profiles[profiles["id_student"] == student_id] if not profiles.empty else profiles
        if rows is None or rows.empty:
            return {}
        row = rows.iloc[0].to_dict()
        row["modules"] = sorted({f"{r['code_module']} ({r['code_presentation']})" for _, r in rows.iterrows()
                                 if "code_module" in rows.columns})
        return row

    # ═══════════════════════════════════════════════════════════════════════
    #  Chat helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _gather(self, student_id: int, intents: list[str]) -> dict:
        """Run the tools needed for this message."""
        data = {"profile": self.get_profile(student_id)}
        try:
            data["mastery"] = self.tools.knowledge_tracer(student_id, memory=self.memory)
            data["gaps"] = self.tools.gap_detector(student_id)
            if "recommend" in intents or "plan" in intents:
                data["recommendations"] = self.tools.resource_recommender(
                    student_id, data["gaps"]["focus_topics"])
                self._remember_recommendations(student_id, data["recommendations"])
            if "plan" in intents:
                data["plan"] = self.tools.study_planner(student_id)
            if "progress" in intents:
                data["progress"] = self.tools.progress_reporter(student_id, self.memory)
        except Exception as e:  # keep chatting even if one tool fails
            logger.error(f"Tool execution failed: {e}")
        return data

    def _context_text(self, student_id: int, data: dict) -> str:
        """Compact, factual context for the LLM (no raw JSON)."""
        lines = [f"Student ID: {student_id}"]
        p = data.get("profile") or {}
        if p:
            lines.append(
                f"Modules: {', '.join(p.get('modules', [])) or 'unknown'}; education: {p.get('highest_education', '?')}; "
                f"age band: {p.get('age_band', '?')}; engagement: {p.get('engagement_level', '?')}; "
                f"VLE clicks: {int(p.get('total_clicks', 0) or 0):,}; average assessment score: "
                f"{p.get('avg_score', float('nan')):.1f}/100; final result: {p.get('final_result', '?')}")
        m = data.get("mastery")
        if m and m["mastery_by_concept"]:
            items = sorted(m["mastery_by_concept"].items(), key=lambda kv: kv[1]["score"])
            lines.append(f"DKT mastery (probability of passing the next assessment of each concept); "
                         f"overall {m['average_mastery']:.0%} ({m['overall_level']}):")
            lines += [f"  - {format_module_name(c)}: {v['score']:.0%} ({v['level']})"
                      + ("" if v.get("attempted", True) else ", not attempted yet") for c, v in items]
        g = data.get("gaps")
        if g:
            if g["weak_topics"]:
                lines.append("Gap detector (XGBoost) — concepts at risk of scoring below 50% next time:")
                lines += [f"  - {format_module_name(x['concept'])}: risk {x['confidence']:.0%}, {x['severity']}"
                          for x in g["weak_topics"]]
            elif g.get("focus_topics"):
                lines.append("Gap detector: no concept is flagged as at-risk. Highest-risk concepts to watch: "
                             + ", ".join(f"{format_module_name(x['concept'])} (risk {x['confidence']:.0%})"
                                         for x in g["focus_topics"]))
        r = data.get("recommendations")
        if r and r["recommendations"]:
            lines.append("Recommended resources (use ONLY these links):")
            lines += [f"  - [{x['title']}]({x['url']}) — {x['type']}, {x['difficulty']}, ~{x['duration_min']} min, "
                      f"for {format_module_name(x['concept'])}" for x in r["recommendations"]]
        plan = data.get("plan")
        if plan:
            lines.append(f"7-day plan ({plan['total_study_minutes']} min total):")
            lines += [f"  - {d['day_label']}: {d['focus_topic']} ({d['total_minutes']} min)" for d in plan["days"]]
        prog = data.get("progress")
        if prog:
            lines.append(f"Progress: overall change {prog['overall_improvement_pct']:+.1f} points; "
                         f"improving {len(prog['improving_concepts'])}, declining {len(prog['declining_concepts'])}.")
        past = self.memory.recommendations.get_past_recommendations(student_id, last_n=3)
        if past:
            lines.append("Previously recommended concepts: " + ", ".join(
                format_module_name(x.get("concept", "")) for x in past))
        return "\n".join(lines)

    def _llm_chat(self, student_id: int, message: str, data: dict) -> Optional[str]:
        system = CHAT_SYSTEM_PROMPT.format(student_context=self._context_text(student_id, data))
        history = self.memory.conversation.get_context_messages(student_id)[:-1][-8:]
        messages = [{"role": "system", "content": system}]
        messages += [{"role": m["role"], "content": m["content"][:1500]} for m in history]
        messages.append({"role": "user", "content": message})
        return self.llm.complete(messages, time_budget=50)

    def _rule_based_chat(self, student_id: int, message: str, intents: list[str], data: dict) -> str:
        """Data-backed answer without an LLM."""
        sections = []
        m, g = data.get("mastery"), data.get("gaps")
        if "mastery" in intents and m:
            strong = ", ".join(format_module_name(c) for c in m["strong_concepts"][:3]) or "none above 70% yet"
            weak = ", ".join(format_module_name(c) for c in m["weak_concepts"][:3]) or "none below 40%"
            sections.append(
                f"📊 **Overall mastery:** {m['average_mastery']:.0%} ({m['overall_level']})\n\n"
                f"💪 **Strongest:** {strong}\n\n📝 **Needs work:** {weak}")
        if ("gaps" in intents or "mastery" in intents) and g:
            if g["weak_topics"]:
                items = "\n".join(f"{i}. **{format_module_name(x['concept'])}** — {x['severity']} "
                                  f"(risk {x['confidence']:.0%})" for i, x in enumerate(g["weak_topics"], 1))
                sections.append(f"🎯 **At-risk topics** (chance of scoring below 50% next time):\n\n{items}")
            elif g.get("focus_topics"):
                items = ", ".join(f"**{format_module_name(x['concept'])}** ({x['confidence']:.0%})"
                                  for x in g["focus_topics"])
                sections.append(f"✅ No topic is flagged as at-risk. Keep an eye on {items}.")
        if "recommend" in intents and data.get("recommendations"):
            items = "\n".join(f"- [{x['title']}]({x['url']}) — {x['type']}, ~{x['duration_min']} min"
                              + (f"  \n  _{x['reason']}_" if x.get("reason") else "")
                              for x in data["recommendations"]["recommendations"])
            sections.append(f"📚 **Recommended resources:**\n\n{items}")
        if "plan" in intents and data.get("plan"):
            plan = data["plan"]
            days = "\n".join(f"- **{d['day_label']}:** {d['focus_topic']} ({d['total_minutes']} min)"
                             for d in plan["days"])
            sections.append(f"📅 **Your 7-day plan** ({plan['total_study_minutes']} min total):\n\n{days}\n\n"
                            f"{plan['motivation']}")
        if "progress" in intents and data.get("progress"):
            sections.append(data["progress"]["report_text"])
        if "tips" in intents:
            sections.append(self._study_tips(message, g))
        if "profile" in intents and data.get("profile"):
            p = data["profile"]
            sections.append(f"👤 **Your profile:** modules {', '.join(p.get('modules', []))}; "
                            f"average score {p.get('avg_score', 0):.1f}/100; engagement {p.get('engagement_level', '?')}.")

        if sections:
            return "\n\n".join(sections)
        return (
            "I can answer questions about your learning data:\n\n"
            "- 📊 *\"What's my mastery level?\"*\n- 🎯 *\"Which topics am I weakest in?\"*\n"
            "- 📚 *\"Recommend resources\"*\n- 📅 *\"Make me a study plan\"*\n- 📈 *\"How am I progressing?\"*\n\n"
            "Open-ended questions (explanations, study advice) need the AI model — add an "
            "`OPENROUTER_API_KEY` in **Settings** or the app's secrets to enable it."
        )

    @staticmethod
    def _study_tips(message: str, gaps: Optional[dict]) -> str:
        """Evidence-based study advice, targeted at the assessment type in question."""
        text = message.lower()
        focus = (gaps or {}).get("focus_topics") or []
        a_type = next((t for t, words in {"Exam": ["exam", "revis"], "TMA": ["assignment", "essay", "tma"],
                                          "CMA": ["quiz", "cma", "test"]}.items()
                       if any(w in text for w in words)), None)
        if not a_type and focus:
            a_type = focus[0]["concept"].split("_")[-1]
        tips = {
            "Exam": ["**Active recall** — close your notes and write everything you remember, then check gaps.",
                     "**Past papers under timed conditions**, marked honestly against the scheme.",
                     "**Spaced repetition** — revisit each topic after 1, 3 and 7 days instead of cramming.",
                     "Make a **one-page summary** per topic and test yourself on it."],
            "TMA": ["Start by **decoding the brief**: underline task words (discuss, evaluate, compare).",
                    "**Outline first** — one paragraph per point, each backed by evidence.",
                    "Leave time to **edit against the marking criteria** and fix referencing.",
                    "Use your **tutor feedback** from the previous assignment as a checklist."],
            "CMA": ["Do **short, frequent practice quizzes** and redo every question you got wrong.",
                    "Turn mistakes into **flashcards** and review them with spaced repetition.",
                    "Read each option fully — eliminate clearly wrong answers first."],
        }.get(a_type, ["Study in **25–45 minute focused blocks** with short breaks.",
                       "Use **active recall and spaced repetition** rather than re-reading.",
                       "Start each session with the **highest-risk topic** while you are fresh."])
        header = {"Exam": "📝 **Exam preparation tips**", "TMA": "✍️ **Assignment (TMA) tips**",
                  "CMA": "✅ **Quiz (CMA) tips**"}.get(a_type, "💡 **Study tips**")
        out = header + "\n\n" + "\n".join(f"- {t}" for t in tips)
        if focus:
            out += ("\n\nStart with **" + format_module_name(focus[0]["concept"]) +
                    f"** — your highest-risk concept ({focus[0]['confidence']:.0%} risk of scoring below 50%).")
        return out

    # ═══════════════════════════════════════════════════════════════════════
    #  Internals
    # ═══════════════════════════════════════════════════════════════════════

    def _load_profiles(self) -> pd.DataFrame:
        if self._profiles is None:
            path = PROCESSED_DIR / "student_profiles.csv"
            self._profiles = pd.read_csv(path) if path.exists() else pd.DataFrame()
        return self._profiles

    def _log_mastery(self, student_id: int, mastery: dict):
        data = {"avg_mastery": mastery["average_mastery"],
                **{c: v["score"] for c, v in mastery["mastery_by_concept"].items()}}
        last = self.memory.history.get_mastery_trajectory(student_id)
        if not last or last[-1]["data"] != data:
            self.memory.history.log_event(student_id, "mastery_update", data)

    def _remember_recommendations(self, student_id: int, recs: dict):
        past = self.memory.recommendations.get_past_recommendations(student_id, last_n=20)
        for concept in recs.get("concepts_covered", []):
            resources = [r for r in recs["recommendations"] if r.get("concept") == concept]
            ids = sorted(r.get("id", "") for r in resources)
            already = any(p.get("concept") == concept and
                          sorted(x.get("id", "") for x in p.get("resources", [])) == ids for p in past)
            if resources and not already:
                self.memory.recommendations.add_recommendation(
                    student_id, concept, resources, reason="Focus topic from gap detection")

    @staticmethod
    def _generate_analysis_summary(student_id: int, results: dict) -> str:
        mastery, gaps, recs = results["mastery"], results["gaps"], results["recommendations"]
        focus = ", ".join(format_module_name(g["concept"]) for g in gaps["focus_topics"]) or "—"
        return (
            f"**Analysis Summary for Student {student_id}**\n\n"
            f"📊 **Overall Mastery:** {mastery['average_mastery']:.1%} ({mastery['overall_level']})\n\n"
            f"🎯 **At-risk topics:** {gaps['total_gaps']} ({gaps['critical_count']} critical) · focus: {focus}\n\n"
            f"📚 **Resources:** {recs['total_resources']} recommended "
            f"across {len(recs['concepts_covered'])} concepts\n\n"
            f"📅 **Study Plan:** 7-day plan ready ({results['study_plan']['total_study_minutes']} min total)"
        )
