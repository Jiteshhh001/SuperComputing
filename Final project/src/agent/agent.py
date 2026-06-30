"""
═══════════════════════════════════════════════════════════════════════════════
  Learning Agent — LangChain-based orchestrator with OpenRouter LLM
  Coordinates all tools with persistent multi-layer memory.
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import logging
from datetime import datetime
from typing import Optional

import httpx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, format_module_name
from src.agent.memory import AgentMemory
from src.agent.tools import AgentTools
from src.agent.prompts import SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT, STUDY_PLAN_TEMPLATE

logger = logging.getLogger(__name__)


class LearningAgent:
    """
    Orchestrates the 5 learning tools with multi-layer memory.
    Uses OpenRouter for LLM reasoning and response generation.

    Architecture:
        1. Receives user query + student context
        2. Determines which tools to invoke (via LLM or rule-based)
        3. Executes tools and collects results
        4. Generates natural language response via LLM
        5. Updates memory with new insights
    """

    def __init__(self):
        self.tools = AgentTools()
        self.memory = AgentMemory()
        self.api_key = OPENROUTER_API_KEY
        self.base_url = OPENROUTER_BASE_URL
        self.model = LLM_MODEL

        logger.info("╔══════════════════════════════════════════════════════════╗")
        logger.info("║         LearnFlow AI Agent — Initialized                ║")
        logger.info(f"║  Model: {self.model:<44} ║")
        logger.info(f"║  LLM Available: {'Yes' if self.api_key else 'No (rule-based mode)':<36} ║")
        logger.info("╚══════════════════════════════════════════════════════════╝")

    # ═══════════════════════════════════════════════════════════════════════
    #  Core Agent Actions
    # ═══════════════════════════════════════════════════════════════════════

    def analyze_student(self, student_id: int) -> dict:
        """
        Run the full analysis pipeline for a student.
        Executes: Knowledge Tracing → Gap Detection → Recommendations → Study Plan → Progress Report
        """
        logger.info(f"{'═' * 60}")
        logger.info(f"  Full Analysis: Student {student_id}")
        logger.info(f"{'═' * 60}")

        results = {}

        # Step 1: Knowledge Tracing
        results["mastery"] = self.tools.knowledge_tracer(student_id, memory=self.memory)

        # Update memory with mastery
        self.memory.profiles.update_mastery(
            student_id,
            {c: v["score"] for c, v in results["mastery"]["mastery_by_concept"].items()}
        )
        self.memory.history.log_event(student_id, "mastery_update", {
            "avg_mastery": results["mastery"]["average_mastery"],
            **{c: v["score"] for c, v in results["mastery"]["mastery_by_concept"].items()}
        })

        # Step 2: Gap Detection
        results["gaps"] = self.tools.gap_detector(student_id)

        # Log gaps to memory
        for gap in results["gaps"]["weak_topics"]:
            self.memory.history.log_event(student_id, "gap_detected", gap)

        # Step 3: Resource Recommendations
        results["recommendations"] = self.tools.resource_recommender(
            student_id, results["gaps"]["weak_topics"]
        )

        # Save recommendations to memory
        for concept in results["recommendations"]["concepts_covered"]:
            concept_resources = [r for r in results["recommendations"]["recommendations"]
                                  if r.get("concept") == concept]
            self.memory.recommendations.add_recommendation(
                student_id, concept, concept_resources,
                reason=f"Gap detected with severity-based recommendation"
            )

        # Step 4: Study Plan
        results["study_plan"] = self.tools.study_planner(student_id)

        # Step 5: Progress Report
        results["progress"] = self.tools.progress_reporter(student_id, self.memory)

        # Generate summary via LLM
        results["summary"] = self._generate_analysis_summary(student_id, results)

        logger.info(f"  ✓ Full analysis complete for student {student_id}")
        return results

    def chat(self, student_id: int, message: str) -> str:
        """
        Handle a conversational message from a student.
        Uses conversation memory and student context for personalized responses.
        """
        # Record user message
        self.memory.conversation.add_message(student_id, "user", message)

        # Build context
        student_context = self.memory.get_full_context(student_id)
        conversation_history = self.memory.conversation.get_context_messages(student_id)

        # Determine if tool use is needed
        tool_results = self._maybe_use_tools(student_id, message)

        # Generate response
        if self.api_key:
            response = self._llm_chat(student_id, message, student_context,
                                       conversation_history, tool_results)
        else:
            response = self._rule_based_chat(student_id, message, tool_results)

        # Record assistant response
        self.memory.conversation.add_message(student_id, "assistant", response)

        return response

    def get_mastery(self, student_id: int) -> dict:
        """Quick mastery check without full analysis."""
        result = self.tools.knowledge_tracer(student_id, memory=self.memory)
        self.memory.profiles.update_mastery(
            student_id,
            {c: v["score"] for c, v in result["mastery_by_concept"].items()}
        )
        return result

    def get_gaps(self, student_id: int) -> dict:
        """Quick gap detection."""
        return self.tools.gap_detector(student_id)

    def get_recommendations(self, student_id: int) -> dict:
        """Quick resource recommendations."""
        gaps = self.tools.gap_detector(student_id)
        return self.tools.resource_recommender(student_id, gaps["weak_topics"])

    def get_study_plan(self, student_id: int, daily_hours: float = 2.0) -> dict:
        """Generate study plan."""
        return self.tools.study_planner(student_id, daily_hours)

    def get_progress(self, student_id: int) -> dict:
        """Generate progress report."""
        return self.tools.progress_reporter(student_id, self.memory)

    def simulate_study_plan_completion(self, student_id: int, plan: dict) -> dict:
        """Simulate completing a study plan to demonstrate progress tracking."""
        for day in plan.get("days", []):
            topic = day.get("raw_topic")
            if topic and "Review" not in topic:
                self.memory.profiles.add_simulated_boost(student_id, topic, 0.15)
        
        self.memory.history.log_event(student_id, "study_session", {"duration_days": 7})
        new_mastery = self.tools.knowledge_tracer(student_id, memory=self.memory)
        self.memory.history.log_event(student_id, "mastery_update", {
            "avg_mastery": new_mastery["average_mastery"],
            **{c: v["score"] for c, v in new_mastery["mastery_by_concept"].items()}
        })
        return new_mastery

    # ═══════════════════════════════════════════════════════════════════════
    #  LLM Integration
    # ═══════════════════════════════════════════════════════════════════════

    def _llm_chat(
        self,
        student_id: int,
        message: str,
        student_context: str,
        conversation_history: list[dict],
        tool_results: Optional[dict] = None,
    ) -> str:
        """Generate a response using OpenRouter LLM."""
        system_prompt = CHAT_SYSTEM_PROMPT.format(student_context=student_context)

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (last 10 messages)
        for msg in conversation_history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Add tool results as context
        if tool_results:
            tool_context = f"\n[Tool Results]\n{json.dumps(tool_results, indent=2, default=str)[:2000]}"
            messages.append({
                "role": "system",
                "content": f"The following tool results are available for your response:{tool_context}"
            })

        # Add current message (it's already in history, but ensure it's the last)
        if not conversation_history or conversation_history[-1]["content"] != message:
            messages.append({"role": "user", "content": message})

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://learnflow-ai.app",
                    "X-Title": "LearnFlow AI",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": LLM_TEMPERATURE,
                    "max_tokens": LLM_MAX_TOKENS,
                },
                timeout=60,
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logger.warning(f"LLM API error {response.status_code}: {response.text[:200]}")
                return self._rule_based_chat(student_id, message, tool_results)

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._rule_based_chat(student_id, message, tool_results)

    def _rule_based_chat(
        self,
        student_id: int,
        message: str,
        tool_results: Optional[dict] = None,
    ) -> str:
        """Fallback rule-based response when LLM is unavailable."""
        message_lower = message.lower()

        if any(w in message_lower for w in ["mastery", "know", "level", "score"]):
            mastery = self.tools.knowledge_tracer(student_id, memory=self.memory)
            strong = ', '.join([format_module_name(c) for c in mastery['strong_concepts'][:3]]) or 'Building up!'
            weak = ', '.join([format_module_name(c) for c in mastery['weak_concepts'][:3]]) or 'Looking good!'
            return (
                f"Here's your current mastery overview:\n\n"
                f"📊 **Overall Mastery:** {mastery['average_mastery']:.1%} ({mastery['overall_level']})\n"
                f"💪 **Strong areas:** {strong}\n"
                f"📝 **Areas to improve:** {weak}\n\n"
                f"Would you like me to create a study plan for your weak areas?"
            )

        elif any(w in message_lower for w in ["weak", "gap", "struggle", "difficult", "hard"]):
            gaps = self.tools.gap_detector(student_id)
            topics = "\n".join([
                f"  {i+1}. **{format_module_name(g['concept'])}** — {g['severity']} severity (confidence: {g['confidence']:.0%})"
                for i, g in enumerate(gaps["weak_topics"][:5])
            ])
            return (
                f"I've identified these areas that need attention:\n\n{topics}\n\n"
                f"Want me to recommend specific resources for any of these topics?"
            )

        elif any(w in message_lower for w in ["recommend", "resource", "video", "study material"]):
            recs = self.tools.resource_recommender(student_id)
            resources = "\n".join([
                f"  📎 **[{r.get('title', 'Resource')}]({r.get('url', '#')})** [{r.get('type', '').upper()}] — "
                f"{r.get('duration_min', '?')} min ({r.get('difficulty', '')})"
                for r in recs["recommendations"][:5]
            ])
            return (
                f"Here are my top recommendations for you:\n\n{resources}\n\n"
                f"These resources were selected based on your specific weak areas and learning style."
            )

        elif any(w in message_lower for w in ["plan", "schedule", "study plan", "week"]):
            plan = self.tools.study_planner(student_id)
            days_text = "\n".join([
                f"  **{d['day_label']}:** {d['focus_topic']} ({d['total_minutes']} min)"
                for d in plan["days"]
            ])
            activities_preview = ""
            for d in plan["days"][:3]:
                acts = ', '.join([a['activity'] for a in d.get('activities', [])[:2]])
                if acts:
                    activities_preview += f"\n  __{d['day_label']}:__ {acts}"
            return (
                f"Here's your personalized 7-day study plan:\n\n{days_text}\n\n"
                f"**Sample Activities:**{activities_preview}\n\n"
                f"📝 Total study time: {plan['total_study_minutes']} minutes\n"
                f"{plan['motivation']}"
            )

        elif any(w in message_lower for w in ["progress", "improve", "report", "how am i"]):
            progress = self.tools.progress_reporter(student_id, self.memory)
            return progress["report_text"]

        else:
            return (
                f"Hi! I'm here to help with your learning journey. I can:\n\n"
                f"📊 **Check your mastery levels** — Tell me 'show my mastery'\n"
                f"🎯 **Find weak areas** — Ask 'what are my weak topics?'\n"
                f"📚 **Recommend resources** — Say 'recommend study materials'\n"
                f"📅 **Create study plan** — Ask 'make me a study plan'\n"
                f"📈 **Show progress** — Ask 'how am I doing?'\n\n"
                f"What would you like to know?"
            )

    def _maybe_use_tools(self, student_id: int, message: str) -> Optional[dict]:
        """Determine if the message requires tool usage and execute if needed."""
        message_lower = message.lower()

        triggers = {
            "mastery": lambda: self.tools.knowledge_tracer(student_id, memory=self.memory),
            "gap": lambda: self.tools.gap_detector(student_id),
            "weak": lambda: self.tools.gap_detector(student_id),
            "recommend": lambda: self.tools.resource_recommender(student_id),
            "plan": lambda: self.tools.study_planner(student_id),
            "progress": lambda: self.tools.progress_reporter(student_id, self.memory),
            "improve": lambda: self.tools.progress_reporter(student_id, self.memory),
        }

        for keyword, tool_fn in triggers.items():
            if keyword in message_lower:
                try:
                    return tool_fn()
                except Exception as e:
                    logger.error(f"Tool execution failed for '{keyword}': {e}")
                    return None

        return None

    def _generate_analysis_summary(self, student_id: int, results: dict) -> str:
        """Generate a natural language summary of the full analysis."""
        mastery = results["mastery"]
        gaps = results["gaps"]
        recs = results["recommendations"]

        summary = (
            f"**Analysis Summary for Student {student_id}**\n\n"
            f"📊 **Overall Mastery:** {mastery['average_mastery']:.1%} ({mastery['overall_level']})\n"
            f"🎯 **Knowledge Gaps:** {gaps['total_gaps']} weak topics identified "
            f"({gaps['critical_count']} critical)\n"
            f"📚 **Resources:** {recs['total_resources']} resources recommended "
            f"across {len(recs['concepts_covered'])} concepts\n"
            f"📅 **Study Plan:** 7-day plan ready "
            f"({results['study_plan']['total_study_minutes']} min total)\n"
        )

        return summary

    def get_student_ids(self) -> list[int]:
        """Get list of available student IDs from processed data."""
        try:
            import pandas as pd
            profiles_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "student_profiles.csv"
            if profiles_path.exists():
                df = pd.read_csv(profiles_path)
                return sorted(df["id_student"].unique().tolist())
        except Exception:
            pass
        return []
