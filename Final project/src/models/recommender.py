"""
═══════════════════════════════════════════════════════════════════════════════
  Dynamic LLM Recommender with Robust Offline Fallback
  Tries OpenRouter first, falls back to a curated real-world catalog.
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import logging
from typing import Optional
from pathlib import Path
import httpx

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  High-Quality Offline Fallback Catalog
#  Real URLs, real resources — guaranteed to work without any API.
# ═══════════════════════════════════════════════════════════════════════════

FALLBACK_CATALOG = {
    "AAA_TMA": {
        "topic": "Social Sciences — Assignments",
        "resources": [
            {"id": "f001", "type": "video", "title": "Crash Course Sociology: Socialization",
             "url": "https://www.youtube.com/watch?v=K-IvvuwsFPE",
             "difficulty": "beginner", "duration_min": 10,
             "description": "Visual intro to socialization — essential for social science essays."},
            {"id": "f002", "type": "pdf", "title": "Harvard Guide to Using Sources",
             "url": "https://usingsources.fas.harvard.edu/",
             "difficulty": "intermediate", "duration_min": 40,
             "description": "Learn to integrate sources into academic writing correctly."},
            {"id": "f003", "type": "practice", "title": "Purdue OWL: Essay Outlining Practice",
             "url": "https://owl.purdue.edu/owl/general_writing/the_writing_process/developing_an_outline/how_to_outline.html",
             "difficulty": "beginner", "duration_min": 30,
             "description": "Practice structuring arguments for assignment essays."},
        ]
    },
    "AAA_CMA": {
        "topic": "Social Sciences — Quizzes",
        "resources": [
            {"id": "f004", "type": "video", "title": "Crash Course Psychology: Research Methods",
             "url": "https://www.youtube.com/watch?v=hFV71QPvX2I",
             "difficulty": "beginner", "duration_min": 11,
             "description": "Quick review of research methods for quiz preparation."},
            {"id": "f005", "type": "practice", "title": "Quizlet: Sociology Flashcards",
             "url": "https://quizlet.com/subject/sociology/",
             "difficulty": "intermediate", "duration_min": 25,
             "description": "Interactive flashcards for key social science terms."},
        ]
    },
    "AAA_Exam": {
        "topic": "Social Sciences — Exams",
        "resources": [
            {"id": "f040", "type": "video", "title": "Crash Course Sociology Full Series",
             "url": "https://www.youtube.com/playlist?list=PL8dPuuaLjXtMJ-AfB_7J1538YKWkZAnGA",
             "difficulty": "intermediate", "duration_min": 60,
             "description": "Complete sociology playlist for comprehensive exam review."},
            {"id": "f041", "type": "practice", "title": "Social Sciences Practice Exams",
             "url": "https://www.khanacademy.org/humanities/us-history",
             "difficulty": "advanced", "duration_min": 45,
             "description": "Practice questions covering major social science concepts."},
        ]
    },
    "BBB_TMA": {
        "topic": "Technology & IT — Programming Projects",
        "resources": [
            {"id": "f006", "type": "video", "title": "Harvard CS50: Computational Thinking",
             "url": "https://www.youtube.com/watch?v=8mAITcNt710",
             "difficulty": "intermediate", "duration_min": 120,
             "description": "Deep dive into algorithms and computational thinking."},
            {"id": "f007", "type": "video", "title": "Python for Beginners (freeCodeCamp)",
             "url": "https://www.youtube.com/watch?v=rfscVS0vtbw",
             "difficulty": "beginner", "duration_min": 240,
             "description": "Comprehensive 4-hour Python programming tutorial."},
            {"id": "f008", "type": "practice", "title": "LeetCode: Arrays & Strings",
             "url": "https://leetcode.com/explore/learn/card/array-and-string/",
             "difficulty": "advanced", "duration_min": 90,
             "description": "Hands-on coding challenges for IT projects."},
        ]
    },
    "BBB_CMA": {
        "topic": "Technology & IT — Data Structures",
        "resources": [
            {"id": "f009", "type": "video", "title": "Data Structures Easy to Advanced",
             "url": "https://www.youtube.com/watch?v=RBSGKlAvoiM",
             "difficulty": "intermediate", "duration_min": 480,
             "description": "Visual walkthrough of all major data structures."},
            {"id": "f010", "type": "practice", "title": "HackerRank: Data Structures",
             "url": "https://www.hackerrank.com/domains/data-structures",
             "difficulty": "advanced", "duration_min": 60,
             "description": "Interactive problems to test algorithm knowledge."},
        ]
    },
    "CCC_TMA": {
        "topic": "Mathematics (STEM) — Problem Solving",
        "resources": [
            {"id": "f011", "type": "video", "title": "3Blue1Brown: Essence of Calculus",
             "url": "https://www.youtube.com/watch?v=WUvTyaaNkzM",
             "difficulty": "beginner", "duration_min": 17,
             "description": "Beautiful visual explanation of calculus foundations."},
            {"id": "f012", "type": "video", "title": "3Blue1Brown: Essence of Linear Algebra",
             "url": "https://www.youtube.com/watch?v=fNk_zzaMoSs",
             "difficulty": "intermediate", "duration_min": 15,
             "description": "Visualizing matrices, vectors, and transformations."},
            {"id": "f013", "type": "practice", "title": "Khan Academy: AP Calculus AB",
             "url": "https://www.khanacademy.org/math/ap-calculus-ab",
             "difficulty": "intermediate", "duration_min": 120,
             "description": "Step-by-step calculus problems with hints and solutions."},
        ]
    },
    "CCC_Exam": {
        "topic": "Mathematics (STEM) — Exam Prep",
        "resources": [
            {"id": "f014", "type": "video", "title": "MIT 18.06: Linear Algebra Review",
             "url": "https://www.youtube.com/watch?v=l_EUgH_iG-M",
             "difficulty": "advanced", "duration_min": 50,
             "description": "Comprehensive linear algebra review for final exams."},
            {"id": "f015", "type": "practice", "title": "Brilliant.org: Math Foundations",
             "url": "https://brilliant.org/courses/math-fundamentals/",
             "difficulty": "intermediate", "duration_min": 60,
             "description": "Interactive problem-solving for exam-level math."},
        ]
    },
    "DDD_TMA": {
        "topic": "Health Sciences — Research & Essays",
        "resources": [
            {"id": "f016", "type": "video", "title": "Crash Course Anatomy & Physiology",
             "url": "https://www.youtube.com/watch?v=uBGl2B17ARY",
             "difficulty": "beginner", "duration_min": 11,
             "description": "Engaging intro to human body systems and terminology."},
            {"id": "f017", "type": "pdf", "title": "NIH Guide to Clinical Research",
             "url": "https://clinicalcenter.nih.gov/training/training/principles.html",
             "difficulty": "advanced", "duration_min": 55,
             "description": "Official guide for health sciences research methodology."},
            {"id": "f018", "type": "practice", "title": "Khan Academy: Health & Medicine",
             "url": "https://www.khanacademy.org/science/health-and-medicine",
             "difficulty": "intermediate", "duration_min": 45,
             "description": "Practice health science concepts with interactive modules."},
        ]
    },
    "DDD_CMA": {
        "topic": "Health Sciences — Knowledge Checks",
        "resources": [
            {"id": "f019", "type": "video", "title": "Ninja Nerd: Cellular Respiration",
             "url": "https://www.youtube.com/watch?v=eJ9ZjcSqFqk",
             "difficulty": "intermediate", "duration_min": 45,
             "description": "Detailed lecture on cellular biology concepts."},
            {"id": "f020", "type": "practice", "title": "Medscape: Clinical Cases",
             "url": "https://www.medscape.com/index/list_3345_0",
             "difficulty": "advanced", "duration_min": 30,
             "description": "Interactive real-world clinical case studies."},
            {"id": "f021", "type": "pdf", "title": "WHO: Basic Epidemiology",
             "url": "https://apps.who.int/iris/handle/10665/43541",
             "difficulty": "advanced", "duration_min": 80,
             "description": "Foundational text for public health concepts."},
        ]
    },
    "EEE_TMA": {
        "topic": "Business & Economics — Assignments",
        "resources": [
            {"id": "f022", "type": "video", "title": "Crash Course Economics",
             "url": "https://www.youtube.com/playlist?list=PL8dPuuaLjXtPNZwz5_o_5uirJ8gQXnhEO",
             "difficulty": "beginner", "duration_min": 15,
             "description": "Fun, fast-paced intro to economics fundamentals."},
            {"id": "f023", "type": "practice", "title": "Khan Academy: Microeconomics",
             "url": "https://www.khanacademy.org/economics-finance-domain/microeconomics",
             "difficulty": "intermediate", "duration_min": 60,
             "description": "Practice supply, demand, and market structure problems."},
        ]
    },
    "EEE_CMA": {
        "topic": "Business & Economics — Quizzes",
        "resources": [
            {"id": "f024", "type": "video", "title": "Principles of Macroeconomics (MIT OCW)",
             "url": "https://www.youtube.com/watch?v=tyZ5a5YiQFY",
             "difficulty": "intermediate", "duration_min": 50,
             "description": "MIT lecture on macroeconomic principles."},
            {"id": "f025", "type": "practice", "title": "Investopedia: Financial Quizzes",
             "url": "https://www.investopedia.com/financial-term-dictionary-4769738",
             "difficulty": "beginner", "duration_min": 20,
             "description": "Test your finance and economics vocabulary."},
        ]
    },
    "FFF_TMA": {
        "topic": "Humanities — Essays",
        "resources": [
            {"id": "f026", "type": "video", "title": "Crash Course Philosophy",
             "url": "https://www.youtube.com/playlist?list=PL8dPuuaLjXtNgK6MZucdYldNkMybYIHKR",
             "difficulty": "beginner", "duration_min": 10,
             "description": "Engaging intro to philosophical concepts."},
            {"id": "f027", "type": "pdf", "title": "Stanford Encyclopedia of Philosophy",
             "url": "https://plato.stanford.edu/",
             "difficulty": "advanced", "duration_min": 45,
             "description": "The gold-standard reference for humanities topics."},
        ]
    },
    "GGG_TMA": {
        "topic": "Arts & Design — Projects",
        "resources": [
            {"id": "f028", "type": "video", "title": "The Futur: Design Principles",
             "url": "https://www.youtube.com/watch?v=a5KYlHNKQB8",
             "difficulty": "beginner", "duration_min": 15,
             "description": "Core design principles explained visually."},
            {"id": "f029", "type": "practice", "title": "Dribbble: Design Inspiration",
             "url": "https://dribbble.com/",
             "difficulty": "intermediate", "duration_min": 30,
             "description": "Browse top designs for inspiration and practice."},
        ]
    },
}

# Build a simple lookup: base code (e.g. "AAA") -> list of all resources
_BASE_CODE_CATALOG = {}
for _key, _val in FALLBACK_CATALOG.items():
    _base = _key.split("_")[0]
    if _base not in _BASE_CODE_CATALOG:
        _BASE_CODE_CATALOG[_base] = []
    for _r in _val["resources"]:
        _BASE_CODE_CATALOG[_base].append({**_r, "concept": _key, "topic": _val["topic"]})


class DynamicLLMRecommender:
    """
    Tries OpenRouter LLM for dynamic recommendations first.
    If the LLM is unavailable or rate-limited, silently falls back
    to the curated offline catalog so the app ALWAYS works.
    """

    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.base_url = OPENROUTER_BASE_URL
        self.model = LLM_MODEL

    def recommend(
        self,
        student_id: int,
        weak_concepts: list[dict],
        student_profile: Optional[dict] = None,
        top_k: int = 5,
    ) -> list[dict]:
        if not weak_concepts:
            logger.info(f"  No weak concepts for student {student_id}")
            return []

        # ── Try LLM first ──
        if self.api_key:
            llm_result = self._try_llm(weak_concepts, student_profile, top_k)
            if llm_result:
                logger.info(f"  ✓ LLM generated {len(llm_result)} recommendations")
                return llm_result

        # ── Fallback to offline catalog ──
        logger.info("  ⚡ Using offline fallback catalog")
        return self._fallback_recommend(weak_concepts, top_k)

    def _try_llm(self, weak_concepts, student_profile, top_k) -> list[dict]:
        """Attempt to get recommendations from OpenRouter. Returns [] on failure."""
        weak_topics_text = "\n".join([
            f"- Concept: {g.get('concept', 'Unknown')} | Severity: {g.get('severity', 'Unknown')}"
            for g in weak_concepts
        ])

        profile_text = "No profile available."
        if student_profile:
            profile_text = (
                f"- Education level: {student_profile.get('highest_education', 'Unknown')}\n"
                f"- Age group: {student_profile.get('age_band', 'Unknown')}\n"
                f"- Engagement level: {student_profile.get('engagement_level', 'Unknown')}\n"
                f"- Risk level: {student_profile.get('risk_level', 'Unknown')}"
            )

        prompt = f"""You are an expert educational AI advisor. A student needs help with:
{weak_topics_text}

Student Profile:
{profile_text}

Generate EXACTLY {top_k} real-world study resources. For each, provide a YouTube search URL or a known educational site.

Respond ONLY with a valid JSON array (no markdown, no explanation). Each object must have:
- "id": unique string (e.g., "rec_1")
- "concept": the exact concept name from above
- "topic": human-readable sub-topic
- "type": "video", "pdf", or "practice"
- "title": realistic resource title
- "url": functional URL
- "difficulty": "beginner", "intermediate", or "advanced"
- "duration_min": integer minutes
- "description": 1-2 sentence explanation
"""

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
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5,
                    "max_tokens": 1500,
                },
                timeout=30,
            )

            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"].strip()
                # Strip markdown fences
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                recs = json.loads(content)
                for r in recs:
                    r["source"] = "llm_dynamic"
                return recs
            else:
                logger.warning(f"  LLM API error {response.status_code}")
                return []

        except Exception as e:
            logger.warning(f"  LLM request failed: {e}")
            return []

    def _fallback_recommend(self, weak_concepts: list[dict], top_k: int) -> list[dict]:
        """Pull resources from the offline catalog for each weak concept."""
        results = []
        seen_ids = set()

        for gap in weak_concepts:
            concept = gap.get("concept", "")

            # Direct match (e.g. "DDD_TMA")
            if concept in FALLBACK_CATALOG:
                for r in FALLBACK_CATALOG[concept]["resources"]:
                    if r["id"] not in seen_ids:
                        seen_ids.add(r["id"])
                        results.append({
                            **r,
                            "concept": concept,
                            "topic": FALLBACK_CATALOG[concept]["topic"],
                            "source": "curated_catalog",
                        })

            # Base-code match (e.g. "DDD" matches DDD_TMA and DDD_CMA)
            base = concept.split("_")[0]
            if base in _BASE_CODE_CATALOG:
                for r in _BASE_CODE_CATALOG[base]:
                    if r["id"] not in seen_ids:
                        seen_ids.add(r["id"])
                        results.append({**r, "source": "curated_catalog"})

        return results[:top_k]

    def fit(self, interaction_matrix=None, **kwargs):
        """No-op: this recommender doesn't require training."""
        logger.info("  ✓ DynamicLLMRecommender ready (no training required)")
        return self

    def save(self, path: Optional[Path] = None):
        """No-op: stateless recommender, nothing to persist."""
        pass

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DynamicLLMRecommender":
        return cls()


# Alias for backward compatibility (run_pipeline.py imports HybridRecommender)
HybridRecommender = DynamicLLMRecommender
