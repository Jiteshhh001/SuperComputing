"""
═══════════════════════════════════════════════════════════════════════════════
  Resource Recommender — curated catalog + LLM ranking
  Candidates come from a curated catalog of real, link-checked resources,
  organised by OULAD domain (Social Sciences / STEM) and assessment type.
  When an LLM is available it ranks the candidates for the student and explains
  each pick; otherwise a severity-aware heuristic ranks them. The LLM never
  invents resources or URLs.
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import logging
import re
from typing import Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import MODULE_DOMAINS, format_module_name

logger = logging.getLogger(__name__)

SOCIAL, STEM, ANY = "Social Sciences", "STEM", "Any"

# Each resource: id, domain, assessment types it suits, type, title, url, difficulty, minutes, description
CATALOG = [
    # ── Social Sciences ───────────────────────────────────────────────────
    {"id": "s01", "domain": SOCIAL, "types": ["TMA", "Exam"], "type": "video",
     "title": "Crash Course Sociology (playlist)",
     "url": "https://www.youtube.com/playlist?list=PL8dPuuaLjXtMJ-AfB_7J1538YKWkZAnGA",
     "difficulty": "beginner", "duration_min": 15,
     "description": "Short, visual explanations of core sociological ideas — good for rebuilding foundations."},
    {"id": "s02", "domain": SOCIAL, "types": ["CMA", "TMA"], "type": "video",
     "title": "Psychological Research — Crash Course Psychology #2",
     "url": "https://www.youtube.com/watch?v=hFV71QPvX2I",
     "difficulty": "beginner", "duration_min": 11,
     "description": "Research methods and evidence — frequently tested in computer-marked quizzes."},
    {"id": "s03", "domain": SOCIAL, "types": ["TMA", "Exam"], "type": "video",
     "title": "Crash Course Economics (playlist)",
     "url": "https://www.youtube.com/playlist?list=PL8dPuuaLjXtPNZwz5_o_5uirJ8gQXnhEO",
     "difficulty": "beginner", "duration_min": 12,
     "description": "Concise lessons on economic reasoning used across social science modules."},
    {"id": "s04", "domain": SOCIAL, "types": ["TMA", "Exam"], "type": "video",
     "title": "Crash Course Philosophy (playlist)",
     "url": "https://www.youtube.com/playlist?list=PL8dPuuaLjXtNgK6MZucdYldNkMybYIHKR",
     "difficulty": "intermediate", "duration_min": 10,
     "description": "Argument analysis and critical thinking for essay-based assessments."},
    {"id": "s05", "domain": SOCIAL, "types": ["TMA", "Exam"], "type": "pdf",
     "title": "OpenStax: Introduction to Sociology 3e (free textbook)",
     "url": "https://openstax.org/details/books/introduction-sociology-3e",
     "difficulty": "intermediate", "duration_min": 45,
     "description": "Peer-reviewed open textbook with chapter summaries and review questions."},
    {"id": "s06", "domain": SOCIAL, "types": ["TMA", "Exam"], "type": "pdf",
     "title": "OpenStax: Psychology 2e (free textbook)",
     "url": "https://openstax.org/details/books/psychology-2e",
     "difficulty": "intermediate", "duration_min": 45,
     "description": "Open textbook covering research methods, cognition and behaviour."},
    {"id": "s07", "domain": SOCIAL, "types": ["TMA"], "type": "pdf",
     "title": "Purdue OWL: Essay Writing",
     "url": "https://owl.purdue.edu/owl/general_writing/academic_writing/essay_writing/index.html",
     "difficulty": "beginner", "duration_min": 30,
     "description": "How to plan and structure argumentative and expository essays for assignments."},
    {"id": "s08", "domain": SOCIAL, "types": ["TMA"], "type": "practice",
     "title": "Purdue OWL: How to Outline an Essay",
     "url": "https://owl.purdue.edu/owl/general_writing/the_writing_process/developing_an_outline/how_to_outline.html",
     "difficulty": "intermediate", "duration_min": 30,
     "description": "Practise turning a question into a clear outline before writing."},
    {"id": "s09", "domain": SOCIAL, "types": ["TMA"], "type": "pdf",
     "title": "Purdue OWL: Research and Citation Resources",
     "url": "https://owl.purdue.edu/owl/research_and_citation/resources.html",
     "difficulty": "intermediate", "duration_min": 25,
     "description": "Using and referencing sources correctly — a common reason assignments lose marks."},
    {"id": "s10", "domain": SOCIAL, "types": ["CMA", "Exam"], "type": "practice",
     "title": "Khan Academy: Society and Culture (practice)",
     "url": "https://www.khanacademy.org/test-prep/mcat/society-and-culture",
     "difficulty": "intermediate", "duration_min": 30,
     "description": "Short lessons with practice questions on social structures and culture."},

    # ── STEM ──────────────────────────────────────────────────────────────
    {"id": "t01", "domain": STEM, "types": ["TMA", "Exam"], "type": "video",
     "title": "Essence of Linear Algebra — 3Blue1Brown (playlist)",
     "url": "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab",
     "difficulty": "intermediate", "duration_min": 20,
     "description": "Visual intuition for vectors, matrices and transformations."},
    {"id": "t02", "domain": STEM, "types": ["TMA", "Exam"], "type": "video",
     "title": "The Essence of Calculus — 3Blue1Brown",
     "url": "https://www.youtube.com/watch?v=WUvTyaaNkzM",
     "difficulty": "beginner", "duration_min": 17,
     "description": "Builds calculus intuition from first principles."},
    {"id": "t03", "domain": STEM, "types": ["TMA", "CMA"], "type": "video",
     "title": "Learn Python — Full Course for Beginners (freeCodeCamp)",
     "url": "https://www.youtube.com/watch?v=rfscVS0vtbw",
     "difficulty": "beginner", "duration_min": 60,
     "description": "Programming fundamentals for computing and data-analysis tasks."},
    {"id": "t04", "domain": STEM, "types": ["TMA"], "type": "video",
     "title": "Harvard CS50 — Full Computer Science Course",
     "url": "https://www.youtube.com/watch?v=8mAITcNt710",
     "difficulty": "intermediate", "duration_min": 60,
     "description": "Problem solving and computational thinking from Harvard's intro course."},
    {"id": "t05", "domain": STEM, "types": ["TMA", "Exam"], "type": "pdf",
     "title": "OpenStax: Introductory Statistics 2e (free textbook)",
     "url": "https://openstax.org/details/books/introductory-statistics-2e",
     "difficulty": "intermediate", "duration_min": 45,
     "description": "Open textbook with worked examples and exercises on statistics."},
    {"id": "t06", "domain": STEM, "types": ["Exam", "TMA"], "type": "pdf",
     "title": "MIT OCW 18.05: Introduction to Probability and Statistics",
     "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/",
     "difficulty": "advanced", "duration_min": 60,
     "description": "MIT course notes, problem sets and exams with solutions."},
    {"id": "t07", "domain": STEM, "types": ["Exam"], "type": "pdf",
     "title": "MIT OCW 18.06: Linear Algebra",
     "url": "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/",
     "difficulty": "advanced", "duration_min": 60,
     "description": "Gilbert Strang's lectures, notes and past exams."},
    {"id": "t08", "domain": STEM, "types": ["CMA", "Exam"], "type": "practice",
     "title": "Khan Academy: Statistics & Probability (practice)",
     "url": "https://www.khanacademy.org/math/statistics-probability",
     "difficulty": "beginner", "duration_min": 30,
     "description": "Bite-sized exercises with instant feedback — ideal before computer-marked quizzes."},
    {"id": "t09", "domain": STEM, "types": ["CMA", "TMA"], "type": "practice",
     "title": "Khan Academy: Algebra (practice)",
     "url": "https://www.khanacademy.org/math/algebra",
     "difficulty": "beginner", "duration_min": 30,
     "description": "Mastery-based algebra practice to close foundational gaps."},
    {"id": "t10", "domain": STEM, "types": ["TMA", "Exam"], "type": "pdf",
     "title": "Paul's Online Math Notes",
     "url": "https://tutorial.math.lamar.edu/",
     "difficulty": "intermediate", "duration_min": 30,
     "description": "Clear notes and practice problems for algebra and calculus."},
    {"id": "t11", "domain": STEM, "types": ["CMA"], "type": "practice",
     "title": "Khan Academy: Biology (practice)",
     "url": "https://www.khanacademy.org/science/biology",
     "difficulty": "beginner", "duration_min": 30,
     "description": "Science fundamentals with quick checks for understanding."},

    # ── Any domain: study skills ──────────────────────────────────────────
    {"id": "g01", "domain": ANY, "types": ["Exam"], "type": "video",
     "title": "How to Revise for Exams Effectively (YouTube search)",
     "url": "https://www.youtube.com/results?search_query=how+to+revise+for+exams+effectively",
     "difficulty": "beginner", "duration_min": 15,
     "description": "Active recall, spaced repetition and past-paper practice for exams."},
    {"id": "g02", "domain": ANY, "types": ["TMA", "CMA", "Exam"], "type": "practice",
     "title": "Learning How to Learn (Coursera, free to audit)",
     "url": "https://www.coursera.org/learn/learning-how-to-learn",
     "difficulty": "beginner", "duration_min": 30,
     "description": "Evidence-based techniques for focus, memory and avoiding procrastination."},
]

_TYPE_LABEL = {"TMA": "assignment", "CMA": "quiz", "Exam": "exam"}
_PREFERRED_DIFFICULTY = {
    "Critical": ["beginner", "intermediate", "advanced"],
    "High": ["beginner", "intermediate", "advanced"],
    "Moderate": ["intermediate", "beginner", "advanced"],
    "Low": ["intermediate", "advanced", "beginner"],
}


def _split_concept(concept: str) -> tuple[str, str]:
    code, _, a_type = concept.partition("_")
    return code, a_type


def candidates_for(concept: str) -> list[dict]:
    """Catalog resources suitable for a concept such as 'CCC_TMA'."""
    code, a_type = _split_concept(concept)
    domain = MODULE_DOMAINS.get(code, ANY)
    out = [r for r in CATALOG if r["domain"] in (domain, ANY) and (not a_type or a_type in r["types"])]
    if len(out) < 3:  # widen to the whole domain if the assessment type is rare
        out += [r for r in CATALOG if r["domain"] == domain and r not in out]
    return out


class Recommender:
    """
    Ranks curated resources for a student's weak concepts.
    Uses an LLM (if configured) to choose and explain the best matches.
    """

    def __init__(self, llm=None):
        self.llm = llm
        self._cache: dict = {}
        self.last_mode = "heuristic"

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

        key = (student_id, tuple((g.get("concept"), g.get("severity")) for g in weak_concepts), top_k)
        if key in self._cache:
            return self._cache[key]

        pool = self._candidate_pool(weak_concepts)
        picks = None
        if self.llm is not None and self.llm.available:
            picks = self._llm_rank(pool, weak_concepts, student_profile, top_k)
        if picks:
            self.last_mode = "llm"
        else:
            picks = self._heuristic_rank(pool, weak_concepts, top_k)
            self.last_mode = "heuristic"

        self._cache[key] = picks
        logger.info(f"  ✓ {len(picks)} recommendations ({self.last_mode})")
        return picks

    # ── Candidate pool ────────────────────────────────────────────────────

    @staticmethod
    def _candidate_pool(weak_concepts: list[dict]) -> list[dict]:
        pool, seen = [], set()
        for gap in weak_concepts:
            concept = gap.get("concept", "")
            for r in candidates_for(concept):
                if (r["id"], concept) in seen:
                    continue
                seen.add((r["id"], concept))
                pool.append({**{k: v for k, v in r.items() if k not in ("domain", "types")},
                             "concept": concept, "topic": format_module_name(concept),
                             "severity": gap.get("severity", "Moderate")})
        return pool

    # ── Heuristic ranking (offline) ───────────────────────────────────────

    @staticmethod
    def _heuristic_rank(pool: list[dict], weak_concepts: list[dict], top_k: int) -> list[dict]:
        """Round-robin over concepts by priority; within a concept prefer the
        difficulty that suits the severity and vary resource types."""
        by_concept = {}
        for r in pool:
            by_concept.setdefault(r["concept"], []).append(r)
        for concept, items in by_concept.items():
            sev = items[0]["severity"]
            order = _PREFERRED_DIFFICULTY.get(sev, _PREFERRED_DIFFICULTY["Moderate"])
            items.sort(key=lambda r: (order.index(r["difficulty"]) if r["difficulty"] in order else 9,
                                      {"video": 0, "practice": 1, "pdf": 2}.get(r["type"], 3)))

        picks, used_ids = [], set()
        concepts = [g["concept"] for g in weak_concepts if g.get("concept") in by_concept]
        while len(picks) < top_k and any(by_concept.get(c) for c in concepts):
            for c in concepts:
                items = by_concept.get(c, [])
                while items and items[0]["id"] in used_ids:
                    items.pop(0)
                if items and len(picks) < top_k:
                    r = items.pop(0)
                    used_ids.add(r["id"])
                    level = _TYPE_LABEL.get(_split_concept(c)[1], "assessment")
                    picks.append({**r, "source": "curated_catalog",
                                  "reason": f"Matches your {r['severity'].lower()}-risk {level} topic "
                                            f"at {r['difficulty']} level."})
        return picks

    # ── LLM ranking ───────────────────────────────────────────────────────

    def _llm_rank(self, pool, weak_concepts, student_profile, top_k) -> Optional[list[dict]]:
        gaps_text = "\n".join(
            f"- {g['concept']} ({format_module_name(g['concept'])}): severity {g.get('severity')}, "
            f"risk {g.get('confidence', 0):.0%}" for g in weak_concepts)
        cand_text = "\n".join(
            f"- id={r['id']} concept={r['concept']} type={r['type']} level={r['difficulty']} "
            f"minutes={r['duration_min']} title=\"{r['title']}\"" for r in pool)
        profile = ""
        if student_profile:
            profile = (f"Education: {student_profile.get('highest_education', 'unknown')}; "
                       f"age band: {student_profile.get('age_band', 'unknown')}; "
                       f"engagement: {student_profile.get('engagement_level', 'unknown')}.")
        prompt = (
            "You are an academic advisor. Choose the best study resources for this student "
            "ONLY from the candidate list. Prioritise the highest-risk concepts, match difficulty "
            "to severity (higher risk → more foundational), and mix resource types.\n\n"
            f"Student: {profile}\nWeak concepts:\n{gaps_text}\n\nCandidates:\n{cand_text}\n\n"
            f"Return JSON only: {{\"picks\": [{{\"id\": \"...\", \"concept\": \"...\", "
            f"\"reason\": \"one short sentence addressed to the student\"}}]}} with exactly {top_k} picks "
            "and no duplicate ids."
        )
        text = self.llm.complete([{"role": "user", "content": prompt}], max_tokens=700,
                                 temperature=0.2, time_budget=30)
        if not text:
            return None
        match = re.search(r"\{.*\}", text, re.S)
        try:
            data = json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            data = None
        if not data or not isinstance(data.get("picks"), list):
            return None

        index = {(r["id"], r["concept"]): r for r in pool}
        by_id = {}
        for r in pool:
            by_id.setdefault(r["id"], r)
        picks, used = [], set()
        for p in data["picks"]:
            rid, concept = str(p.get("id", "")), str(p.get("concept", ""))
            r = index.get((rid, concept)) or by_id.get(rid)
            if not r or rid in used:
                continue
            used.add(rid)
            picks.append({**r, "source": "llm_ranked", "reason": str(p.get("reason", "")).strip()[:200]})
        if len(picks) < min(top_k, len(by_id)):
            # top up with heuristic picks if the model returned too few valid ids
            for r in self._heuristic_rank(pool, weak_concepts, top_k * 2):
                if r["id"] not in used and len(picks) < top_k:
                    used.add(r["id"])
                    picks.append(r)
        return picks or None

    # ── Compatibility with the pipeline runner ────────────────────────────

    def fit(self, interaction_matrix=None, **kwargs):
        """No training needed: ranking uses the curated catalog and the LLM."""
        logger.info(f"  ✓ Recommender ready ({len(CATALOG)} curated resources)")
        return self

    def save(self, path: Optional[Path] = None):
        pass

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Recommender":
        return cls()


# Backwards-compatible names used elsewhere in the project
DynamicLLMRecommender = Recommender
HybridRecommender = Recommender
