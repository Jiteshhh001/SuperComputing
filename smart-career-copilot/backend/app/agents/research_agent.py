"""
Deep Research Agent — multi-source research with citations and charts.
Implements: plan → search → read → synthesize → cite → chart.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from app.config import settings
from app.models.schemas import ResearchReport, ResearchSource
from app.tools.research_tools import arxiv_search, read_webpage, tavily_search
from app.rag.document_processor import DocumentProcessor
from app.rag.vectorstore import vectorstore_manager
from app.memory.long_term import long_term_memory
from app.utils.logger import logger


RESEARCH_COLLECTION = "research_documents"


class ResearchAgentRunner:
    """Orchestrates deep research tasks across multiple sources."""

    def __init__(self):
        self._llm = None
        self.doc_processor = DocumentProcessor()

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=settings.worker_model,
                temperature=0.2,
                openai_api_key=settings.openai_api_key,
            )
        return self._llm

    async def research(
        self,
        query: str,
        depth: str = "standard",
        sources: List[str] = None,
        max_results: int = 5,
    ) -> ResearchReport:
        """Execute a multi-source research pipeline."""
        sources = sources or ["web", "arxiv"]
        all_results = []
        thinking_steps = []

        # Step 1: Plan research approach
        thinking_steps.append(f"Planning research for: {query}")

        # Step 2: Web search
        if "web" in sources:
            thinking_steps.append("Searching the web via Tavily...")
            try:
                web_results = tavily_search.invoke({
                    "query": query,
                    "max_results": max_results,
                })
                for r in web_results:
                    all_results.append(
                        ResearchSource(
                            title=r.get("title", "Web Result"),
                            url=r.get("url", ""),
                            snippet=r.get("content", "")[:300],
                            source_type="web",
                        )
                    )
                thinking_steps.append(f"Found {len(web_results)} web results")
            except Exception as e:
                logger.error("Web search failed: %s", str(e))

        # Step 3: Academic search
        if "arxiv" in sources:
            thinking_steps.append("Searching academic papers on arXiv...")
            try:
                papers = arxiv_search.invoke({
                    "query": query,
                    "max_results": min(max_results, 3),
                })
                for p in papers:
                    all_results.append(
                        ResearchSource(
                            title=p.get("title", "Paper"),
                            url=p.get("url", ""),
                            snippet=p.get("summary", "")[:300],
                            source_type="arxiv",
                        )
                    )
                thinking_steps.append(f"Found {len(papers)} academic papers")
            except Exception as e:
                logger.error("Arxiv search failed: %s", str(e))

        # Step 4: Deep read (for standard/deep depth)
        if depth in ("standard", "deep") and all_results:
            thinking_steps.append("Reading top sources in detail...")
            urls_to_read = [
                r.url for r in all_results[:3] if r.url and r.source_type == "web"
            ]
            for url in urls_to_read:
                try:
                    page_content = read_webpage.invoke({"url": url})
                    # Store in RAG for future reference
                    docs = self.doc_processor.process_text(
                        page_content.get("content", ""),
                        metadata={"source": url, "type": "research"},
                    )
                    vectorstore_manager.add_documents(RESEARCH_COLLECTION, docs)
                except Exception as e:
                    logger.error("Page read failed: %s", str(e))

        # Step 5: Synthesize findings
        thinking_steps.append("Synthesizing findings...")
        summary, key_findings = await self._synthesize(query, all_results)

        # Step 6: Generate citations
        citations = [
            f"[{i+1}] {r.title}. {r.url}" for i, r in enumerate(all_results) if r.url
        ]

        # Step 7: Generate chart data
        charts = await self._generate_charts(query, summary)

        # Store in long-term memory
        long_term_memory.store_memory(
            content=f"Research on '{query}': {summary[:500]}",
            memory_type="research",
            metadata={"query": query},
        )

        return ResearchReport(
            query=query,
            summary=summary,
            key_findings=key_findings,
            sources=all_results,
            charts=charts,
            citations=citations,
        )

    async def _synthesize(
        self,
        query: str,
        sources: List[ResearchSource],
    ) -> tuple[str, List[str]]:
        """Synthesize research findings into a coherent summary."""
        source_texts = "\n\n".join(
            f"Source: {s.title}\n{s.snippet}" for s in sources[:8]
        )

        prompt = f"""Based on these research sources, provide a comprehensive answer to: "{query}"

Sources:
{source_texts}

Provide:
1. A detailed summary (2-3 paragraphs)
2. 4-6 key findings as bullet points

Format your response as JSON:
{{"summary": "...", "key_findings": ["...", "..."]}}

Return ONLY valid JSON."""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip().strip("```json").strip("```")
            data = json.loads(content)
            return data.get("summary", ""), data.get("key_findings", [])
        except Exception as e:
            logger.error("Synthesis error: %s", str(e))
            return "Unable to synthesize findings.", []

    async def _generate_charts(
        self,
        query: str,
        summary: str,
    ) -> List[Dict[str, Any]]:
        """Generate chart data from research findings."""
        prompt = f"""Based on this research summary, generate chart data if applicable.

Query: {query}
Summary: {summary[:1000]}

If charts make sense for this data, return a JSON array of chart objects:
[{{"type": "bar", "title": "Chart Title", "data": [{{"label": "X", "value": 10}}]}}]

If no charts are appropriate, return an empty array: []
Return ONLY valid JSON."""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip().strip("```json").strip("```")
            return json.loads(content)
        except Exception:
            return []

    async def summarize_url(self, url: str) -> str:
        """Summarize content from a specific URL."""
        try:
            page = read_webpage.invoke({"url": url})
            content = page.get("content", "")

            prompt = f"""Summarize this webpage content concisely:

{content[:3000]}

Provide a clear, informative summary in 2-3 paragraphs."""

            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"Error summarizing URL: {str(e)}"

    def export_to_markdown(self, report: ResearchReport) -> str:
        """Export a research report as markdown."""
        md = f"# Research Report: {report.query}\n\n"
        md += f"## Summary\n\n{report.summary}\n\n"

        if report.key_findings:
            md += "## Key Findings\n\n"
            for finding in report.key_findings:
                md += f"- {finding}\n"
            md += "\n"

        if report.sources:
            md += "## Sources\n\n"
            for i, src in enumerate(report.sources):
                md += f"{i+1}. [{src.title}]({src.url})\n"
                if src.snippet:
                    md += f"   > {src.snippet[:150]}...\n"
            md += "\n"

        if report.citations:
            md += "## Citations\n\n"
            for cite in report.citations:
                md += f"- {cite}\n"

        return md

    async def run(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle a research-related chat message."""
        # Check if there's relevant past research in RAG
        rag_context = ""
        try:
            docs = vectorstore_manager.similarity_search(
                RESEARCH_COLLECTION, message, k=3
            )
            if docs:
                rag_context = "\n\nRelevant past research:\n" + "\n".join(
                    d.page_content[:300] for d in docs
                )
        except Exception:
            pass

        # Perform quick research
        report = await self.research(query=message, depth="standard")

        response = f"{report.summary}\n\n"
        if report.key_findings:
            response += "### Key Findings\n"
            for f in report.key_findings:
                response += f"- {f}\n"

        sources_data = [
            {"title": s.title, "url": s.url, "snippet": s.snippet}
            for s in report.sources
        ]

        return {
            "response": response,
            "agent_type": "research",
            "sources": sources_data,
            "artifacts": {"charts": report.charts},
            "thinking_steps": [
                "Planned research approach",
                f"Searched {len(report.sources)} sources",
                "Synthesized findings",
                "Generated citations",
            ],
        }
