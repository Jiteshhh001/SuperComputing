"""
Research-specific tools for the Deep Research Agent.
- Tavily Search: web search via Tavily API
- Website Reader: fetch and extract content from URLs
- Arxiv Search: search academic papers
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from langchain_core.tools import tool

from app.config import settings
from app.utils.logger import logger


@tool
def tavily_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search the web using Tavily API for real-time information.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of search results with title, url, and content.
    """
    try:
        from langchain_tavily import TavilySearch

        search = TavilySearch(
            max_results=max_results,
            search_depth="basic",
            tavily_api_key=settings.tavily_api_key,
        )
        results = search.invoke(query)

        if isinstance(results, str):
            return [{"title": "Search Result", "url": "", "content": results}]

        if isinstance(results, list):
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                }
                for r in results
            ]

        return [{"title": "Search Result", "url": "", "content": str(results)}]

    except Exception as e:
        logger.error("Tavily search error: %s", str(e))
        return [{"title": "Error", "url": "", "content": f"Search failed: {str(e)}"}]


@tool
def read_webpage(url: str) -> Dict[str, str]:
    """Fetch and extract text content from a webpage URL.

    Args:
        url: The URL to fetch content from.

    Returns:
        Dictionary with title and extracted text content.
    """
    try:
        import httpx
        from bs4 import BeautifulSoup

        response = httpx.get(url, timeout=15, follow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        title = soup.title.string if soup.title else "No Title"
        text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        # Truncate if too long
        if len(clean_text) > 5000:
            clean_text = clean_text[:5000] + "\n\n[Content truncated...]"

        return {"title": title, "url": url, "content": clean_text}

    except Exception as e:
        logger.error("Webpage read error for %s: %s", url, str(e))
        return {"title": "Error", "url": url, "content": f"Failed to read: {str(e)}"}


@tool
def arxiv_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search for academic papers on arXiv.

    Args:
        query: The search query for academic papers.
        max_results: Maximum number of papers to return.

    Returns:
        List of papers with title, authors, summary, and URL.
    """
    try:
        import arxiv

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = []
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "authors": ", ".join(a.name for a in paper.authors[:5]),
                "summary": paper.summary[:500],
                "url": paper.entry_id,
                "published": str(paper.published.date()) if paper.published else "",
                "categories": ", ".join(paper.categories[:3]),
            })

        logger.info("Arxiv search '%s': %d results", query[:50], len(results))
        return results

    except Exception as e:
        logger.error("Arxiv search error: %s", str(e))
        return [{"title": "Error", "summary": f"Search failed: {str(e)}", "url": ""}]


# Export all tools
RESEARCH_TOOLS = [tavily_search, read_webpage, arxiv_search]
