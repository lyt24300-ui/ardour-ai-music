"""Web Search & Webpage Reader for plugin discovery and audio production research."""

from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup

from .models import WebSearchConfig


async def web_search(query: str, cfg: WebSearchConfig, max_results: int | None = None) -> dict[str, Any]:
    """Perform web search via DuckDuckGo (free default) or configured API provider."""
    if not cfg.enabled:
        return {"ok": False, "error": "Web Search is disabled in Studio Settings.", "results": []}

    limit = max_results or cfg.max_results
    provider = cfg.provider

    try:
        if provider == "duckduckgo":
            results = await asyncio.to_thread(_search_ddg, query, limit)
        elif provider == "tavily":
            results = await _search_tavily(query, cfg.api_key, limit)
        elif provider == "brave":
            results = await _search_brave(query, cfg.api_key, limit)
        elif provider == "serper":
            results = await _search_serper(query, cfg.api_key, limit)
        elif provider == "searxng":
            results = await _search_searxng(query, cfg.base_url, limit)
        else:
            results = await asyncio.to_thread(_search_ddg, query, limit)

        return {"ok": True, "provider": provider, "query": query, "results": results}
    except Exception as exc:
        return {"ok": False, "provider": provider, "query": query, "error": str(exc), "results": []}


def _search_ddg(query: str, limit: int) -> list[dict[str, str]]:
    from duckduckgo_search import DDGS

    out: list[dict[str, str]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=limit):
            out.append({
                "title": str(r.get("title", "")),
                "url": str(r.get("href", r.get("link", ""))),
                "snippet": str(r.get("body", r.get("snippet", ""))),
            })
    return out


async def _search_tavily(query: str, api_key: str, limit: int) -> list[dict[str, str]]:
    if not api_key:
        raise ValueError("Tavily API key is required.")
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": limit},
        )
        resp.raise_for_status()
        data = resp.json()
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in data.get("results", [])
    ]


async def _search_brave(query: str, api_key: str, limit: int) -> list[dict[str, str]]:
    if not api_key:
        raise ValueError("Brave Search API key is required.")
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": limit},
            headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")}
        for r in data.get("web", {}).get("results", [])
    ]


async def _search_serper(query: str, api_key: str, limit: int) -> list[dict[str, str]]:
    if not api_key:
        raise ValueError("Serper API key is required.")
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": limit},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
    return [
        {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
        for r in data.get("organic", [])
    ]


async def _search_searxng(query: str, base_url: str, limit: int) -> list[dict[str, str]]:
    if not base_url:
        raise ValueError("SearXNG base URL is required.")
    url = base_url.rstrip("/") + "/search"
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(url, params={"q": query, "format": "json"})
        resp.raise_for_status()
        data = resp.json()
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in data.get("results", [])[:limit]
    ]


async def fetch_webpage_text(url: str, max_chars: int = 8000) -> dict[str, Any]:
    """Fetch a webpage and extract clean text (for reading plugin specs/install steps)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ArdourAIMusicStudio/1.0",
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
    }
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "text/html" in ctype or "<html" in resp.text[:500].lower():
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "noscript", "nav", "footer", "svg"]):
                    tag.decompose()
                title = soup.title.string.strip() if soup.title and soup.title.string else ""
                text = soup.get_text(separator="\n")
                text = re.sub(r"\n{3,}", "\n\n", text)
                text = re.sub(r"[ \t]{2,}", " ", text).strip()
            else:
                title = ""
                text = resp.text.strip()
        return {
            "ok": True,
            "url": url,
            "title": title,
            "content": text[:max_chars],
            "truncated": len(text) > max_chars,
        }
    except Exception as exc:
        return {"ok": False, "url": url, "error": str(exc)}
