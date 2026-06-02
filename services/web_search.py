from __future__ import annotations

import httpx

from config import Settings
from schemas import WebSearchResult


class WebSearchService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(self, query: str, limit: int = 5) -> list[WebSearchResult]:
        if not self.settings.web_search_enabled:
            return []
        if self.settings.web_search_provider == "fake":
            return [
                WebSearchResult(
                    title="Contexto web simulado",
                    url="https://example.com/nova-web-context",
                    snippet=f"Resultado simulado para: {query[:180]}",
                    source="fake",
                )
            ]
        if not self.settings.web_search_api_key:
            return []

        provider = self.settings.web_search_provider
        try:
            if provider == "tavily":
                return await self._search_tavily(query, limit)
            if provider == "brave":
                return await self._search_brave(query, limit)
            if provider == "serpapi":
                return await self._search_serpapi(query, limit)
        except httpx.HTTPError:
            return []
        return []

    async def _search_tavily(self, query: str, limit: int) -> list[WebSearchResult]:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.settings.web_search_api_key,
                    "query": query,
                    "max_results": limit,
                    "include_answer": False,
                },
            )
            response.raise_for_status()
        results = response.json().get("results", [])
        return [
            WebSearchResult(
                title=item.get("title") or item.get("url") or "Resultado web",
                url=item.get("url") or "",
                snippet=item.get("content") or "",
                published_at=item.get("published_date"),
                source="tavily",
            )
            for item in results
            if item.get("url")
        ]

    async def _search_brave(self, query: str, limit: int) -> list[WebSearchResult]:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": limit},
                headers={"X-Subscription-Token": self.settings.web_search_api_key or ""},
            )
            response.raise_for_status()
        results = response.json().get("web", {}).get("results", [])
        return [
            WebSearchResult(
                title=item.get("title") or item.get("url") or "Resultado web",
                url=item.get("url") or "",
                snippet=item.get("description") or "",
                published_at=item.get("age"),
                source="brave",
            )
            for item in results
            if item.get("url")
        ]

    async def _search_serpapi(self, query: str, limit: int) -> list[WebSearchResult]:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                "https://serpapi.com/search.json",
                params={"q": query, "api_key": self.settings.web_search_api_key, "num": limit},
            )
            response.raise_for_status()
        results = response.json().get("organic_results", [])
        return [
            WebSearchResult(
                title=item.get("title") or item.get("link") or "Resultado web",
                url=item.get("link") or "",
                snippet=item.get("snippet") or "",
                published_at=item.get("date"),
                source="serpapi",
            )
            for item in results
            if item.get("link")
        ]
