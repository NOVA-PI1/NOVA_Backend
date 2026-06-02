import unittest

from config import Settings
from services.web_search import WebSearchService


class WebSearchServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_search_returns_empty_list(self):
        service = WebSearchService(Settings(web_search_enabled=False))

        self.assertEqual(await service.search("consulta"), [])

    async def test_fake_provider_returns_stable_result_without_api_key(self):
        service = WebSearchService(Settings(web_search_enabled=True, web_search_provider="fake"))

        results = await service.search("consulta")

        self.assertEqual(results[0].source, "fake")
        self.assertIn("consulta", results[0].snippet)

    async def test_real_provider_without_api_key_returns_empty_list(self):
        service = WebSearchService(Settings(web_search_enabled=True, web_search_provider="brave", web_search_api_key=None))

        self.assertEqual(await service.search("consulta"), [])


if __name__ == "__main__":
    unittest.main()
