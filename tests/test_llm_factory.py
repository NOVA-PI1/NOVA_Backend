import unittest

from config import Settings
from llm.factory import create_llm_provider


class LLMFactoryTests(unittest.TestCase):
    def test_fake_provider_is_selectable(self):
        settings = Settings(llm_provider="fake", llm_model="nova-fake")

        provider = create_llm_provider(settings)

        self.assertEqual(provider.name, "fake")

    def test_public_openai_compatible_providers_are_selectable(self):
        cases = [
            ("groq", {"groq_api_key": "test-key"}),
            ("openrouter", {"openrouter_api_key": "test-key"}),
            ("together", {"together_api_key": "test-key"}),
            ("ollama", {}),
        ]

        for provider_name, credentials in cases:
            with self.subTest(provider=provider_name):
                settings = Settings(
                    llm_provider=provider_name,
                    llm_model="test-model",
                    **credentials,
                )

                provider = create_llm_provider(settings)

                self.assertEqual(provider.name, provider_name)


if __name__ == "__main__":
    unittest.main()
