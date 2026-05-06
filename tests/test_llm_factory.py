import unittest

from config import Settings
from llm.factory import create_llm_provider


class LLMFactoryTests(unittest.TestCase):
    def test_fake_provider_is_selectable(self):
        settings = Settings(llm_provider="fake", llm_model="nova-fake")

        provider = create_llm_provider(settings)

        self.assertEqual(provider.name, "fake")


if __name__ == "__main__":
    unittest.main()
