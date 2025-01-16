from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch, AsyncMock

# ...existing code...
from src.mmkg_rag.utils.llm import LLM


class TestLLM(IsolatedAsyncioTestCase):
    def setUp(self):
        self.llm = LLM()

    @patch("src.mgrag.utils.llm.AsyncOpenAI")
    async def test_chat(self, MockAsyncOpenAI):
        mock_client = MockAsyncOpenAI.return_value
        mock_response = AsyncMock()
        mock_response.choices[0].message.content = (
            "I'm an AI, so I don't have feelings, but I'm here to help you!"
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        input_text = "Hello, how are you?"
        expected_response = (
            "I'm an AI, so I don't have feelings, but I'm here to help you!"
        )

        response = await self.llm.chat(input_text)
        self.assertEqual(response, expected_response)
        mock_client.chat.completions.create.assert_called_once()
