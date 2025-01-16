import unittest
from unittest.mock import patch, mock_open
import base64
from pathlib import Path

from src.mmkg_rag.index.mmodal import image_description
from src.mmkg_rag.types.image import Image


class ImageDescriptionTest(unittest.TestCase):
    def setUp(self):
        # Mock image data
        self.test_image_data = b"fake_image_data"
        self.test_image_b64 = base64.b64encode(self.test_image_data).decode("utf-8")
        self.test_image_path = "examples/rag/images/graphrag_0.jpg"
        self.test_context = "This is a test image context"

    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    @patch("mgrag.utils.llm.chat_msg_sync")
    async def test_normal_case(self, mock_chat, mock_file, mock_exists):
        # Setup mocks
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = self.test_image_data
        mock_chat.return_value = """
        {
            "caption": "Test Caption",
            "text_snippets": ["test text 1", "test text 2"],
            "description": "Test Description"
        }
        """

        # Call function
        result = await image_description(self.test_image_path, self.test_context)

        # Verify result
        self.assertIsInstance(result, Image)

    @patch("pathlib.Path.exists")
    async def test_file_not_found(self, mock_exists):
        # Setup mock
        mock_exists.return_value = False

        # Call function and verify result
        result = await image_description(self.test_image_path, self.test_context)
        self.assertIsNone(result)

    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    @patch("mgrag.utils.llm.chat_msg_sync")
    async def test_invalid_llm_response(self, mock_chat, mock_file, mock_exists):
        # Setup mocks
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = self.test_image_data
        mock_chat.return_value = "Invalid JSON response"

        # Call function
        result = await image_description(self.test_image_path, self.test_context)

        # Verify result still contains path but empty fields
        self.assertIsInstance(result, Image)
