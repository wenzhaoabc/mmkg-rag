import unittest
import asyncio
from pathlib import Path
from src.mmkg_rag.index.pipe import index_graph, process_files
from src.mmkg_rag.index.mmodal import extract_images


class MModalTest(unittest.TestCase):
    def test_extract_images(self):
        # Simple test case
        simple_text = "Here is ![test image](test.png) in text."
        results = extract_images(simple_text)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "test.png")
        self.assertEqual(results[0][1], "Here is ![test image](test.png) in text.")

        # Test with no images
        no_image_text = "This is a text without any images."
        self.assertEqual(len(extract_images(no_image_text)), 0)

        # Complex test cases from file
        test_file = Path(__file__).parent / "assets" / "mmodaltest.md"
        with open(test_file, "r", encoding="utf-8") as f:
            content = f.read()

        results = extract_images(content)
        self.assertEqual(len(results), 5)  # Should find 5 images

        # Check specific cases
        self.assertEqual(results[0][0], "images/test1.png")
        self.assertTrue("first image" in results[0][1].lower())

        # Check multiple images in same paragraph
        self.assertEqual(results[2][0], "images/test3.png")
        self.assertEqual(results[3][0], "images/test4.jpg")
        self.assertTrue(
            all(len(context) <= 400 for _, context in results)
        )  # Check context length


class IndexTest(unittest.TestCase):
    async def test_index_graph(self):
        # Test case 1: Basic indexing
        file_path = "output/UniST_1-4/md/UniST_1-4.md"
        entities, relations, imgs, image_rels = await index_graph(
            file_path, output_path="UniST_1-4"
        )
        self.assertGreater(len(entities), 3)
        self.assertGreater(len(relations), 2)

    async def test_index_graph_pdf(self):
        # Test case 2: PDF indexing
        file_paths = [
            "examples/rag/graphrag_short.md",
            # "examples/rag/selfrag.pdf",
        ]
        es, rs, imgs, irs = await process_files(file_paths, database="RAG")
        self.assertGreater(len(es), 3)
        self.assertGreater(len(rs), 2)
        self.assertGreater(len(imgs), 1)
        self.assertGreater(len(irs), 3)

    def test_index_graph_sync(self):
        asyncio.run(self.test_index_graph_pdf())


