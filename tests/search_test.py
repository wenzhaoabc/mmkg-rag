import unittest
from pathlib import Path
from src.mmkg_rag.types import Entity, Image, Relation
from src.mmkg_rag.retrieval.search import (
    _search_entities,
    _search_images,
    load_default_ers,
)


class SearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load test data before running tests"""
        # Use absolute path to test data
        test_data_dir = str(Path("databases/RAG"))
        load_default_ers(test_data_dir)

    def test_search_entities_basic(self):
        """Test basic entity search functionality"""
        # Test exact match
        results = _search_entities(
            ["overall architecture", "Graph RAG", "architecture diagram", "explanation"],
            max_num=5,
            similarity_threshold=10,
        )
        print(len(results))
        print([e.name for e in results])
