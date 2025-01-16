import unittest
import asyncio
from unittest.mock import patch
from src.mmkg_rag.index.deduplicate import (
    deduplicate,
    group_by_name_alias,
    group_by_name_alias_v2,
    group_relations,
)
from src.mmkg_rag.types.entity import Entity
from src.mmkg_rag.types.relation import Relation


class TestDeduplicate(unittest.TestCase):
    def test_group_by_name_alias(self):
        # Test case 1: Basic name matching
        e1 = Entity(name="John", description="desc1", label="person")
        e2 = Entity(name="John", description="desc2", label="person")
        result = group_by_name_alias_v2([e1, e2])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)

        # Test case 2: Alias matching
        e1 = Entity(
            name="John", description="desc1", label="person", aliases=["Johnny"]
        )
        e2 = Entity(name="Johnny", description="desc2", label="person")
        result = group_by_name_alias_v2([e1, e2])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)

        # Test case 3: Transitive grouping through aliases
        e1 = Entity(
            name="John", description="desc1", label="person", aliases=["Johnny"]
        )
        e2 = Entity(name="Johnny", description="desc2", label="person", aliases=["J"])
        e3 = Entity(name="J", description="desc3", label="person")
        result = group_by_name_alias_v2([e1, e2, e3])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 3)

        # Test case 4: Empty input
        result = group_by_name_alias_v2([])
        self.assertEqual(len(result), 0)

        # Test case 5: No overlapping names
        e1 = Entity(name="John", description="desc1", label="person")
        e2 = Entity(name="Jane", description="desc2", label="person")
        result = group_by_name_alias_v2([e1, e2])
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(len(result[1]), 1)

        # Test case 6: capitalization and whitespace
        e1 = Entity(name="John", description="desc1", label="person")
        e2 = Entity(name="john", description="desc2", label="person")
        result = group_by_name_alias_v2([e1, e2])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)

    @patch("src.mgrag.index.deduplicate.llm.chat")
    async def async_test_deduplicate(self, mock_chat):
        # Mock LLM response in the format specified by DEDUPLICATE_SYSTEM
        mock_chat.return_value = """
        - <John Smith, person, "A person known by multiple names including John and Johnny. Combined description from multiple sources.", ["ref1", "ref2"]>
        """

        # Test case 1: Basic deduplication
        e1 = Entity(
            name="John",
            description="desc1",
            label="person",
            aliases=["Johnny"],
            chunks=[1],
            images=["img1"],
            references=["ref1"],
        )
        e2 = Entity(
            name="Johnny",
            description="desc2",
            label="person",
            aliases=["JS"],
            chunks=[2],
            images=["img2"],
            references=["ref2"],
        )
        r1 = Relation(source="John", target="Someone", label="knows")
        r2 = Relation(source="Someone", target="Johnny", label="friend_of")

        new_entities, new_relations = await deduplicate([e1, e2], [r1, r2])

        # Verify results
        self.assertEqual(len(new_entities), 1)
        merged_entity = new_entities[0]
        self.assertEqual(merged_entity.name, "John Smith")
        self.assertEqual(merged_entity.label, "person")

        # Verify relations are updated
        self.assertEqual(len(new_relations), 2)
        self.assertEqual(new_relations[0].source, "John Smith")
        self.assertEqual(new_relations[1].target, "John Smith")

    def test_deduplicate(self):
        # Run async test using asyncio
        asyncio.run(self.async_test_deduplicate())

    def test_group_relations(self):
        # Test case 1: Basic grouping
        r1 = Relation(source="John", target="Someone", label="knows")
        r2 = Relation(source="Someone", target="John", label="friend_of")
        result = group_relations([r1, r2])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)

        # Test case 2: No overlapping relations
        r1 = Relation(source="John", target="Someone", label="knows")
        r2 = Relation(source="Jahn", target="Someone", label="knows")
        result = group_relations([r1, r2])
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(len(result[1]), 1)
