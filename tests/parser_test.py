import unittest
from src.mmkg_rag.index.parser import (
    parse_er,
    parse_alias,
    parse_image_description,
    parse_json_list,
)
from src.mmkg_rag.retrieval.parser import parse_classify_response
from src.mmkg_rag.types import Entity, Relation


class TestParser(unittest.TestCase):
    def test_parse_er_valid_input(self):
        """Test parsing valid input"""
        rawtext = """
        {
            "name": "Entity1",
            "label": "Type1",
            "description": "Description1",
            "aliases": ["Alias1"],
            "references": ["Ref1"]
        }
        {
            "source": "Entity1",
            "label": "RelationType",
            "target": "Entity2",
            "description": "RelationDescription",
            "references": ["Ref2"]
        }
        """
        entities, relations = parse_er(rawtext)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].name, "Entity1")
        self.assertEqual(entities[0].label, "Type1")
        self.assertEqual(entities[0].description, "Description1")
        self.assertEqual(entities[0].aliases, ["Alias1"])
        self.assertEqual(entities[0].references, ["Ref1"])

        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0].source, "Entity1")
        self.assertEqual(relations[0].label, "RelationType")
        self.assertEqual(relations[0].target, "Entity2")
        self.assertEqual(relations[0].description, "RelationDescription")
        self.assertEqual(relations[0].references, ["Ref2"])

    def test_parse_er_invalid_json(self):
        """Test parsing invalid JSON input"""
        rawtext = """
        {
            "name": "Entity1",
            "label": "Type1",
            "description": "Description1",
            "aliases": ["Alias1"],
            "references": ["Ref1"]
        """
        entities, relations = parse_er(rawtext)
        self.assertEqual(len(entities), 0)
        self.assertEqual(len(relations), 0)

    def test_parse_er_empty_input(self):
        """Test parsing empty input"""
        rawtext = ""
        entities, relations = parse_er(rawtext)
        self.assertEqual(len(entities), 0)
        self.assertEqual(len(relations), 0)

    def test_parse_er_partial_input(self):
        """Test parsing partial input"""
        rawtext = """
        {
            "name": "Entity1",
            "label": "Type1",
            "description": "Description1",
            "aliases": ["Alias1"],
            "references": ["Ref1"]
        }
        """
        entities, relations = parse_er(rawtext)
        self.assertEqual(len(entities), 1)
        self.assertEqual(len(relations), 0)

    def test_parse_er_multiple_entities_relations(self):
        """Test parsing multiple entities and relations"""
        rawtext = """
        {
            "name": "Entity1",
            "label": "Type1",
            "description": "Description1",
            "aliases": ["Alias1"],
            "references": ["Ref1"]
        }
        {
            "name": "Entity2",
            "label": "Type2",
            "description": "Description2",
            "aliases": ["Alias2"],
            "references": ["Ref2"]
        }
        {
            "source": "Entity1",
            "label": "RelationType1",
            "target": "Entity2",
            "description": "RelationDescription1",
            "references": ["Ref3"]
        }
        {
            "source": "Entity2",
            "label": "RelationType2",
            "target": "Entity1",
            "description": "RelationDescription2",
            "references": ["Ref4"]
        }
        """
        entities, relations = parse_er(rawtext)
        self.assertEqual(len(entities), 2)
        self.assertEqual(len(relations), 2)
        self.assertEqual(entities[0].name, "Entity1")
        self.assertEqual(entities[1].name, "Entity2")
        self.assertEqual(relations[0].source, "Entity1")
        self.assertEqual(relations[1].source, "Entity2")


class TestAliasParser(unittest.TestCase):
    def test_basic_alias_parsing(self):
        """Test basic alias parsing"""
        text = """
        <Natural Language Processing, ["NLP"]>
        <Bidirectional Encoder Representations from Transformers, ["BERT", "Bidirectional Transformers"]>
        """
        aliases = parse_alias(text)

        self.assertEqual(len(aliases), 2)
        self.assertEqual(aliases[0], ("Natural Language Processing", ["NLP"]))
        self.assertEqual(
            aliases[1],
            (
                "Bidirectional Encoder Representations from Transformers",
                ["BERT", "Bidirectional Transformers"],
            ),
        )

    def test_quoted_entities(self):
        """Test parsing with quoted entities"""
        text = """
        <"Entity With Spaces", ["E1", "E2"]>
        <"Complex, Entity Name", ["CEN"]>
        """
        aliases = parse_alias(text)

        self.assertEqual(len(aliases), 2)
        self.assertEqual(aliases[0], ("Entity With Spaces", ["E1", "E2"]))
        self.assertEqual(aliases[1], ("Complex, Entity Name", ["CEN"]))

    def test_empty_alias_list(self):
        """Test parsing with empty alias lists"""
        text = "<Entity1, []>"
        aliases = parse_alias(text)

        self.assertEqual(len(aliases), 1)
        self.assertEqual(aliases[0], ("Entity1", []))

    def test_invalid_json(self):
        """Test handling of invalid JSON in alias list"""
        text = """
        <Entity1, ["valid"]>
        
        <Entity3, ["valid2"]>
        """
        aliases = parse_alias(text)

        self.assertEqual(len(aliases), 2)
        self.assertEqual(aliases[0], ("Entity1", ["valid"]))
        self.assertEqual(aliases[1], ("Entity3", ["valid2"]))

    def test_empty_input(self):
        """Test empty input handling"""
        aliases = parse_alias("")
        self.assertEqual(len(aliases), 0)

    def test_malformed_input(self):
        """Test handling of malformed input"""
        text = """
        <Incomplete>
        <NoAliasList, >
        <Valid, ["alias"]>
        """
        aliases = parse_alias(text)

        self.assertEqual(len(aliases), 1)
        self.assertEqual(aliases[0], ("Valid", ["alias"]))

    def test_whitespace_handling(self):
        """Test handling of various whitespace patterns"""
        text = """
        <Entity1,["a1"]>
        < Entity2 , ["a2"] >
        """
        aliases = parse_alias(text)

        self.assertEqual(len(aliases), 2)
        self.assertEqual(aliases[0], ("Entity1", ["a1"]))
        self.assertEqual(aliases[1], ("Entity2", ["a2"]))


class TestImageDescJSONParse(unittest.TestCase):
    def setUp(self):
        """Set up test cases with sample inputs"""
        self.valid_json = """
        {
            "caption": "Figure 1: Test Graph",
            "text_snippets": ["x-axis", "y-axis", "data points"],
            "description": "A graph showing test data"
        }
        """

        self.missing_fields_json = """
        {
            "caption": "Figure 1: Test Graph"
        }
        """

        self.empty_json = "{}"

    def test_valid_json(self):
        """Test parsing valid JSON input"""
        caption, snippets, desc = parse_image_description(self.valid_json)
        self.assertEqual(caption, "Figure 1: Test Graph")
        self.assertEqual(snippets, ["x-axis", "y-axis", "data points"])
        self.assertEqual(desc, "A graph showing test data")

    def test_missing_fields(self):
        """Test parsing JSON with missing fields"""
        caption, snippets, desc = parse_image_description(self.missing_fields_json)
        self.assertEqual(caption, "Figure 1: Test Graph")
        self.assertEqual(snippets, [])  # Default empty list
        self.assertEqual(desc, "")  # Default empty string

    def test_empty_json(self):
        """Test parsing empty JSON"""
        caption, snippets, desc = parse_image_description(self.empty_json)
        self.assertEqual(caption, "")
        self.assertEqual(snippets, [])
        self.assertEqual(desc, "")

    def test_invalid_json(self):
        """Test parsing invalid JSON"""
        invalid_json = "{invalid json}"
        caption, snippets, desc = parse_image_description(invalid_json)
        self.assertEqual(caption, "")
        self.assertEqual(snippets, [])
        self.assertEqual(desc, "")

    def test_empty_input(self):
        """Test parsing empty input"""
        caption, snippets, desc = parse_image_description("")
        self.assertEqual(caption, "")
        self.assertEqual(snippets, [])
        self.assertEqual(desc, "")

    def test_json_with_extra_text(self):
        """Test parsing JSON with surrounding text"""
        text = """
        Some text before
        {
            "caption": "Figure 1",
            "text_snippets": ["text1"],
            "description": "desc1"
        }
        Some text after
        """
        caption, snippets, desc = parse_image_description(text)
        self.assertEqual(caption, "Figure 1")
        self.assertEqual(snippets, ["text1"])
        self.assertEqual(desc, "desc1")


class TestJsonListParse(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        with open("tests/assets/json_list_test_txt.txt", "r", encoding="utf-8") as file:
            self.test_text = file.read()

    def test_valid_json_list(self):
        """Test parsing valid JSON list"""
        result = parse_json_list(self.test_text.split("\n\n")[0])  # Get first json list
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "John")

    def test_json_list_with_fields(self):
        """Test parsing JSON list with specific fields"""
        fields = ["name", "age"]
        result = parse_json_list(self.test_text.split("\n\n")[0], fields)
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result[0].keys()), set(fields))
        self.assertEqual(result[1]["name"], "Alice")
        self.assertEqual(result[1]["age"], 25)

    def test_json_list_with_surrounding_text(self):
        """Test parsing JSON list with text before and after"""
        result = parse_json_list(
            self.test_text.split("\n\n")[1]
        )  # Get second json list
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["field1"], "value1")
        self.assertEqual(result[1]["field1"], "value2")

    def test_incomplete_json_list(self):
        """Test parsing incomplete JSON list"""
        result = parse_json_list(self.test_text.split("\n\n")[2])  # Get third json list
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["incomplete"], "data")
        self.assertEqual(result[2], {})

    def test_invalid_json_list(self):
        """Test parsing invalid JSON list"""
        result = parse_json_list(
            self.test_text.split("\n\n")[3]
        )  # Get fourth json list
        self.assertEqual(result, [])

    def test_empty_json_list(self):
        """Test parsing empty JSON list"""
        result = parse_json_list(self.test_text.split("\n\n")[4])  # Get fifth json list
        self.assertEqual(result, [])


class ClassifyParseTest(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.direct_response = """
        {
            "classification": "direct",
            "analysis": "This is just analysis without answer",
            "response": "This is a direct answer to the query"
        }
        """

        self.keyword_response = """
        ["keyword1", "keyword2", "keyword3"]
        {
            "classification": "retrieval",
            "analysis": "The query requires external knowledge retrieval.",
            "keywords": ["keyword1", "keyword2", "keyword3"]
        }
        """

        self.keyword_response_with_text = """
        Here are the extracted keywords:
        ["machine learning", "neural networks", "deep learning"]
        {
            "classification": "retrieval",
            "analysis": "The query requires external knowledge retrieval.",
            "keywords": ["machine learning", "neural networks", "deep learning"]
        }
        """

        self.invalid_response = "Invalid response format"

    def test_direct_answer_parsing(self):
        """Test parsing direct answer format"""
        response_type, content = parse_classify_response(self.direct_response)
        self.assertEqual(response_type, "direct")
        self.assertEqual(content, "This is a direct answer to the query")

    def test_keyword_parsing(self):
        """Test parsing keyword extraction format"""
        response_type, content = parse_classify_response(self.keyword_response)
        self.assertEqual(response_type, "retrieval")
        self.assertEqual(content, ["keyword1", "keyword2", "keyword3"])

    def test_keyword_with_surrounding_text(self):
        """Test parsing keywords with surrounding text"""
        response_type, content = parse_classify_response(
            self.keyword_response_with_text
        )
        self.assertEqual(response_type, "retrieval")
        self.assertEqual(
            content, ["machine learning", "neural networks", "deep learning"]
        )

    def test_invalid_format(self):
        """Test handling invalid response format"""
        response_type, content = parse_classify_response(self.invalid_response)
        self.assertEqual(response_type, "retrieval")
        self.assertEqual(content, [])

    def test_empty_response(self):
        """Test handling empty response"""
        response_type, content = parse_classify_response("")
        self.assertEqual(response_type, "retrieval")
        self.assertEqual(content, [])

    def test_partial_direct_response(self):
        """Test handling partial direct response format"""
        partial_response = "Analysis: This is just analysis without answer"
        response_type, content = parse_classify_response(partial_response)
        self.assertEqual(response_type, "retrieval")
        self.assertEqual(content, [])
