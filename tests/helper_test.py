import unittest
from src.mmkg_rag.utils.helper import md5, rename_markdown_images


class TestHelper(unittest.TestCase):
    def test_md5_empty_string(self):
        self.assertEqual(md5(""), "d41d8cd98f00b204e9800998ecf8427e")

    def test_md5_normal_strings(self):
        self.assertEqual(md5("test"), "098f6bcd4621d373cade4e832627b4f6")
        self.assertEqual(md5("hello"), "5d41402abc4b2a76b9719d911017c592")

    def test_md5_special_chars(self):
        self.assertEqual(md5("!@#$%^"), "c92b51b2f4d93d4e1081670bd9273402")
        self.assertEqual(md5("   "), "628631f07321b22d8c176c200c855e1b")

    def test_md5_numbers(self):
        self.assertEqual(md5("12345"), "827ccb0eea8a706c4c34a16891f84e7b")
        self.assertEqual(md5("0"), "cfcd208495d565ef66e7dff9f98764da")

    def test_rename_graphrag_md_iamges(self):
        res = rename_markdown_images("examples/rag/lightrag.md")
        self.assertGreater(len(res), 10)
