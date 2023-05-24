import unittest
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from lib_neutral_prompt import perp_parser


class TestPromptParser(unittest.TestCase):
    def setUp(self):
        self.simple_prompt = perp_parser.parse_root("hello :1.0")
        self.and_prompt = perp_parser.parse_root("hello AND goodbye :2.0")
        self.and_perp_prompt = perp_parser.parse_root("hello :1.0 AND_PERP [goodbye :2.0]")
        self.nested_and_perp_prompt = perp_parser.parse_root("hello :1.0 AND_PERP [goodbye :2.0 AND_PERP [welcome :3.0]]")
        self.invalid_weight = perp_parser.parse_root("hello :not_a_float")

    def test_simple_prompt_child_count(self):
        self.assertEqual(len(self.simple_prompt.children), 1)

    def test_simple_prompt_child_weight(self):
        self.assertEqual(self.simple_prompt.children[0].weight, 1.0)

    def test_simple_prompt_child_prompt(self):
        self.assertEqual(self.simple_prompt.children[0].prompt, "hello ")

    def test_and_prompt_child_count(self):
        self.assertEqual(len(self.and_prompt.children), 2)

    def test_and_prompt_child_weights_and_prompts(self):
        self.assertEqual(self.and_prompt.children[0].weight, 1.0)
        self.assertEqual(self.and_prompt.children[0].prompt, "hello ")
        self.assertEqual(self.and_prompt.children[1].weight, 2.0)
        self.assertEqual(self.and_prompt.children[1].prompt, " goodbye ")

    def test_and_perp_prompt_child_count(self):
        self.assertEqual(len(self.and_perp_prompt.children), 2)

    def test_and_perp_prompt_child_types(self):
        self.assertIsInstance(self.and_perp_prompt.children[0], perp_parser.ComposablePrompt)
        self.assertIsInstance(self.and_perp_prompt.children[1], perp_parser.CompositePrompt)

    def test_and_perp_prompt_nested_child(self):
        nested_child = self.and_perp_prompt.children[1].children[0]
        self.assertEqual(nested_child.weight, 2.0)
        self.assertEqual(nested_child.prompt, "goodbye ")

    def test_nested_and_perp_prompt_child_count(self):
        self.assertEqual(len(self.nested_and_perp_prompt.children), 2)

    def test_nested_and_perp_prompt_child_types(self):
        self.assertIsInstance(self.nested_and_perp_prompt.children[0], perp_parser.ComposablePrompt)
        self.assertIsInstance(self.nested_and_perp_prompt.children[1], perp_parser.CompositePrompt)

    def test_nested_and_perp_prompt_nested_child_types(self):
        nested_child = self.nested_and_perp_prompt.children[1].children[0]
        self.assertIsInstance(nested_child, perp_parser.ComposablePrompt)
        nested_child = self.nested_and_perp_prompt.children[1].children[1]
        self.assertIsInstance(nested_child, perp_parser.CompositePrompt)

    def test_nested_and_perp_prompt_nested_child(self):
        nested_child = self.nested_and_perp_prompt.children[1].children[1].children[0]
        self.assertEqual(nested_child.weight, 3.0)
        self.assertEqual(nested_child.prompt, "welcome ")

    def test_invalid_weight_child_count(self):
        self.assertEqual(len(self.invalid_weight.children), 1)

    def test_invalid_weight_child_weight(self):
        self.assertEqual(self.invalid_weight.children[0].weight, 1.0)

    def test_invalid_weight_child_prompt(self):
        self.assertEqual(self.invalid_weight.children[0].prompt, "hello :not_a_float")


if __name__ == '__main__':
    unittest.main()
