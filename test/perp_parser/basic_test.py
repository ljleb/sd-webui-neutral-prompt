import unittest
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from lib_neutral_prompt import neutral_prompt_parser


class TestPromptParser(unittest.TestCase):
    def setUp(self):
        self.simple_prompt = neutral_prompt_parser.parse_root("hello :1.0")
        self.and_prompt = neutral_prompt_parser.parse_root("hello AND goodbye :2.0")
        self.and_perp_prompt = neutral_prompt_parser.parse_root("hello :1.0 AND_PERP goodbye :2.0")
        self.and_salt_prompt = neutral_prompt_parser.parse_root("hello :1.0 AND_SALT goodbye :2.0")
        self.nested_and_perp_prompt = neutral_prompt_parser.parse_root("hello :1.0 AND_PERP [goodbye :2.0 AND_PERP welcome :3.0]")
        self.nested_and_salt_prompt = neutral_prompt_parser.parse_root("hello :1.0 AND_SALT [goodbye :2.0 AND_SALT welcome :3.0]")
        self.invalid_weight = neutral_prompt_parser.parse_root("hello :not_a_float")

    def test_simple_prompt_child_count(self):
        self.assertEqual(len(self.simple_prompt.children), 1)

    def test_simple_prompt_child_weight(self):
        self.assertEqual(self.simple_prompt.children[0].weight, 1.0)

    def test_simple_prompt_child_prompt(self):
        self.assertEqual(self.simple_prompt.children[0].prompt, "hello ")

    def test_square_weight_prompt(self):
        prompt = "a [b c d e : f g h :1.5]"
        parsed = neutral_prompt_parser.parse_root(prompt)
        self.assertEqual(parsed.children[0].prompt, prompt)

        composed_prompt = f"{prompt} AND_PERP other prompt"
        parsed = neutral_prompt_parser.parse_root(composed_prompt)
        self.assertEqual(parsed.children[0].prompt, prompt)

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
        self.assertIsInstance(self.and_perp_prompt.children[0], neutral_prompt_parser.LeafPrompt)
        self.assertIsInstance(self.and_perp_prompt.children[1], neutral_prompt_parser.LeafPrompt)

    def test_and_perp_prompt_nested_child(self):
        nested_child = self.and_perp_prompt.children[1]
        self.assertEqual(nested_child.weight, 2.0)
        self.assertEqual(nested_child.prompt.strip(), "goodbye")

    def test_nested_and_perp_prompt_child_count(self):
        self.assertEqual(len(self.nested_and_perp_prompt.children), 2)

    def test_nested_and_perp_prompt_child_types(self):
        self.assertIsInstance(self.nested_and_perp_prompt.children[0], neutral_prompt_parser.LeafPrompt)
        self.assertIsInstance(self.nested_and_perp_prompt.children[1], neutral_prompt_parser.CompositePrompt)

    def test_nested_and_perp_prompt_nested_child_types(self):
        nested_child = self.nested_and_perp_prompt.children[1].children[0]
        self.assertIsInstance(nested_child, neutral_prompt_parser.LeafPrompt)
        nested_child = self.nested_and_perp_prompt.children[1].children[1]
        self.assertIsInstance(nested_child, neutral_prompt_parser.LeafPrompt)

    def test_nested_and_perp_prompt_nested_child(self):
        nested_child = self.nested_and_perp_prompt.children[1].children[1]
        self.assertEqual(nested_child.weight, 3.0)
        self.assertEqual(nested_child.prompt.strip(), "welcome")

    def test_invalid_weight_child_count(self):
        self.assertEqual(len(self.invalid_weight.children), 1)

    def test_invalid_weight_child_weight(self):
        self.assertEqual(self.invalid_weight.children[0].weight, 1.0)

    def test_invalid_weight_child_prompt(self):
        self.assertEqual(self.invalid_weight.children[0].prompt, "hello :not_a_float")

    def test_and_salt_prompt_child_count(self):
        self.assertEqual(len(self.and_salt_prompt.children), 2)

    def test_and_salt_prompt_child_types(self):
        self.assertIsInstance(self.and_salt_prompt.children[0], neutral_prompt_parser.LeafPrompt)
        self.assertIsInstance(self.and_salt_prompt.children[1], neutral_prompt_parser.LeafPrompt)

    def test_and_salt_prompt_nested_child(self):
        nested_child = self.and_salt_prompt.children[1]
        self.assertEqual(nested_child.weight, 2.0)
        self.assertEqual(nested_child.prompt.strip(), "goodbye")

    def test_nested_and_salt_prompt_child_count(self):
        self.assertEqual(len(self.nested_and_salt_prompt.children), 2)

    def test_nested_and_salt_prompt_child_types(self):
        self.assertIsInstance(self.nested_and_salt_prompt.children[0], neutral_prompt_parser.LeafPrompt)
        self.assertIsInstance(self.nested_and_salt_prompt.children[1], neutral_prompt_parser.CompositePrompt)

    def test_nested_and_salt_prompt_nested_child_types(self):
        nested_child = self.nested_and_salt_prompt.children[1].children[0]
        self.assertIsInstance(nested_child, neutral_prompt_parser.LeafPrompt)
        nested_child = self.nested_and_salt_prompt.children[1].children[1]
        self.assertIsInstance(nested_child, neutral_prompt_parser.LeafPrompt)

    def test_nested_and_salt_prompt_nested_child(self):
        nested_child = self.nested_and_salt_prompt.children[1].children[1]
        self.assertEqual(nested_child.weight, 3.0)
        self.assertEqual(nested_child.prompt.strip(), "welcome")

    def test_start_with_prompt_editing(self):
        prompt = "[(long shot:1.2):0.1] detail.."
        res = neutral_prompt_parser.parse_root(prompt)
        self.assertEqual(res.children[0].weight, 1.0)
        self.assertEqual(res.children[0].prompt, prompt)


if __name__ == '__main__':
    unittest.main()
