import unittest
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from lib_neutral_prompt import perp_parser


class TestMaliciousPromptParser(unittest.TestCase):
    def setUp(self):
        self.parser = perp_parser

    def test_zero_weight(self):
        result = self.parser.parse_root("hello :0.0")
        self.assertEqual(result.children[0].weight, 0.0)

    def test_mixed_positive_and_negative_weights(self):
        result = self.parser.parse_root("hello :1.0 AND goodbye :-2.0")
        self.assertEqual(result.children[0].weight, 1.0)
        self.assertEqual(result.children[1].weight, -2.0)

    def test_erroneous_syntax(self):
        result = self.parser.parse_root("hello :1.0 AND_PERP [goodbye :2.0")
        self.assertEqual(result.children[0].weight, 1.0)
        self.assertEqual(result.children[1].children[0].prompt, "goodbye ")
        self.assertEqual(result.children[1].children[0].weight, 2.0)

        result = self.parser.parse_root("hello :1.0 AND_PERP goodbye :2.0]")
        self.assertEqual(result.children[0].weight, 1.0)
        self.assertEqual(result.children[1].children[0].prompt, " goodbye ")

        result = self.parser.parse_root("hello :1.0 AND_PERP AND goodbye :2.0")
        self.assertEqual(result.children[0].weight, 1.0)
        self.assertEqual(result.children[2].prompt, " goodbye ")

    def test_huge_number_of_prompt_parts(self):
        result = self.parser.parse_root(" AND ".join(f"hello{i} :{i}" for i in range(10**4)))
        self.assertEqual(len(result.children), 10**4)

    def test_prompt_ending_with_weight(self):
        result = self.parser.parse_root("hello :1.0 AND :2.0")
        self.assertEqual(result.children[0].weight, 1.0)
        self.assertEqual(result.children[1].prompt, "")
        self.assertEqual(result.children[1].weight, 2.0)

    def test_huge_input_string(self):
        big_string = "hello :1.0 AND " * 10**4
        result = self.parser.parse_root(big_string)
        self.assertEqual(len(result.children), 10**4 + 1)

    def test_deeply_nested_prompt(self):
        deeply_nested_prompt = "hello :1.0" + " AND_PERP [goodbye :2.0" * 100 + "]" * 100
        result = self.parser.parse_root(deeply_nested_prompt)
        self.assertIsInstance(result.children[1], perp_parser.CompositePrompt)

    def test_complex_nested_prompts(self):
        complex_prompt = "hello :1.0 AND goodbye :2.0 AND_PERP [welcome :3.0 AND farewell :4.0 AND_PERP [greetings :5.0]]"
        result = self.parser.parse_root(complex_prompt)
        self.assertEqual(result.children[0].weight, 1.0)
        self.assertEqual(result.children[1].weight, 2.0)
        self.assertEqual(result.children[2].children[0].weight, 3.0)
        self.assertEqual(result.children[2].children[1].weight, 4.0)
        self.assertEqual(result.children[2].children[2].children[0].weight, 5.0)


if __name__ == '__main__':
    unittest.main()
