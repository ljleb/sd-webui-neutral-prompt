import unittest
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from lib_neutral_prompt import neutral_prompt_parser


class TestMaliciousPromptParser(unittest.TestCase):
    def setUp(self):
        self.parser = neutral_prompt_parser

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
        self.assertIsInstance(result.children[1], neutral_prompt_parser.PerpPrompt)

    def test_complex_nested_prompts(self):
        complex_prompt = "hello :1.0 AND goodbye :2.0 AND_PERP [welcome :3.0 AND farewell :4.0 AND_PERP [greetings :5.0]]"
        result = self.parser.parse_root(complex_prompt)
        self.assertEqual(result.children[0].weight, 1.0)
        self.assertEqual(result.children[1].weight, 2.0)
        self.assertEqual(result.children[2].children[0].weight, 3.0)
        self.assertEqual(result.children[2].children[1].weight, 4.0)
        self.assertEqual(result.children[2].children[2].children[0].weight, 5.0)

    def test_string_with_random_characters(self):
        random_chars = "ASDFGHJKL:@#$/.,|}{><~`12[3]456AND_PERP7890"
        try:
            self.parser.parse_root(random_chars)
        except Exception:
            self.fail("parse_root couldn't handle a string with random characters.")

    def test_string_with_unexpected_symbols(self):
        unexpected_symbols = "hello :1.0 AND $%^&*()goodbye :2.0"
        try:
            self.parser.parse_root(unexpected_symbols)
        except Exception:
            self.fail("parse_root couldn't handle a string with unexpected symbols.")

    def test_string_with_unconventional_structure(self):
        unconventional_structure = "hello :1.0 AND_PERP :2.0 AND [goodbye]"
        try:
            self.parser.parse_root(unconventional_structure)
        except Exception:
            self.fail("parse_root couldn't handle a string with unconventional structure.")

    def test_string_with_mixed_alphabets_and_numbers(self):
        mixed_alphabets_and_numbers = "123hello :1.0 AND goodbye456 :2.0"
        try:
            self.parser.parse_root(mixed_alphabets_and_numbers)
        except Exception:
            self.fail("parse_root couldn't handle a string with mixed alphabets and numbers.")

    def test_string_with_nested_brackets(self):
        nested_brackets = "hello :1.0 AND [goodbye :2.0 AND [[welcome :3.0]]]"
        try:
            self.parser.parse_root(nested_brackets)
        except Exception:
            self.fail("parse_root couldn't handle a string with nested brackets.")

    def test_unmatched_opening_braces(self):
        unmatched_opening_braces = "hello [[[[[[[[[ :1.0 AND_PERP goodbye :2.0"
        try:
            self.parser.parse_root(unmatched_opening_braces)
        except Exception:
            self.fail("parse_root couldn't handle a string with unmatched opening braces.")

    def test_unmatched_closing_braces(self):
        unmatched_closing_braces = "hello :1.0 AND_PERP goodbye ]]]]]]]]] :2.0"
        try:
            self.parser.parse_root(unmatched_closing_braces)
        except Exception:
            self.fail("parse_root couldn't handle a string with unmatched closing braces.")

    def test_repeating_colons(self):
        repeating_colons = "hello ::::::: :1.0 AND_PERP goodbye :::: :2.0"
        try:
            self.parser.parse_root(repeating_colons)
        except Exception:
            self.fail("parse_root couldn't handle a string with repeating colons.")

    def test_excessive_whitespace(self):
        excessive_whitespace = "hello    :1.0  AND_PERP     goodbye  :2.0"
        try:
            self.parser.parse_root(excessive_whitespace)
        except Exception:
            self.fail("parse_root couldn't handle a string with excessive whitespace.")

    def test_repeating_AND_keyword(self):
        repeating_AND_keyword = "hello :1.0 AND AND AND AND AND goodbye :2.0"
        try:
            self.parser.parse_root(repeating_AND_keyword)
        except Exception:
            self.fail("parse_root couldn't handle a string with repeating AND keyword.")

    def test_repeating_AND_PERP_keyword(self):
        repeating_AND_PERP_keyword = "hello :1.0 AND_PERP AND_PERP AND_PERP AND_PERP goodbye :2.0"
        try:
            self.parser.parse_root(repeating_AND_PERP_keyword)
        except Exception:
            self.fail("parse_root couldn't handle a string with repeating AND_PERP keyword.")


if __name__ == '__main__':
    unittest.main()
