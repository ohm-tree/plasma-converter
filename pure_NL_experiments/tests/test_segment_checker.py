import unittest
from unittest.mock import patch

from pure_NL_experiments.segment_checker import ScrambleVerifier


class TestScrambleVerifier(unittest.TestCase):
    def setUp(self):
        self.problem_statement = "Prove that the sum of two even numbers is even."
        self.segments = [
            "Let n and m be two even numbers.",
            "Then n = 2a and m = 2b for some integers a and b.",
            "Thus, n + m = 2a + 2b = 2(a + b), which is even."
        ]
        self.segment_index = 2
        self.verifier = ScrambleVerifier(
            self.problem_statement, self.segments, self.segment_index)

    @patch('segment_checker.chat_gpt')
    def test_generate_scrambled_versions(self, mock_chat_gpt):
        # Mock the GPT response for generate_scrambled_versions
        mock_chat_gpt.return_value = (
            "1. Thus, n + m = 2a + 2b = a + b, which is odd.\n"
            "2. Thus, n + m = 2a + 2b = 4(a + b), which is divisible by 4.\n"
            "3. Thus, n + m = 2a + 2b = 2(a - b), which may be even or odd."
        )
        context = ' '.join(self.segments[:self.segment_index])
        scrambled_segments = self.verifier.generate_scrambled_versions(
            context, self.segments[self.segment_index])
        expected_segments = [
            "Thus, n + m = 2a + 2b = a + b, which is odd.",
            "Thus, n + m = 2a + 2b = 4(a + b), which is divisible by 4.",
            "Thus, n + m = 2a + 2b = 2(a - b), which may be even or odd."
        ]
        self.assertEqual(scrambled_segments, expected_segments)

    @patch('segment_checker.chat_gpt')
    def test_parse_gpt_variations(self, mock_chat_gpt):
        # No need to mock chat_gpt here since we're testing parse_gpt_variations directly
        content = (
            "1. First variation.\n"
            "2. Second variation.\n"
            "3. Third variation."
        )
        variations = self.verifier.parse_gpt_variations(content)
        expected_variations = ["First variation.",
                               "Second variation.", "Third variation."]
        self.assertEqual(variations, expected_variations)

    @patch('segment_checker.chat_gpt')
    @patch('segment_checker.random.shuffle')
    def test_multiple_choice_evaluation(self, mock_shuffle, mock_chat_gpt):
        # Mock the shuffle function to do nothing
        mock_shuffle.side_effect = lambda x: None

        # Mock the GPT response for multiple_choice_evaluation
        mock_chat_gpt.return_value = '4'
        context = ' '.join(self.segments[:self.segment_index])
        options = [
            "Option 1.",
            "Option 2.",
            "Option 3.",
            self.segments[self.segment_index]
        ]
        choice, correct_option_number = self.verifier.multiple_choice_evaluation(
            context, options)
        self.assertEqual(choice, '4')
        self.assertEqual(correct_option_number, 4)

    @patch('segment_checker.chat_gpt')
    def test_explain_error(self, mock_chat_gpt):
        # Mock the GPT response for explain_error
        mock_chat_gpt.return_value = "The error is in the calculation step."
        context = ' '.join(self.segments[:self.segment_index])
        incorrect_segment = "Incorrect segment."
        self.verifier.explain_error(context, incorrect_segment)
        self.assertEqual(self.verifier.explanation,
                         "The error is in the calculation step.")

    @patch('segment_checker.chat_gpt')
    @patch('segment_checker.random.shuffle')
    def test_check_segment_correct(self, mock_shuffle, mock_chat_gpt):
        # Mock the shuffle function to do nothing
        mock_shuffle.side_effect = lambda x: None

        # Define the side effects for chat_gpt
        def side_effect(prompt):
            if "Generate" in prompt:
                return (
                    "1. Scrambled version 1.\n"
                    "2. Scrambled version 2.\n"
                    "3. Scrambled version 3."
                )
            elif "Which of the following options" in prompt:
                return '4'  # GPT selects the correct segment (position 4)
            return ''
        mock_chat_gpt.side_effect = side_effect

        result = self.verifier.check_segment()
        self.assertTrue(result)
        self.assertIsNone(self.verifier.explanation)

    @patch('segment_checker.chat_gpt')
    def test_check_segment_incorrect(self, mock_chat_gpt):
        # Mock the GPT responses for check_segment when the segment is incorrect
        def side_effect(prompt):
            if "Generate" in prompt:
                return (
                    "1. Scrambled version 1.\n"
                    "2. Scrambled version 2.\n"
                    "3. Scrambled version 3."
                )
            elif "Which of the following options" in prompt:
                return '1'  # GPT selects an incorrect segment
            elif "Please provide a detailed explanation" in prompt:
                return "Explanation of the error."
            return ''
        mock_chat_gpt.side_effect = side_effect
        # Introduce an error in the segment
        self.verifier.segments[self.segment_index] = "Incorrect segment."
        result = self.verifier.check_segment()
        self.assertFalse(result)
        self.assertEqual(self.verifier.explanation,
                         "Explanation of the error.")


if __name__ == '__main__':
    unittest.main()
