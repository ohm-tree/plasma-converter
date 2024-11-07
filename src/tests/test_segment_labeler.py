import unittest
from unittest.mock import patch

from src.NL.segment_labeler import BasicSegmentLabeler


class TestBasicSegmentLabeler(unittest.TestCase):
    def setUp(self):
        self.problem_statement = "Prove that the square of an even number is even."
        self.segments = [
            "Let n be an even number.",
            "Then n = 2k for some integer k.",
            "Therefore, n^2 = (2k)^2 = 4k^2.",
            "Thus, n^2 is divisible by 4 and hence even.",
            "Because n is even, n^2 is also even.",
            "Consider the function f(x) = x^2.",
            "We have f(n) = n^2.",
            "Since n is even, f(n) is even."
        ]

    def test_label_segment_with_keyword(self):
        # Test segments that contain deduction keywords
        indices_with_deduction = [1, 2, 3, 4, 7]
        for index in indices_with_deduction:
            labeler = BasicSegmentLabeler(
                self.problem_statement, self.segments, index)
            label = labeler.label_segment()
            self.assertEqual(
                label, 'deduction', f"Segment {index+1} should be labeled as 'deduction'.")

    @patch('src.NL.segment_labeler.chat_gpt')
    def test_label_segment_without_keyword(self, mock_chat_gpt):
        # Test segments without deduction keywords
        indices_without_deduction = [0, 5, 6]
        # Mock GPT to return 'proposition'
        mock_chat_gpt.return_value = 'Proposition'
        for index in indices_without_deduction:
            labeler = BasicSegmentLabeler(
                self.problem_statement, self.segments, index)
            label = labeler.label_segment()
            self.assertEqual(
                label, 'proposition', f"Segment {index+1} should be labeled as 'proposition'.")
            # Ensure that GPT was called with the correct prompt containing context
            mock_chat_gpt.assert_called()
            last_call_args = mock_chat_gpt.call_args[0][0]
            self.assertIn('Given the context:', last_call_args)
            mock_chat_gpt.reset_mock()

    @patch('src.NL.segment_labeler.chat_gpt')
    def test_label_segment_gpt_deduction(self, mock_chat_gpt):
        # Test when GPT determines it's a deduction
        mock_chat_gpt.return_value = 'Deduction'
        index = 5  # "Consider the function f(x) = x^2."
        labeler = BasicSegmentLabeler(
            self.problem_statement, self.segments, index)
        label = labeler.label_segment()
        self.assertEqual(label, 'deduction',
                         f"Segment {index+1} should be labeled as 'deduction'.")
        mock_chat_gpt.assert_called_once()

    @patch('src.NL.segment_labeler.chat_gpt')
    def test_label_segment_gpt_proposition(self, mock_chat_gpt):
        # Test when GPT determines it's a proposition
        mock_chat_gpt.return_value = 'Proposition'
        index = 0  # "Let n be an even number."
        labeler = BasicSegmentLabeler(
            self.problem_statement, self.segments, index)
        label = labeler.label_segment()
        self.assertEqual(
            label, 'proposition', f"Segment {index+1} should be labeled as 'proposition'.")
        mock_chat_gpt.assert_called_once()


if __name__ == '__main__':
    unittest.main()
