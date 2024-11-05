import unittest
from unittest.mock import patch

from pure_NL_experiments.segmentation import AtomicSegmentation, SentenceSegmentation


class TestSentenceSegmentation(unittest.TestCase):
    def test_sentence_segmentation_simple(self):
        problem_text = "Prove that the sum of two odd numbers is even."
        proof_text = "Let a and b be odd integers. Then a = 2k + 1 and b = 2m + 1. Therefore, a + b = 2(k + m + 1), which is even."
        segmenter = SentenceSegmentation(problem_text, proof_text)
        expected_segments = [
            "Let a and b be odd integers.",
            "Then a = 2k + 1 and b = 2m + 1.",
            "Therefore, a + b = 2(k + m + 1), which is even."
        ]
        self.assertEqual(segmenter.segment(), expected_segments)

    def test_sentence_segmentation_no_punctuation(self):
        problem_text = "Prove that any integer multiplied by zero is zero."
        proof_text = "Let n be any integer Multiplying n by zero gives zero"
        segmenter = SentenceSegmentation(problem_text, proof_text)
        expected_segments = [
            "Let n be any integer Multiplying n by zero gives zero"]
        self.assertEqual(segmenter.segment(), expected_segments)

    def test_sentence_segmentation_empty_string(self):
        problem_text = "Prove that zero is zero."
        proof_text = ""
        segmenter = SentenceSegmentation(problem_text, proof_text)
        expected_segments = []
        self.assertEqual(segmenter.segment(), expected_segments)

    def test_sentence_segmentation_whitespace_string(self):
        problem_text = "Prove that one equals one."
        proof_text = "   "
        segmenter = SentenceSegmentation(problem_text, proof_text)
        expected_segments = []
        self.assertEqual(segmenter.segment(), expected_segments)


class TestAtomicSegmentation(unittest.TestCase):
    @patch('segmentation.OpenAI')
    def test_atomic_segmentation_basic(self, mock_openai):
        problem_text = "Prove that the sum of two even numbers is even."
        proof_text = "Let a and b be even numbers. Then a = 2k and b = 2m for some integers k and m. Therefore, a + b = 2(k + m), which is even."

        # Mock the OpenAI client and its response
        mock_client_instance = mock_openai.return_value
        mock_response = type('obj', (object,), {
            'choices': [
                type('obj', (object,), {
                    'message': type('obj', (object,), {
                        'content': (
                            "1. Let a and b be even numbers.\n"
                            "2. Then a = 2k for some integer k.\n"
                            "3. And b = 2m for some integer m.\n"
                            "4. Therefore, a + b = 2k + 2m.\n"
                            "5. Simplify to a + b = 2(k + m).\n"
                            "6. Since k + m is an integer, a + b is even."
                        )
                    })
                })
            ]
        })
        mock_client_instance.chat.completions.create.return_value = mock_response

        segmenter = AtomicSegmentation(problem_text, proof_text)
        expected_segments = [
            "Let a and b be even numbers.",
            "Then a = 2k for some integer k.",
            "And b = 2m for some integer m.",
            "Therefore, a + b = 2k + 2m.",
            "Simplify to a + b = 2(k + m).",
            "Since k + m is an integer, a + b is even."
        ]
        self.assertEqual(segmenter.segment(), expected_segments)

    @patch('segmentation.OpenAI')
    def test_atomic_segmentation_complex(self, mock_openai):
        problem_text = "Prove that every bounded sequence has a convergent subsequence."
        proof_text = "Let (a_n) be a bounded sequence. By the Bolzano-Weierstrass theorem, there exists a convergent subsequence of (a_n)."

        # Mock the OpenAI client and its response
        mock_client_instance = mock_openai.return_value
        mock_response = type('obj', (object,), {
            'choices': [
                type('obj', (object,), {
                    'message': type('obj', (object,), {
                        'content': (
                            "1. Let (a_n) be a bounded sequence.\n"
                            "2. By the Bolzano-Weierstrass theorem, any bounded sequence has a convergent subsequence.\n"
                            "3. Therefore, (a_n) has a convergent subsequence."
                        )
                    })
                })
            ]
        })
        mock_client_instance.chat.completions.create.return_value = mock_response

        segmenter = AtomicSegmentation(problem_text, proof_text)
        expected_segments = [
            "Let (a_n) be a bounded sequence.",
            "By the Bolzano-Weierstrass theorem, any bounded sequence has a convergent subsequence.",
            "Therefore, (a_n) has a convergent subsequence."
        ]
        self.assertEqual(segmenter.segment(), expected_segments)

    @patch('segmentation.OpenAI')
    def test_atomic_segmentation_empty(self, mock_openai):
        problem_text = "Prove that zero equals zero."
        proof_text = ""

        # Mock the OpenAI client and its response
        mock_client_instance = mock_openai.return_value
        mock_response = type('obj', (object,), {
            'choices': [
                type('obj', (object,), {
                    'message': type('obj', (object,), {
                        'content': ""
                    })
                })
            ]
        })
        mock_client_instance.chat.completions.create.return_value = mock_response

        segmenter = AtomicSegmentation(problem_text, proof_text)
        expected_segments = []
        self.assertEqual(segmenter.segment(), expected_segments)

    @patch('segmentation.OpenAI')
    def test_atomic_segmentation_non_numbered(self, mock_openai):
        problem_text = "Prove that if n is odd, then n^2 is odd."
        proof_text = "Suppose n is odd. Then n = 2k + 1 for some integer k. Therefore, n^2 = 4k^2 + 4k + 1, which is odd."

        # Mock the OpenAI client and its response
        mock_client_instance = mock_openai.return_value
        mock_response = type('obj', (object,), {
            'choices': [
                type('obj', (object,), {
                    'message': type('obj', (object,), {
                        'content': (
                            "1. Suppose n is odd.\n"
                            "2. Then n = 2k + 1 for some integer k.\n"
                            "3. Calculate n^2 = (2k + 1)^2.\n"
                            "4. Expand to get n^2 = 4k^2 + 4k + 1.\n"
                            "5. Since 4k^2 + 4k is even, n^2 is odd."
                        )
                    })
                })
            ]
        })
        mock_client_instance.chat.completions.create.return_value = mock_response

        segmenter = AtomicSegmentation(problem_text, proof_text)
        expected_segments = [
            "Suppose n is odd.",
            "Then n = 2k + 1 for some integer k.",
            "Calculate n^2 = (2k + 1)^2.",
            "Expand to get n^2 = 4k^2 + 4k + 1.",
            "Since 4k^2 + 4k is even, n^2 is odd."
        ]
        self.assertEqual(segmenter.segment(), expected_segments)


if __name__ == '__main__':
    unittest.main()
