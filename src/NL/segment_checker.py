# segment_checker.py

import os
import random
from abc import ABC, abstractmethod
from typing import Optional

from openai import OpenAI


class SegmentChecker(ABC):
    """
    Abstract base class for checking a single proof segment.
    """

    def __init__(self, problem_statement: str, segments: list[str], segment_index: int):
        self.problem_statement = problem_statement
        self.segments = segments
        self.segment_index = segment_index
        self.explanation = None

    @abstractmethod
    def check_segment(self) -> bool:
        """
        Checks the segment and returns True if it is correct, False otherwise.
        """
        pass


class ScrambleVerifier(SegmentChecker):
    """
    Verifies a segment by generating scrambled versions and performing multiple-choice evaluation.
    """

    def __init__(
        self,
        problem_statement: str,
        segments: list[str],
        segment_index: int,
        num_scrambles: int = 3,
        num_multiple_choice_repeats: int = 5,
        sample_n: int = 100,
        accept_threshold: float = 0.5,
        client: Optional[OpenAI] = None,
        verbose: bool = False
    ):
        super().__init__(problem_statement, segments, segment_index)
        self.num_scrambles = num_scrambles
        self.num_multiple_choice_repeats = num_multiple_choice_repeats
        self.sample_n = sample_n
        if client is None:
            self.client = OpenAI()
        else:
            self.client = client
        self.accept_threshold = accept_threshold
        self.verbose = verbose

    def generate_scrambled_versions(self, context: str, segment: str) -> list[str]:
        """
        Generates scrambled versions of a segment.
        """
        prompt = (
            f"Given the problem statement:\n{self.problem_statement}\n\n"
            f"Given the context:\n{context}\n\n"
            f"And the next correct segment:\n{segment}\n\n"
            f"Generate {self.num_scrambles} variations of the next segment that are similar but contain a subtle mathematical error."
            " Provide the variations as a numbered list."
        )
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content.strip()
        scrambled_segments = self.parse_gpt_variations(content)
        return scrambled_segments[:self.num_scrambles]

    def parse_gpt_variations(self, content: str) -> list[str]:
        """
        Parses the GPT response to extract the list of scrambled segments.
        """
        import re
        variations = []
        lines = content.split('\n')
        for line in lines:
            # Match lines that start with a number followed by a period
            match = re.match(r'^\d+\.\s*(.*)', line)
            if match:
                variation = match.group(1).strip()
                if variation:
                    variations.append(variation)
        if self.verbose:
            print(f"Scrambled segments: {variations}")
        return variations

    def _generate_multiple_choice_prompt(self, context: str, options_text: str) -> list[dict]:
        """
        Generates the message history for multiple choice evaluation with few-shot examples.
        """
        few_shot_messages = [
            {"role": "user", "content":
                "Given the problem statement:\n"
                "Prove that the sum of two even numbers is even.\n\n"
                "Given the context:\n"
                "Let n and m be two even numbers. Then $n = 2a$ and $m = 2b$ for some integers $a$ and $b$.\n\n"
                "Which of the following options logically follows from the context?\n"
                "1. Thus, $n + m = 2a + 2b = 2(a + b)$, which is even.\n"
                "2. Thus, $n + m = 2a + 2b = 2(a - b)$, which is even.\n"
                "3. Thus, $n + m = 2a + 2b = 4(a + b)$, which is even.\n"
                "4. None of the above.\n"
                "Please provide the number of the correct option."
             },
            {"role": "assistant", "content": "1"},
            {"role": "user", "content":
                "Given the problem statement:\n"
                "Evaluate the integral of $\\frac{1}{x} dx$.\n\n"
                "Given the context:\n"
                "We need to find the antiderivative of $\\frac{1}{x}$.\n\n"
                "Which of the following options logically follows from the context?\n"
                "1. Therefore, $\\int \\frac{1}{x} dx = -\\frac{1}{x^2} + C$\n"
                "2. Therefore, $\\int \\frac{1}{x} dx = \\frac{1}{x} + C$\n"
                "3. Therefore, $\\int \\frac{1}{x} dx = x + C$\n"
                "4. None of the above.\n"
                "Please provide the number of the correct option."
             },
            {"role": "assistant", "content": "4"},
        ]

        if context.strip() != "":
            content = (
                f"Given the problem statement:\n{self.problem_statement}\n\n"
                f"Given the context:\n{context}\n\n"
                f"Which of the following options logically follows from the context?\n{options_text}\n"
                "Please say ONLY the number of the correct option and NOTHING ELSE!"
            )
        else:
            content = (
                f"Given the problem statement:\n{self.problem_statement}\n\n"
                f"Which of the following options logically follows from the problem statement?\n{options_text}\n"
                "Please say ONLY the number of the correct option and NOTHING ELSE!"
            )

        return few_shot_messages + [{"role": "user", "content": content}]

    def multiple_choice_evaluation(self, context: str, options: list[str]) -> dict:
        """
        Presents multiple choices to the model and asks which one logically follows.

        Parameters:
        ----------
        context: str
            The context of the problem statement.
        options: list[str]
            The options to choose from.

        Returns:
        -------
        dict
            Dictionary containing:
            - 'frequencies': list of frequencies for each option
            - 'none_of_above': count of "None of the above" selections
            - 'parse_errors': count of responses that couldn't be parsed
        """
        frequencies = [0] * len(options)
        none_of_above_count = 0
        parse_errors = 0

        for _ in range(self.num_multiple_choice_repeats):
            permutation = list(range(len(options)))
            random.shuffle(permutation)
            shuffled_options = [options[i] for i in permutation]

            # Add "None of the above" as the last option
            all_options = shuffled_options + ["None of the above"]

            options_with_numbers = [
                f"{idx + 1}. {option}" for idx, option in enumerate(all_options)]
            options_text = '\n'.join(options_with_numbers)

            messages = self._generate_multiple_choice_prompt(
                context, options_text)

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                n=self.sample_n,
                max_tokens=3  # Only need to return the number
            )
            for choice in response.choices:
                response_text = choice.message.content.strip()
                # if the response text contains a period,
                # truncate to everything before it;
                # the model likely output "x. Some text"
                # where x is the number we want
                if '.' in response_text:
                    response_text = response_text.split('.')[0]
                try:
                    response_idx = int(response_text) - 1
                    # If "None of the above" was chosen
                    if response_idx == len(all_options) - 1:
                        none_of_above_count += 1
                    else:
                        frequencies[permutation[response_idx]] += 1
                except ValueError:
                    print(f"Error parsing response: {response_text}")
                    parse_errors += 1
                    continue

        return {
            'frequencies': frequencies,
            'none_of_above': none_of_above_count,
            'parse_errors': parse_errors
        }

    def explain_error(self, context: str, segment: str):
        """
        Asks the model to explain why the proof does not work at the error segment.
        """
        prompt = (
            f"There is an error in the following proof segment:\n{segment}\n\n"
            f"Given the problem statement:\n{self.problem_statement}\n\n"
            f"Given the context:\n{context}\n\n"
            "Please provide a detailed explanation of why this segment is incorrect."
        )
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        self.explanation = response.choices[0].message.content.strip()

    def check_segment(self) -> bool:
        """
        Checks the segment and returns True if it is correct, False otherwise.
        """
        context = ' '.join(self.segments[:self.segment_index])
        correct_segment = self.segments[self.segment_index]
        scrambled_segments = self.generate_scrambled_versions(
            context, correct_segment)
        options = [correct_segment] + scrambled_segments
        results = self.multiple_choice_evaluation(context, options)
        if self.verbose:
            print(f"Results: {results}")
        # Determine if the correct segment was chosen majority of the time
        # The correct segment is always the index 0
        correct_votes = results['frequencies'][0]
        if correct_votes / self.num_multiple_choice_repeats >= self.accept_threshold:
            # Segment is verified
            return True
        else:
            # Segment is incorrect
            self.explain_error(context, correct_segment)
            return False


def main():
    problem_statement = "Prove that the sum of two even numbers is even."
    segments = [
        "Let n and m be two even numbers.",
        "Then $n = 2a$ and $m = 2b$ for some integers $a$ and $b$.",
        "Thus, $n + m = 2a + 2b = 2(a + b)$, which is even."
    ]
    print("Sum of Two Even Numbers")
    for i in [1, 2]:  # Segment 0 is a proposition, and hence should be skipped
        verifier = ScrambleVerifier(
            problem_statement, segments, i, accept_threshold=0.5, verbose=True)
        result = verifier.check_segment()
        print(f"Segment {i+1}: {'Correct' if result else 'Incorrect'}")
        if not result:
            print(f"Explanation: {verifier.explanation}\n")
    print("\n")

    # Incorrect Calculus Example
    problem_statement_calc = "Evaluate the integral of $\\frac{1}{x} dx$."
    segments_calc = [
        "We know that the integral of $\\frac{1}{x} dx$ is $-\\frac{1}{x^2} + C$.",
        "Therefore, $\\int \\frac{1}{x} dx = -\\frac{1}{x^2} + C$."
    ]
    print("Integral of 1/x (Calculus)")
    verifier = ScrambleVerifier(
        problem_statement_calc, segments_calc, 0, accept_threshold=0.5, verbose=True)
    result = verifier.check_segment()
    print(f"Segment 1: {'Correct' if result else 'Incorrect'}")
    if not result:
        print(f"Explanation: {verifier.explanation}\n")
    print("\n")

    # Incorrect Number Theory Example
    problem_statement_nt = "Prove that the sum of any two odd numbers is even."
    segments_nt = [
        "Let $m$ and $n$ be odd numbers.",
        "Then $m = 2a + 1$ and $n = 2b + 1$ for some integers $a$ and $b$.",
        "Therefore, $m + n = 2a + 1 + 2b + 1 = 2(a + b + 1)$.",
        "Thus, $m + n$ is odd."
    ]
    print("Sum of Two Odd Numbers (Number Theory)")
    verifier = ScrambleVerifier(
        problem_statement_nt, segments_nt, 3, accept_threshold=0.5, verbose=True)
    result = verifier.check_segment()
    print(f"Segment 4: {'Correct' if result else 'Incorrect'}")
    print(f"Explanation: {verifier.explanation}\n")
    print("\n")


if __name__ == "__main__":
    main()
