# segment_checker.py

import os
import random
import re
from abc import ABC, abstractmethod
from typing import Optional

from openai import OpenAI

from src.utils.prompt import FewShotPrompter


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


def _selection_prompt(problem_statement: str, context: str, options_text: str) -> list[dict]:
    if context.strip() != "":
        content = (
            f"Given the problem statement:\n{problem_statement}\n\n"
            f"Given the context:\n{context}\n\n"
            f"Which of the following options logically follows from the context?\n{options_text}\n"
            "Please think carefully before answering. When you done, say \"Answer: <your answer>\"."
        )
    else:
        content = (
            f"Given the problem statement:\n{problem_statement}\n\n"
            f"Which of the following options logically follows from the problem statement?\n{options_text}\n"
            "Please think carefully before answering. When you done, say \"Answer: <your answer>\"."
        )
    return [{"role": "user", "content": content}]


selection_prompt = FewShotPrompter(
    system_prompt=(
        "You are a mathematical proof assistant. Your task is to "
        "identify which statement logically follows from the given context."
    ),
    function_prompt=_selection_prompt,
    few_shot_examples=[
        {
            "problem_statement": "Prove that the sum of two even numbers is even.",
            "context": "Let n and m be two even numbers. Then $n = 2a$ and $m = 2b$ for some integers $a$ and $b$.",
            "options_text": (
                "1. Thus, $n + m = 2a + 2b = 2(a + b)$, which is even.\n"
                "2. Thus, $n + m = 2a + 2b = 2(a - b)$, which is even.\n"
                "3. Thus, $n + m = 2a + 2b = 4(a + b)$, which is even.\n"
                "4. None of the above."
            )
        },
        {
            "problem_statement": "Evaluate the integral of $\\frac{1}{x} dx$.",
            "context": "We need to find the antiderivative of $\\frac{1}{x}$.",
            "options_text": (
                "1. Therefore, $\\int \\frac{1}{x} dx = -\\frac{1}{x^2} + C$\n"
                "2. Therefore, $\\int \\frac{1}{x} dx = \\frac{1}{x} + C$\n"
                "3. Therefore, $\\int \\frac{1}{x} dx = x + C$\n"
                "4. None of the above."
            )
        }
    ],
    few_shot_responses=[
        (
            "We are given that $n$ and $m$ are even numbers, and we have correctly "
            "deduced that $n = 2a$ and $m = 2b$ for some integers $a$ and $b$. "
            "Adding these two equations, we get $n + m = 2a + 2b = 2(a + b)$, "
            "as written in option 1.\n\n"
            "Answer: 1"
        ),
        (
            "We need to find the antiderivative of $\\frac{1}{x}$. "
            "It is well-known that the antiderivative of $\\frac{1}{x}$ is $\\ln|x| + C$, as written in option 2.\n\n"
            "Answer: 2"
        )
    ]
)


def _scramble_prompt(problem_statement: str, context: str, segment: str, num_scrambles: int) -> list[dict]:
    if context.strip() != "":
        content = (
            f"Given the problem statement:\n{problem_statement}\n\n"
            f"Given the context:\n{context}\n\n"
            f"And the next correct segment:\n{segment}\n\n"
            f"Generate {num_scrambles} variations of the next segment that are similar but contain a subtle mathematical error."
            " Provide the variations as a numbered list."
        )
    else:
        content = (
            f"Given the problem statement:\n{problem_statement}\n\n"
            f"And the next correct segment:\n{segment}\n\n"
            f"Generate {num_scrambles} variations of the next segment that are similar but contain a subtle mathematical error."
            " Provide the variations as a numbered list."
        )
    return [{"role": "user", "content": content}]


scramble_prompt = FewShotPrompter(
    system_prompt=(
        "You are a mathematical proof assistant. Your task is to "
        "generate variations of proof segments that contain subtle "
        "mathematical errors."
    ),
    function_prompt=_scramble_prompt,
    few_shot_examples=[
        {
            "problem_statement": "Prove that the sum of two even numbers is even.",
            "context": "Let n and m be two even numbers.",
            "segment": "Then $n = 2a$ and $m = 2b$ for some integers $a$ and $b$.",
            "num_scrambles": 3
        },
        {
            "problem_statement": "Prove that if n is odd, then n² is odd.",
            "context": "Let n be an odd number.",
            "segment": "Then $n = 2k + 1$ for some integer $k$.",
            "num_scrambles": 3
        }
    ],
    few_shot_responses=[
        (
            "1. Then $n = 2a + 1$ and $m = 2b$ for some integers $a$ and $b$.\n"
            "2. Then $n = a$ and $m = b$ for some odd integers $a$ and $b$.\n"
            "3. Then $n = 2a$ and $m = 2b + 1$ for some integers $a$ and $b$."
        ),
        (
            "1. Then $n = 2k$ for some integer $k$.\n"
            "2. Then $n = 3k - 1$ for some integer $k$.\n"
            "3. Then $n = k + 1$ for some odd integer $k$."
        )
    ]
)


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
        sample_n: int = 10,
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
        # Initialize the prompter with few-shot examples
        self.selection_prompter = selection_prompt
        self.scramble_prompter = scramble_prompt

    def generate_scrambled_versions(self, context: str, segment: str) -> list[str]:
        """
        Generates scrambled versions of a segment.
        """
        messages = self.scramble_prompter.few_shot_prompt(
            problem_statement=self.problem_statement,
            context=context,
            segment=segment,
            num_scrambles=self.num_scrambles
        )

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
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

            messages = self.selection_prompter.few_shot_prompt(
                problem_statement=self.problem_statement,
                context=context,
                options_text=options_text
            )

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                n=self.sample_n,
            )
            for choice in response.choices:
                text = choice.message.content.strip()
                # Find where the text says "Answer: <number>"
                match = re.search(r'Answer: (\d+)', text)
                if match:
                    response_idx = int(match.group(1)) - 1
                    # If "None of the above" was chosen
                    if response_idx == len(all_options) - 1:
                        none_of_above_count += 1
                    else:
                        frequencies[permutation[response_idx]] += 1
                else:
                    if self.verbose:
                        print(f"Error parsing response: {text}")
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
        if correct_votes / (self.num_multiple_choice_repeats * self.sample_n) >= self.accept_threshold:
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
