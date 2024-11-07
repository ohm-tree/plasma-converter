# segment_checker.py

import os
import random
from abc import ABC, abstractmethod

from openai import OpenAI

# Set up OpenAI API Key
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # Or set your API key directly here
)


def chat_gpt(prompt: str, model: str = "gpt-4") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


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
        num_repeats: int = 1
    ):
        super().__init__(problem_statement, segments, segment_index)
        self.num_scrambles = num_scrambles
        self.num_repeats = num_repeats

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
        content = chat_gpt(prompt)
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
        return variations

    def multiple_choice_evaluation(self, context: str, options: list[str]) -> str:
        """
        Presents multiple choices to the model and asks which one logically follows.
        """
        # Shuffle options and keep track of the index of the correct segment
        random.shuffle(options)
        correct_segment = self.segments[self.segment_index]
        try:
            correct_index = options.index(correct_segment)
        except ValueError:
            # In case the correct segment is not in options due to an error
            return None

        options_with_numbers = [
            f"{idx + 1}. {option}" for idx, option in enumerate(options)]
        options_text = '\n'.join(options_with_numbers)
        if context.strip() != "":
            prompt = (
                f"Given the problem statement:\n{self.problem_statement}\n\n"
                f"Given the context:\n{context}\n\n"
                f"Which of the following options logically follows from the context?\n{options_text}\n"
                "Please provide the number of the correct option."
            )
        else:
            prompt = (
                f"Given the problem statement:\n{self.problem_statement}\n\n"
                f"Which of the following options logically follows from the problem statement?\n{options_text}\n"
                "Please provide the number of the correct option."
            )
        choice = chat_gpt(prompt)
        # Return GPT's choice and the correct option number
        return choice.strip(), correct_index + 1

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
        self.explanation = chat_gpt(prompt)

    def check_segment(self) -> bool:
        """
        Checks the segment and returns True if it is correct, False otherwise.
        """
        context = ' '.join(self.segments[:self.segment_index])
        correct_segment = self.segments[self.segment_index]
        scrambled_segments = self.generate_scrambled_versions(
            context, correct_segment)
        options = scrambled_segments + [correct_segment]

        votes = {}
        for _ in range(self.num_repeats):
            choice, correct_option_number = self.multiple_choice_evaluation(
                context, options)
            if choice.isdigit():
                votes[choice] = votes.get(choice, 0) + 1
            else:
                # Handle unexpected responses
                continue

        # Determine if the correct segment was chosen majority of the time
        correct_votes = votes.get(str(correct_option_number), 0)
        if correct_votes / self.num_repeats >= 0.5:
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
        "Then n = 2a and m = 2b for some integers a and b.",
        "Thus, n + m = 2a + 2b = 2(a + b), which is even."
    ]
    for i in range(len(segments)):
        verifier = ScrambleVerifier(problem_statement, segments, i)
        result = verifier.check_segment()
        print(f"Segment {i+1}: {'Correct' if result else 'Incorrect'}")
        if not result:
            print(f"Explanation: {verifier.explanation}\n")

    # Incorrect Calculus Example
    problem_statement_calc = "Evaluate the integral of 1/x dx."
    segments_calc = [
        "We know that the integral of 1/x dx is -1/x^2 + C.",
        "Therefore, ∫1/x dx = -1/x^2 + C."
    ]
    print("Integral of 1/x (Calculus)")
    for i in range(len(segments_calc)):
        verifier = ScrambleVerifier(problem_statement_calc, segments_calc, i)
        result = verifier.check_segment()
        print(f"Segment {i+1}: {'Correct' if result else 'Incorrect'}")
        if not result:
            print(f"Explanation: {verifier.explanation}\n")
    print("\n")

    # Incorrect Number Theory Example
    problem_statement_nt = "Prove that the sum of any two odd numbers is even."
    segments_nt = [
        "Let m and n be odd numbers.",
        "Then m = 2a + 1 and n = 2b + 1 for some integers a and b.",
        "Therefore, m + n = 2a + 1 + 2b + 1 = 2(a + b + 1).",
        "Thus, m + n is odd."
    ]
    print("Sum of Two Odd Numbers (Number Theory)")
    for i in range(len(segments_nt)):
        verifier = ScrambleVerifier(problem_statement_nt, segments_nt, i)
        result = verifier.check_segment()
        print(f"Segment {i+1}: {'Correct' if result else 'Incorrect'}")
        if not result:
            print(f"Explanation: {verifier.explanation}\n")
            break  # Stop after the first incorrect segment
    print("\n")


if __name__ == "__main__":
    main()
