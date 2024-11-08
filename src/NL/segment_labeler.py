from abc import ABC, abstractmethod
from typing import Optional

from openai import OpenAI


class SegmentLabeler(ABC):
    """
    Abstract base class for labeling a proof segment as containing deductions or only propositions.
    """

    def __init__(self, problem_statement: str, segments: list[str], segment_index: int):
        self.problem_statement = problem_statement
        self.segments = segments
        self.segment_index = segment_index
        # Context is the preceding segments
        self.context = ' '.join(segments[:segment_index])
        self.label = None  # 'deduction' or 'proposition'

    @abstractmethod
    def label_segment(self) -> str:
        """
        Labels the segment and returns 'deduction' if it contains deductions,
        or 'proposition' if it contains only propositions.
        """
        pass


class BasicSegmentLabeler(SegmentLabeler):
    """
    Labels a segment based on keywords and GPT evaluation.
    """

    def __init__(self, problem_statement: str, segments: list[str], segment_index: int, client: Optional[OpenAI] = None):
        super().__init__(problem_statement, segments, segment_index)
        if client is None:
            self.client = OpenAI()
        else:
            self.client = client
        self.deduction_keywords = [
            'then', 'therefore', 'thus', 'hence', 'implies', 'consequently',
            'as a result', 'so', 'accordingly', 'because', 'since', 'due to'
        ]

    def label_segment(self) -> str:
        segment = self.segments[self.segment_index]
        # Pre-processing step: check for deduction keywords
        segment_lower = segment.lower()
        if any(keyword in segment_lower for keyword in self.deduction_keywords):
            self.label = 'deduction'
            return self.label
        else:
            # Replace chat_gpt call with direct OpenAI API call
            prompt = (
                f"Given the problem statement:\n{self.problem_statement}\n\n"
                f"Given the context:\n{self.context}\n\n"
                f"Given the following segment:\n{segment}\n\n"
                "Does this segment contain any logical deductions or "
                "is it only stating propositions or definitions?\n"
                "Please answer 'deduction' or 'proposition'."
            )
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            response_clean = response.choices[0].message.content.strip(
            ).lower()
            if 'deduction' in response_clean:
                self.label = 'deduction'
            else:
                self.label = 'proposition'
            return self.label


def main():
    # Example problem statement and segments
    problem_statement = "Prove that the square of an even number is even."
    segments = [
        "Let n be an even number.",
        "Then n = 2k for some integer k.",
        "Therefore, n^2 = (2k)^2 = 4k^2.",
        "Thus, n^2 is divisible by 4 and hence even.",
        "Because n is even, n^2 is also even.",
        "Consider the function f(x) = x^2.",
        "We have f(n) = n^2.",
        "Since n is even, f(n) is even."
    ]

    print("Labeling segments:")
    for i in range(len(segments)):
        labeler = BasicSegmentLabeler(problem_statement, segments, i)
        label = labeler.label_segment()
        print(f"Segment {i+1}: '{segments[i]}'")
        print(f"Label: {label.capitalize()}\n")


if __name__ == "__main__":
    main()
