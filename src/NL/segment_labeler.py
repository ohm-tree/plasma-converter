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
        'proposition' if it contains only propositions, 'definition' if it
        contains definitions, or 'informal' if it contains informal reasoning.
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

    def _create_segment_prompt(self, problem_statement: str, context: str, segment: str) -> str:
        """Creates a prompt for segment labeling, optionally including context."""
        prompt = f"Given the problem statement:\n{problem_statement}\n\n"

        if context.strip():
            prompt += f"Given the context:\n{context}\n\n"

        prompt += f"Label this segment:\n{segment}"
        return prompt

    def _create_few_shot_examples(self) -> list[dict]:
        """Creates few-shot examples using the prompt function."""
        example_problem = "Prove that $\\sqrt{2}$ is irrational."
        examples = [
            {
                "prompt": self._create_segment_prompt(
                    problem_statement=example_problem,
                    context="",
                    segment="Let's prove that $\\sqrt{2}$ is irrational."
                ),
                "response": "proposition"
            },
            {
                "prompt": self._create_segment_prompt(
                    problem_statement=example_problem,
                    context="Let's prove that $\\sqrt{2}$ is irrational.",
                    segment="Then it can be expressed as a fraction $\\frac{a}{b}$ in lowest terms."
                ),
                "response": "deduction"
            },
            {
                "prompt": self._create_segment_prompt(
                    problem_statement=example_problem,
                    context="Let's prove that $\\sqrt{2}$ is irrational. Assume, for contradiction, that $\\sqrt{2}$ is rational.",
                    segment="So, $\\sqrt{2} = \\frac{a}{b}$ implies $2 = \\frac{a^2}{b^2}$."
                ),
                "response": "deduction"
            },
            {
                "prompt": self._create_segment_prompt(
                    problem_statement=example_problem,
                    context="Let's prove that $\\sqrt{2}$ is irrational. Assume, for contradiction, that $\\\sqrt{2}$ is rational. Then $\\\sqrt{2} = \\frac{a}{b}$ implies $2 = \\frac{a^2}{b^2}$.",
                    segment="Therefore, $a^2 = 2b^2$."
                ),
                "response": "deduction"
            },
            {
                "prompt": self._create_segment_prompt(
                    problem_statement=example_problem,
                    context="Let's prove that $\\\sqrt{2}$ is irrational. Assume, for contradiction, that $\\\sqrt{2}$ is rational.",
                    segment="This is a key insight that will help us reach a contradiction later."
                ),
                "response": "informal"
            }
        ]
        return examples

    def _generate_messages(self, segment: str) -> list[dict]:
        """Generates the full message list for the API call."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a mathematical proof analyzer that labels segments "
                    "of proofs as either 'deduction', 'proposition', 'definition', or 'informal'."
                )
            }
        ]

        # Add few-shot examples
        examples = self._create_few_shot_examples()
        for example in examples:
            messages.extend([
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["response"]}
            ])

        # Add the actual segment to be labeled
        final_prompt = self._create_segment_prompt(
            problem_statement=self.problem_statement,
            context=self.context,
            segment=segment
        )
        messages.append({"role": "user", "content": final_prompt})

        return messages

    def label_segment(self) -> str:
        segment = self.segments[self.segment_index]

        # Pre-processing step: check for deduction keywords
        segment_lower = segment.lower()
        segment_lower_tokens = segment_lower.split()
        if any(keyword in segment_lower_tokens for keyword in self.deduction_keywords):
            self.label = 'deduction'
            return self.label

        # Generate messages and make API call
        messages = self._generate_messages(segment)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        response_clean = response.choices[0].message.content.strip().lower()
        if 'deduction' in response_clean:
            self.label = 'deduction'
        elif 'definition' in response_clean:
            self.label = 'definition'
        elif 'informal' in response_clean:
            self.label = 'informal'
        elif 'proposition' in response_clean:
            self.label = 'proposition'
        else:
            print("Warning: Unknown label:", response_clean)
            self.label = 'unknown'
        return self.label


def main():
    # Example problem statement and segments
    problem_statement = "Prove that the square of an even number is even."
    segments = [
        "Let $n$ be an even number.",
        "Then $n = 2k$ for some integer $k$.",
        "Therefore, $n^2 = (2k)^2 = 4k^2$.",
        "Thus, $n^2$ is divisible by 4 and hence even.",
        "Because $n$ is even, $n^2$ is also even.",
        "Consider the function $f(x) = x^2$.",
        "We have $f(n) = n^2$.",
        "Since $n$ is even, $f(n)$ is even."
    ]

    print("Labeling segments:")
    for i in range(len(segments)):
        labeler = BasicSegmentLabeler(problem_statement, segments, i)
        label = labeler.label_segment()
        print(f"Segment {i+1}: '{segments[i]}'")
        print(f"Label: {label.capitalize()}\n")


if __name__ == "__main__":
    main()
