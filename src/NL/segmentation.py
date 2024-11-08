from typing import Optional

from openai import OpenAI


class ProofSegmentation:
    """
    Base class for proof segmentation methods.
    """

    def __init__(self, problem_text: str, proof_text: str):
        self.problem_text = problem_text
        self.proof_text = proof_text

    def segment(self) -> list[str]:
        """
        Segments the proof text into a list of segments.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class SentenceSegmentation(ProofSegmentation):
    """
    Segments the proof into sentences by checking for punctuation marks.
    """

    def segment(self) -> list[str]:
        import re

        # Use regex to split the proof text into sentences based on punctuation marks
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_endings.split(self.proof_text.strip())
        return [sentence.strip() for sentence in sentences if sentence.strip()]


class AtomicSegmentation(ProofSegmentation):
    """
    Segments the proof into atomic ideas using GPT with improved few-shot prompting.
    """

    def __init__(self, problem_text: str, proof_text: str, client: Optional[OpenAI] = None):
        super().__init__(problem_text, proof_text)
        if client is None:
            # Initialize the OpenAI client (API key is read from the environment variable OPENAI_API_KEY)
            self.client = OpenAI()
        else:
            self.client = client

    def segment(self) -> list[str]:
        # Simulate previous user-assistant interactions
        few_shot_conversation = [
            # Example 1
            {"role": "user", "content": (
                "Segment the proof in the following problem into atomic logical steps. "
                "Each step should represent a single logical idea or inference. "
                "Do not segment the problem statement.\n\n"
                "Problem:\n"
                "Prove that the sum of two even numbers is even.\n"
                "Proof:\n"
                "Let a and b be even numbers. Then a = 2k and b = 2m for some integers k and m. "
                "Therefore, a + b = 2(k + m), which is even."
            )},
            {"role": "assistant", "content": (
                "1. Let a and b be even numbers.\n"
                "2. Then a = 2k for some integer k.\n"
                "3. And b = 2m for some integer m.\n"
                "4. Therefore, a + b = 2k + 2m.\n"
                "5. Simplify to a + b = 2(k + m).\n"
                "6. Since k + m is an integer, a + b is even."
            )},
            # Example 2
            {"role": "user", "content": (
                "Segment the proof in the following problem into atomic logical steps. "
                "Each step should represent a single logical idea or inference. "
                "Do not segment the problem statement.\n\n"
                "Problem:\n"
                "Prove that if a function is differentiable, then it is continuous.\n"
                "Proof:\n"
                "Let f be a differentiable function at point c. "
                "Then the derivative f'(c) exists. "
                "Therefore, the limit of (f(c + h) - f(c))/h as h approaches 0 exists. "
                "Hence, f is continuous at c."
            )},
            {"role": "assistant", "content": (
                "1. Let f be a differentiable function at point c.\n"
                "2. Then the derivative f'(c) exists.\n"
                "3. Therefore, the limit of [f(c + h) - f(c)]/h as h approaches 0 exists.\n"
                "4. Hence, f is continuous at c."
            )},
            # You can add more examples here
        ]

        # Add the current problem and proof to be segmented
        current_proof = {
            "role": "user",
            "content": (
                "Segment the proof in the following problem into atomic logical steps. "
                "Each step should represent a single logical idea or inference. "
                "Do not segment the problem statement.\n\n"
                f"Problem:\n{self.problem_text}\n"
                f"Proof:\n{self.proof_text}\n"
            )
        }

        # Combine the few-shot examples with the current proof
        messages = few_shot_conversation + [current_proof]

        # Call the OpenAI API with the constructed messages
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )

        # Extract the assistant's reply
        content = response.choices[0].message.content.strip()

        # Parse the segments from the assistant's response
        segments = self.parse_gpt_segments(content)
        return segments

    def parse_gpt_segments(self, content: str) -> list[str]:
        """
        Parses the GPT response to extract the list of segments.
        """
        import re
        segments = []
        lines = content.strip().split('\n')
        for line in lines:
            # Match lines that start with a number followed by a period
            match = re.match(r'^\d+\.\s*(.*)', line)
            if match:
                segment = match.group(1).strip()
                if segment:
                    segments.append(segment)
        return segments


def main():
    # Example problem and proof
    problem_text = (
        "Prove that the square of any odd integer is odd."
    )
    proof_text = (
        "Let n be an odd integer. Then n = 2k + 1 for some integer k. "
        "Therefore, n^2 = (2k + 1)^2 = 4k^2 + 4k + 1, which is odd."
    )

    segmenter = SentenceSegmentation(problem_text, proof_text)
    segments = segmenter.segment()

    print("Problem:")
    print(problem_text)
    print("\nSegmented Proof:")
    for idx, segment in enumerate(segments, 1):
        print(f"{idx}. {segment}")

    segmenter = AtomicSegmentation(problem_text, proof_text)
    segments = segmenter.segment()

    print("Problem:")
    print(problem_text)
    print("\nSegmented Proof:")
    for idx, segment in enumerate(segments, 1):
        print(f"{idx}. {segment}")

    # Expected output (results may vary due to the randomness in GPT-4):
    # Problem:
    # Prove that the square of any odd integer is odd.
    #
    # Segmented Proof:
    # 1. Let n be an odd integer.
    # 2. Then n = 2k + 1 for some integer k.
    # 3. Therefore, n^2 = (2k + 1)^2 = 4k^2 + 4k + 1, which is odd.
    # Problem:
    # Prove that the square of any odd integer is odd.
    #
    # Segmented Proof:
    # 1. Let n be an odd integer.
    # 2. Then n = 2k + 1 for some integer k.
    # 3. Therefore, n^2 = (2k + 1)^2.
    # 4. Simplify to n^2 = 4k^2 + 4k + 1.
    # 5. Since the form 4k^2 + 4k + 1 is in the format of 2m+1, n^2 is odd.


if __name__ == '__main__':
    main()
