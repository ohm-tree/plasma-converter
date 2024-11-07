# proof_checker.py

import os

from openai import OpenAI

from src.NL.segmentation import AtomicSegmentation, SentenceSegmentation

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


class ProofChecker:
    def __init__(
        self,
        problem_statement: str,
        proof_text: str,
        num_scrambles: int = 3,
        segmentation_method: str = 'sentence',
        num_repeats: int = 1
    ):
        self.problem_statement = problem_statement
        self.proof_text = proof_text
        self.num_scrambles = num_scrambles
        self.segmentation_method = segmentation_method
        self.num_repeats = num_repeats
        self.segments = self.segment_proof()
        self.is_verified = False
        self.error_segment = None
        self.explanation = None

    def segment_proof(self) -> list[str]:
        """
        Segments the proof into smaller chunks using the specified segmentation method.
        """
        if self.segmentation_method == 'sentence':
            segmentation = SentenceSegmentation(
                self.problem_statement, self.proof_text)
        elif self.segmentation_method == 'atomic':
            segmentation = AtomicSegmentation(
                self.problem_statement, self.proof_text)
        else:
            raise ValueError(
                f"Unknown segmentation method: {self.segmentation_method}")
        segments = segmentation.segment()
        return segments

    def verify_proof(self):
        """
        Verifies the proof by iteratively checking each segment using a SegmentChecker.
        """
        for i in range(len(self.segments)):
            verifier = ScrambleVerifier(
                self.problem_statement,
                self.segments,
                i,
                num_scrambles=self.num_scrambles,
                num_repeats=self.num_repeats
            )
            if not verifier.check_segment():
                # Proof is rejected
                self.is_verified = False
                self.error_segment = self.segments[i]
                self.explanation = verifier.explanation
                return
        # All segments verified
        self.is_verified = True

    def get_result(self) -> tuple[bool, str, str]:
        """
        Returns the verification result, error segment (if any), and explanation.
        """
        return self.is_verified, self.error_segment, self.explanation


# Example usage
if __name__ == "__main__":
    problem_statement = "Prove that the square root of 2 is irrational."
    proof_text = """
    Let's prove that the square root of 2 is irrational.
    Assume, for contradiction, that sqrt(2) is rational.
    Then it can be expressed as a fraction a/b in lowest terms.
    So, sqrt(2) = a/b implies 2 = a^2 / b^2.
    Therefore, a^2 = 2b^2.
    This means that a^2 is even, so a must be even.
    Let a = 2k.
    Substituting back, we get (2k)^2 = 2b^2, so 4k^2 = 2b^2.
    Simplifying, 2k^2 = b^2.
    Therefore, b^2 is even, so b is even.
    But this contradicts the assumption that a/b is in lowest terms.
    Therefore, sqrt(2) is irrational.
    """

    # Choose the segmentation method: 'sentence' or 'atomic'
    checker = ProofChecker(
        problem_statement,
        proof_text,
        segmentation_method='atomic',
        num_scrambles=3,
        num_repeats=3
    )
    checker.verify_proof()
    is_verified, error_segment, explanation = checker.get_result()

    if is_verified:
        print("The proof is verified and correct.")
    else:
        print("The proof is incorrect.")
        print(f"Error found in segment: {error_segment}")
        print(f"Explanation: {explanation}")
