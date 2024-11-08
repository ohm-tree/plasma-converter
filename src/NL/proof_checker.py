# proof_checker.py

import os
from typing import Optional

from openai import OpenAI

from src.NL.segment_checker import ScrambleVerifier
from src.NL.segment_labeler import BasicSegmentLabeler
from src.NL.segmentation import AtomicSegmentation, SentenceSegmentation


class ProofChecker:
    def __init__(
        self,
        problem_statement: str,
        proof_text: str,
        num_scrambles: int = 3,
        segmentation_method: str = 'sentence',
        verification_method: str = 'scramble',
        num_repeats: int = 1,
        client: Optional[OpenAI] = None,
        verbose: bool = False
    ):
        self.problem_statement = problem_statement
        self.proof_text = proof_text
        self.num_scrambles = num_scrambles
        self.segmentation_method = segmentation_method
        self.num_repeats = num_repeats
        self.segments = self.segment_proof()
        self.labels = self.label_segments()
        self.is_verified = False
        self.error_segment = None
        self.explanation = None
        if client is None:
            self.client = OpenAI()
        else:
            self.client = client
        self.verify_proof(verbose=verbose)

    def segment_proof(self) -> list[str]:
        """
        Segments the proof into smaller chunks using the specified segmentation method.

        Returns:
        -------
        list[str]
            A list of segments.
        """
        if self.segmentation_method == 'sentence':
            segmentation = SentenceSegmentation(
                self.problem_statement, self.proof_text)
        elif self.segmentation_method == 'atomic':
            segmentation = AtomicSegmentation(
                self.problem_statement, self.proof_text, self.client
            )
        else:
            raise ValueError(
                f"Unknown segmentation method: {self.segmentation_method}")
        segments = segmentation.segment()
        return segments

    def label_segments(self) -> list[str]:
        """
        Labels each segment as either 'deduction' or 'verification'.

        Returns:
        -------
        list[str]
            A list of labels for each segment.  
        """
        labeled_segments = []
        for i in range(len(self.segments)):
            labeler = BasicSegmentLabeler(
                self.problem_statement, self.segments, i, self.client
            )
            labeled_segments.append(labeler.label_segment())
        return labeled_segments

    def verify_proof(self, verbose: bool = False):
        """
        Verifies the proof by iteratively checking each deduction using a SegmentChecker.
        """
        for i in range(len(self.segments)):
            if self.labels[i] == 'proposition':
                continue
            verifier = ScrambleVerifier(
                self.problem_statement,
                self.segments,
                i,
                num_scrambles=self.num_scrambles,
                num_repeats=self.num_repeats,
                client=self.client
            )
            if not verifier.check_segment():
                # Proof is rejected
                self.is_verified = False
                self.error_segment = self.segments[i]
                self.explanation = verifier.explanation
                if verbose:
                    print(
                        f"Proof is incorrect. Error found in segment: {self.error_segment}")
                    print(f"Explanation: {self.explanation}")
                return
            else:
                if verbose:
                    print(f"Segment {i} verified.")

        # All segments verified
        self.is_verified = True

    def get_result(self) -> tuple[bool, str, str]:
        """
        Returns the verification result, error segment (if any), and explanation.

        Returns:
        -------
        tuple[bool, str, str]
            A tuple containing a boolean indicating whether the proof is verified,
            the error segment (if any), and the explanation.
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
    is_verified, error_segment, explanation = checker.get_result()

    if is_verified:
        print("The proof is verified and correct.")
    else:
        print("The proof is incorrect.")
        print(f"Error found in segment: {error_segment}")
        print(f"Explanation: {explanation}")
