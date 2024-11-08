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
        segmentation_method: str = 'sentence',
        segmentation_kwargs: dict = {},
        checker_kwargs: dict = {},
        client: Optional[OpenAI] = None,
        verbose: bool = False
    ):
        self.problem_statement = problem_statement
        self.proof_text = proof_text
        self.segmentation_method = segmentation_method
        self.segmentation_kwargs = segmentation_kwargs
        self.checker_kwargs = checker_kwargs
        self.verbose = verbose
        if self.verbose:
            self.checker_kwargs['verbose'] = True

        self.is_verified = False
        self.error_segment = None
        self.explanation = None
        if client is None:
            self.client = OpenAI()
        else:
            self.client = client

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
                self.problem_statement, self.proof_text, **self.segmentation_kwargs
            )
        elif self.segmentation_method == 'atomic':
            segmentation = AtomicSegmentation(
                self.problem_statement, self.proof_text, client=self.client, **self.segmentation_kwargs
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

    def check_segments(self, verbose: bool = False):
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
                client=self.client,
                **self.checker_kwargs
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

    def verify(self) -> tuple[bool, str, str]:
        """
        Returns the verification result, error segment (if any), and explanation.

        Returns:
        -------
        tuple[bool, str, str]
            A tuple containing a boolean indicating whether the proof is verified,
            the error segment (if any), and the explanation.
        """

        self.segments = self.segment_proof()
        self.labels = self.label_segments()
        self.check_segments(verbose=self.verbose)

        return self.is_verified, self.error_segment, self.explanation


# Example usage
if __name__ == "__main__":
    problem_statement = "Prove that the $\\sqrt{2}$ is irrational."
    proof_text = (
        "Let's prove that the $\\sqrt{2}$ is irrational.\n"
        "Assume, for contradiction, that $\\sqrt{2}$ is rational.\n"
        "Then it can be expressed as a fraction $\\frac{a}{b}$ in lowest terms.\n"
        "So, $\\sqrt{2} = \\frac{a}{b}$ implies $2 = \\frac{a^2}{b^2}$.\n"
        "Therefore, $a^2 = 2b^2$.\n"
        "This means that $a^2$ is even, so $a$ must be even.\n"
        "Let $a = 2k$.\n"
        "Substituting back, we get $(2k)^2 = 2b^2$, so $4k^2 = 2b^2$.\n"
        "Simplifying, $2k^2 = b^2$.\n"
        "Therefore, $b^2$ is even, so $b$ is even.\n"
        "But this contradicts the assumption that $\\frac{a}{b}$ is in lowest terms.\n"
        "Therefore, $\\sqrt{2}$ is irrational.\n"
    )

    # Choose the segmentation method: 'sentence' or 'atomic'
    checker = ProofChecker(
        problem_statement,
        proof_text,
        segmentation_method='atomic',
        verbose=True
    )
    is_verified, error_segment, explanation = checker.verify()

    if is_verified:
        print("The proof is verified and correct.")
    else:
        print("The proof is incorrect.")
        print(f"Error found in segment: {error_segment}")
        print(f"Explanation: {explanation}")
