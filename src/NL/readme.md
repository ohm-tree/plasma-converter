# Natural Language Experiments

In this folder, I experiment with pure NL proofs. Given that Cursor can
write infinite code for me, I might as well try out some ideas.

### Philosophy

Surely, the gold standard for an RL LLM fine-tunes a pre-trained model
by treating it as a policy network in a game of next-token or corpus-rewriting
prediction and running UCB.

In order to achieve this, we would need to make sure our grader is robust.
For P vs NP reasons, we expect that proof verification is significantly
easier than proof generation. The naive approach fails, because LLMs are
easily fooled by convincing-sounding text.

To force the LLM to think, I take segments and generate $N$ small variations
on the segment. Then, I ask the LLM to select the most promising variation.
This should have at most an $O(1/N)$ false-positive rate, because in theory
small incorrect variations should be indistinguishable from the original
incorrect segment. False-negative rates have no such justified bound, but
it is much less of a problem, because they can indicate e.g. an insufficiently
careful reasoning step. These rates are sharpened through many iterations.

In the `proof_checker.py` file, I implement
the following checking mechanism:

1. Partition the proof into atomic chunks called "segments".
2. Label each segment as either a proposition, a step of logical deduction,
   or an informal remark. Propositions introduce new variables or lemmas,
   and do not need to be verified; neither do informal remarks.
3. Verify each step of logical deduction by checking if it follows
   logically from previous steps.

In `problem_scraper.py`, I implement a scraper for problems from Art of Problem
Solving.

