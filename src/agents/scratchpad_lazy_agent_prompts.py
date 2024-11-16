from src.utils.prompt import FewShotPrompter

# For COT prompting, we like to provide examples of the output format.

EXAMPLE_NATURAL_LANGUAGE_PROBLEM_STATEMENT = (
    "The second and fourth terms of a geometric sequence are $2$ and $6$. Which of the following is a possible first term?\n"
    "$\\textbf{(A) } -\\sqrt{3}  \\qquad\\textbf{(B) } -\\frac{2\\sqrt{3}}{3} \\qquad\\textbf{(C) } -\\frac{\\sqrt{3}}{3} \\qquad\\textbf{(D) } \\sqrt{3} \\qquad\\textbf{(E) } 3$ Show that it is \\textbf{(B)}\\ -\\frac{2\\sqrt{3}}{3}."
)

EXAMPLE_NATURAL_LANGUAGE_PROOF = (
    "Call our geometric sequence $u$.\n"
    "We are given that $u_1 = 2$ and $u_3 = 6$.\n"
    "We know that $u_n = ar^{n-1}$ for all $n$, so $u_1 = ar = 2$ and $u_3 = ar^3 = 6$.\n"
    "Dividing the two equations, we get $r^2 = 3$.\n"
    "Hence, $r = \\sqrt{3}$ or $r = -\\sqrt{3}$.\n"
    "Substituting back into the first equation, we get $a = \\pm \\frac{2}{\\sqrt{3}}$.\n"
    "Thus, $u_0 = a = -\\frac{2\\sqrt{3}}{3}$, so the answer is $\\textbf{(B)}$.\n"
)

EXAMPLE_RAW_CODE_INPUT = (
    "theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2) (h₂ : u 3 = 6) : "
    "u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by"
)

EXAMPLE_ANNOTATED_CODE_INPUT = (
    "theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2) (h₂ : u 3 = 6) : "
    "u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by"
)

EXAMPLE_SCRATCHPAD_OUTPUT = (
    "To translate the natural language proof into a Lean 4 proof, I will break down each step and use Lean's mathematical constructs to mirror the reasoning presented in the natural language solution. Here's a detailed description of the process:\n"
    "\n"
    "1. **Define Variables and Hypotheses**: The problem statement in Lean already defines \\( a \\) (the first term), \\( r \\) (the common ratio), and \\( u \\) (the sequence) with a recursive definition based on the general term \\( u_k = a \\cdot r^k \\). We also have two hypotheses, \\( u_1 = 2 \\) and \\( u_3 = 6 \\), which we will use to solve for \\( r \\) and \\( a \\).\n"
    "\n"
    "2. **Setting Up the Ratios**: The goal is to show that \\( u_0 \\) (which corresponds to \\( a \\)) is equal to \\( \\pm \\frac{2}{\\sqrt{3}} \\). From the hypothesis that \\( u_1 = a \\cdot r \\) and \\( u_3 = a \\cdot r^3 \\), we will divide these two expressions to obtain an equation for \\( r^2 \\), which should yield \\( r^2 = 3 \\).\n"
    "\n"
    "3. **Solving for \\( r \\)**: After obtaining \\( r^2 = 3 \\), we can deduce \\( r = \\sqrt{3} \\) or \\( r = -\\sqrt{3} \\). Lean's real number library has methods to work with square roots, so we'll use `Real.sqrt` to express this.\n"
    "\n"
    "4. **Substituting Back to Find \\( a \\)**: With the value(s) of \\( r \\) substituted back into the equation for \\( u_1 \\), we can solve for \\( a \\). This will yield two possible values for \\( a \\), both of which involve dividing \\( 2 \\) by \\( \\sqrt{3} \\).\n"
    "\n"
    "5. **Completing the Proof**: Finally, we will show that \\( u_0 \\), the value of \\( a \\), is indeed one of the two possibilities given in the problem statement: \\( \\pm \\frac{2}{\\sqrt{3}} \\). Lean allows for disjunction (or statements), so we'll conclude the proof by stating that \\( u_0 = \\frac{2}{\\sqrt{3}} \\) or \\( u_0 = -\\frac{2}{\\sqrt{3}} \\)."
)

EXAMPLE_CODE_OUTPUT = (
    "```lean4\n"
    "theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2) (h₂ : u 3 = 6) : "
    "u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by\n"
    "  -- First, we want to re-write the condition about the second\n"
    "  -- and fourth terms of the geometric sequence using the definition of a geometric sequence\n"
    "  simp_all only [Nat.one_eq_succ_zero, Nat.zero_eq, zero_add, Nat.add_succ, Nat.add_zero, "
    "    Nat.succ_add]\n"
    "  have h₁' : a * r = 2 := by simpa [h₀] using h₁\n"
    "  have h₂' : a * r ^ 3 = 6 := by simpa [h₀] using h₂\n"
    "  -- Now we can divide the two equations to eliminate $a$ and determine $r$\n"
    "  have h₃ : r ^ 2 = 3 := by\n"
    "    nlinarith\n"
    "  -- Finally, we can substitute back to find $a$\n"
    "  have h₄ : a = 2 / Real.sqrt 3 ∨ a = -(2 / Real.sqrt 3) := by\n"
    "    apply eq_or_eq_neg_of_sq_eq_sq <;>\n"
    "    field_simp <;>\n"
    "    nlinarith\n"
    "  simpa [h₀] using h₄\n"
    "```\n"
)


SCRATCHPAD_SYSTEM_PROMPT = (
    "You are an expert mathematician and Lean prover assistant. "
    "You are attempting to translate a natural language proof into a Lean proof."
)

CONTEXT = (
    "The problem statement is:\n"
    "{problem_statement}\n"
    "The natural language proof is:\n"
    "{natural_language_proof}\n"
    "Here are my current notes in the scratchpad:\n"
    "<START SCRATCHPAD>\n"
    "{scratchpad}\n"
    "<END SCRATCHPAD>\n"
    "Here is the current state of a Lean proof:\n"
    "```lean4\n"
    "{annotated_code}\n"
    "```\n"
)

SCRATCHPAD_PROMPT = CONTEXT + (
    # "This code currently has the following error message:\n"
    # "{error_message}\n"
    "Please analyze the current state of the proof, "
    "especially noting the error messages. "
    "What should we try next? "
    "What insights can we draw from our previous attempts? "
    "Write your thoughts as a continuation of the scratchpad."
)


scratchpad_prompter = FewShotPrompter(
    system_prompt=SCRATCHPAD_SYSTEM_PROMPT,
    fstring_prompt=SCRATCHPAD_PROMPT
)


CODE_SYSTEM_PROMPT = (
    "You are an expert mathematician and Lean prover assistant. "
    "You are attempting to translate a natural language proof into a Lean proof."
    "You are given a Lean proof with error annotations and some notes about the proof."
    "Your task is to re-write the proof to fix the errors."
)

CODE_PROMPT = CONTEXT + (
    # "This code currently has the following error message:\n"
    # "{error_message}\n"
    "Please re-write the proof to fix the errors."
)


# CODE_ASSISTANT_PREFIX = (
#     "```lean4\n"
#     "{problem_statement}\n"
# )  # The assistant will output a continuation of this prefix.


code_prompter = FewShotPrompter(
    system_prompt=CODE_SYSTEM_PROMPT,
    fstring_prompt=CODE_PROMPT,
    few_shot_examples=[
        {
            "problem_statement": EXAMPLE_NATURAL_LANGUAGE_PROBLEM_STATEMENT,
            "natural_language_proof": EXAMPLE_NATURAL_LANGUAGE_PROOF,
            "scratchpad": EXAMPLE_SCRATCHPAD_OUTPUT,
            "annotated_code": EXAMPLE_ANNOTATED_CODE_INPUT,
        }
    ],
    few_shot_responses=[
        EXAMPLE_CODE_OUTPUT
    ]
)


VALUE_SYSTEM_PROMPT = "You are an expert mathematician and Lean prover assistant."

VALUE_PROMPT = CONTEXT + (
    "Based on the state of this code and the included annotations, "
    "rate the likelihood that this proof will succeed on a scale of 0 (very unlikely) to 100 (very likely). "
    "Please reason step by step about what's working and what isn't, then put your final answer in \\boxed{{}}."
)


def get_value_messages(annotated_code: str, scratchpad: str) -> list[dict]:
    return [
        {"role": "system", "content": VALUE_SYSTEM_PROMPT},
        {"role": "user", "content": VALUE_PROMPT.format(
            annotated_code=annotated_code, scratchpad=scratchpad)}
    ]


value_prompter = FewShotPrompter(
    examples=[
        {
            "input": EXAMPLE_ANNOTATED_CODE_INPUT,
            "output": EXAMPLE_VALUE_OUTPUT
        }
    ],
    system_prompt=VALUE_SYSTEM_PROMPT,
    user_prompt=VALUE_PROMPT
)
