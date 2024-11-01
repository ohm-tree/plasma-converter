# For COT prompting, we like to provide examples of the output format.

EXAMPLE_RAW_CODE_INPUT = """theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2) (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
"""

EXAMPLE_ANNOTATED_CODE_INPUT = """theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2) (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
"""

EXAMPLE_CODE_OUTPUT = """```lean4
theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2) (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
  -- First, we want to re-write the condition about the second
  -- and fourth terms of the geometric sequence using the definition of a geometric sequence
  simp_all only [Nat.one_eq_succ_zero, Nat.zero_eq, zero_add, Nat.add_succ, Nat.add_zero,
    Nat.succ_add]
  have h₁' : a * r = 2 := by simpa [h₀] using h₁
  have h₂' : a * r ^ 3 = 6 := by simpa [h₀] using h₂
  -- Now we can divide the two equations to eliminate $a$ and determine $r$
  have h₃ : r ^ 2 = 3 := by
    nlinarith
  -- Finally, we can substitute back to find $a$
  have h₄ : a = 2 / Real.sqrt 3 ∨ a = -(2 / Real.sqrt 3) := by
    apply eq_or_eq_neg_of_sq_eq_sq <;>
    field_simp <;>
    nlinarith
  simpa [h₀] using h₄
```
"""


SCRATCHPAD_SYSTEM_PROMPT = "You are an expert mathematician and Lean prover assistant."

SCRATCHPAD_PROMPT = """Here is the current state of a Lean proof:
```lean4
{annotated_code}
```
Here are my current notes in the scratchpad:
{scratchpad}

Please analyze the current state of the proof, especially noting the error messages.
What should we try next? What insights can we draw from our previous attempts?
Write your thoughts as a continuation of the scratchpad."""


def get_scratchpad_messages(annotated_code: str, scratchpad: str) -> list[dict]:
    return [
        {"role": "system", "content": SCRATCHPAD_SYSTEM_PROMPT},
        {"role": "user", "content": SCRATCHPAD_PROMPT.format(
            annotated_code=annotated_code, scratchpad=scratchpad)}
    ]


CODE_SYSTEM_PROMPT = "You are an expert mathematician and Lean prover assistant."

CODE_PROMPT = """Here is a Lean proof with error annotations:
```lean4
{annotated_code}
```

Here are my notes about the proof:
{scratchpad}

Please re-write the proof to fix the errors.
"""

CODE_ASSISTANT_PREFIX = """```lean4
{problem_statement}
"""  # The assistant will output a continuation of this prefix.


def get_code_messages(annotated_code: str, problem_statement: str, scratchpad: str) -> list[dict]:
    return [
        {"role": "system", "content": CODE_SYSTEM_PROMPT},
        {"role": "user", "content": CODE_PROMPT.format(
            annotated_code=annotated_code, scratchpad=scratchpad)},
        {"role": "assistant", "content": CODE_ASSISTANT_PREFIX.format(
            problem_statement=problem_statement)}
    ]


VALUE_SYSTEM_PROMPT = "You are an expert mathematician and Lean prover assistant."

VALUE_PROMPT = """Here is a Lean proof with error annotations:
```lean4
{annotated_code}
```

Here are the notes from attempting this proof:
{scratchpad}

Based on the state of this code and the included annotations, rate the likelihood that this proof will succeed on a scale of 0 (very unlikely) to 100 (very likely).
Please reason step by step about what's working and what isn't, then put your final answer in \\boxed{{}}."""


def get_value_messages(annotated_code: str, scratchpad: str) -> list[dict]:
    return [
        {"role": "system", "content": VALUE_SYSTEM_PROMPT},
        {"role": "user", "content": VALUE_PROMPT.format(
            annotated_code=annotated_code, scratchpad=scratchpad)}
    ]
