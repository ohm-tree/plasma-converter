# Plasma Converter

> "a mcts lean prover"

Plasma Converter is a personal project aimed at problems related to automated formal mathematical reasoning. Currently, it uses upper-confidence tree search to improve the inference performance of a LLM by exploring the space of possible proofs. Current directions include fine-tuning pre-training policy/value networks, autoformalization, and exploring tree search heuristics.

### Development

The git repository contains submodules. To clone it, you should use

```
git clone --recurse-submodules https://github.com/ohm-tree/plasma-converter.git
```

This git repository requires a companion project, [wayfinder](https://github.com/ohm-tree/wayfinder).

### Examples of solutions found

```lean
import Mathlib
import Aesop

set_option maxHeartbeats 0
set_option linter.all false
open BigOperators Real Nat Topology Rat

/-- The second and fourth terms of a geometric sequence are $2$ and $6$. Which of the following is a possible first term?

$\textbf{(A) } -\sqrt{3}  \qquad\textbf{(B) } -\frac{2\sqrt{3}}{3} \qquad\textbf{(C) } -\frac{\sqrt{3}}{3} \qquad\textbf{(D) } \sqrt{3} \qquad\textbf{(E) } 3$ Show that it is \textbf{(B)}\ -\frac{2\sqrt{3}}{3}.-/
theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2)
  (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
  simp_all only [h₀, Nat.cast_one, Nat.cast_zero, Nat.cast_succ, Nat.cast_sub, Nat.cast_mul,
    Nat.cast_pow]
  have h₃ : a * r = 2 := by linarith
  have h₄ : a * r ^ 3 = 6 := by linarith
  have h₅ : r ^ 2 = 3 := by
    apply Eq.symm
    nlinarith
  have h₆ : a = 2 / Real.sqrt 3 ∨ a = -(2 / Real.sqrt 3) := by
    apply eq_or_eq_neg_of_sq_eq_sq
    field_simp
    nlinarith
  simp_all only [pow_zero, mul_one]

-- Problem: amc12b_2003_p6
-- Split: valid
-- Result: 1.0
```

```lean
import Mathlib
import Aesop

set_option maxHeartbeats 0
set_option linter.all false
open BigOperators Real Nat Topology Rat

/-- How many integers are in the solution of the inequality $|x + 4|< 9$? Show that it is 17.-/
theorem mathd_algebra_185 (s : Finset ℤ) (f : ℤ → ℤ) (h₀ : ∀ x, f x = abs (x + 4))
  (h₁ : ∀ x, x ∈ s ↔ f x < 9) : s.card = 17 := by
  have h₂ : s = Finset.Ioo (-13) 5 := by
    ext x
    simp only [Finset.mem_Ioo, h₁, h₀, abs_lt, and_assoc, and_comm, sub_neg_eq_add, sub_lt_iff_lt_add]
    constructor
    · intro hx
      constructor
      · linarith
      · linarith
    · intro hx
      constructor
      · linarith
      · linarith
  simp only [h₂]
  rfl

-- Problem: mathd_algebra_185
-- Split: valid
-- Result: 1.0
```

```lean
import Mathlib
import Aesop

set_option maxHeartbeats 0
set_option linter.all false
open BigOperators Real Nat Topology Rat

/-- Solve for $x$: $\frac{x+1}{x-1} = \frac{x-2}{x+2}$ Show that it is 0.-/
theorem mathd_algebra_267 (x : ℝ) (h₀ : x ≠ 1) (h₁ : x ≠ -2)
  (h₂ : (x + 1) / (x - 1) = (x - 2) / (x + 2)) : x = 0 := by
  have h₃ : x - 1 ≠ 0 := by
    intro h
    apply h₀
    linarith
  have h₄ : x + 2 ≠ 0 := by
    intro h
    apply h₁
    linarith
  field_simp [h₃, h₄] at h₂
  ring_nf at h₂
  linarith

-- Problem: mathd_algebra_267
-- Split: valid
-- Result: 1.0
```

```lean
import Mathlib
import Aesop

set_option maxHeartbeats 0
set_option linter.all false
open BigOperators Real Nat Topology Rat

/-- If $f (x) = x + 2$ and $g (x) = x^2$, then for what value of $x$ does $f(g(x)) = g(f(x))$? Express your answer as a common fraction. Show that it is $-\frac{1}{2}$.-/
theorem mathd_algebra_132 (x : ℝ) (f g : ℝ → ℝ) (h₀ : ∀ x, f x = x + 2) (h₁ : ∀ x, g x = x ^ 2)
  (h₂ : f (g x) = g (f x)) : x = -1 / 2 := by
  rw [h₀, h₁] at h₂
  rw [h₁, h₀] at h₂
  have h₃ : x ^ 2 + 2 = (x + 2) ^ 2 := by
    linarith
  have h₄ : x ^ 2 + 2 = x ^ 2 + 4 * x + 4 := by
    linarith
  have h₅ : 4 * x = -2 := by
    linarith
  have h₆ : x = -1 / 2 := by
    linarith
  exact h₆

-- Problem: mathd_algebra_132
-- Split: valid
-- Result: 1.0
```