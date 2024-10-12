/-
Copyright (c) 2023 The Compfiles Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors:
-/

import Mathlib.Tactic

import ProblemExtraction

problem_file { tags := [.NumberTheory] }

/-!
# International Mathematical Olympiad 1997, Problem 5

Determine all pairs of integers 1 ≤ a,b that satisfy a ^ (b ^ 2) = b ^ a.
-/

namespace Imo1997P5

determine solution_set : Set (ℕ × ℕ) := sorry

problem imo1997_p5 (a b : ℕ) (ha : 1 ≤ a) (hb : 1 ≤ b) :
    ⟨a,b⟩ ∈ solution_set ↔ a ^ (b ^ 2) = b ^ a := by
  sorry


end Imo1997P5
