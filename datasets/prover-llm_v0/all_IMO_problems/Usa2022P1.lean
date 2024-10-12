import Mathlib.Data.Finset.Card
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Fintype.Prod
import Mathlib.Tactic.Positivity

import ProblemExtraction

problem_file { tags := [.Combinatorics] }

/-!
# USA Mathematical Olympiad 2022, Problem 1

Let a and b be positive integers. The cells of an (a+b+1) × (a+b+1) grid
are colored amber and bronze such that there are at least a² + ab - b
amber cells and at least b² + ab - a bronze cells. Prove that it is
possible to choose a amber cells and b bronze cells such that no two
of the a + b chosen cells lie in the same row or column.
-/

namespace Usa2022P1

problem usa2022_p1
    (a b : ℕ)
    (ha : 0 < a)
    (hb : 0 < b)
    (color : Fin (a + b + 1) × Fin (a + b + 1) → Fin 2)
    (c0 : a^2 + a * b - b ≤ Fintype.card {s // color s = 0})
    (c1 : b^2 + a * b - a ≤ Fintype.card {s // color s = 1}) :
    ∃ A B : Finset (Fin (a + b + 1) × Fin (a + b + 1)),
      A.card = a ∧ B.card = b ∧
      (∀ x ∈ A, color x = 0) ∧
      (∀ y ∈ B, color y = 1) ∧
      ∀ x ∈ A ∪ B, ∀ y ∈ A ∪ B, x ≠ y → x.fst ≠ y.fst ∧ x.snd ≠ y.snd := by
  sorry


end Usa2022P1
