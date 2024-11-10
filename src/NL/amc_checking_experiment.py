import json
import os

import tqdm
from problem_scraper import scrape_amc
from proof_checker import ProofChecker

# Load the AoPS problems and verify their solutions!
for level in ["10A", "10B", "12A", "12B"]:
    for problem_number in range(1, 25):
        problem = scrape_amc(2024, level, problem_number)
        if problem is None:
            continue
        # check if solutions document already exists.
        filename = f"src/NL/data/amc/{2024}/{level}/{problem_number}/solutions.json"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                solutions = json.load(f)
        else:
            solutions = {}
            for solution in problem["latex_solutions"]:
                checker = ProofChecker(
                    problem["latex_problem_statement"],
                    solution,
                    segmentation_method='atomic',
                    verbose=True
                )
                res = checker.verify()
                if res["is_verified"]:
                    print("The proof is verified and correct.")
                else:
                    print("The proof is incorrect.")
                    print(f"Error found in segment: {res['error_segment']}")
                    print(f"Explanation: {res['explanation']}")
                solutions[solution] = res
            with open(filename, "w") as f:
                json.dump(solutions, f)
