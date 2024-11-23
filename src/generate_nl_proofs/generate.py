# Iterate over the problem statements in /home/ubuntu/ohm-illinois/plasma-converter/datasets/minif2f.jsonl
# they have keys "name", "split", "informal_prefix", "formal_statement", "header".
# take the informal prefix, give it to gpt-4o-mini, and obtain a solution.
# save these solutions in /home/ubuntu/ohm-illinois/plasma-converter/datasets/minif2f_nl_proofs.jsonl
# the format is the same as minif2f.jsonl, but with an additional "natural_language_proof" field.

import json
import os
import time

import openai

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

LOAD_PATH = "/home/ubuntu/ohm-illinois/plasma-converter/datasets/minif2f.jsonl"
SAVE_PATH = "/home/ubuntu/ohm-illinois/plasma-converter/datasets/minif2f_nl_proofs.jsonl"

# # Testing
# LOAD_PATH = "/home/ubuntu/ohm-illinois/plasma-converter/datasets/small_dataset.jsonl"
# SAVE_PATH = "/home/ubuntu/ohm-illinois/plasma-converter/datasets/small_dataset_nl_proofs.jsonl"

# this is a JSONL file!!!
problem_statements = [json.loads(line) for line in open(LOAD_PATH)]

for problem_statement in problem_statements:
    # if the file already exists, skip
    if os.path.exists(f"/home/ubuntu/ohm-illinois/plasma-converter/datasets/nl_proofs/{problem_statement['name']}.json"):
        # load the file
        with open(f"/home/ubuntu/ohm-illinois/plasma-converter/datasets/nl_proofs/{problem_statement['name']}.json", "r") as f:
            problem_statement = json.load(f)
        continue

    successful_proof = False
    print(problem_statement["name"])
    print()
    print(problem_statement["informal_prefix"])
    print()
    print()
    informal_prefix = problem_statement["informal_prefix"]
    # trim
    informal_prefix = informal_prefix.strip()
    assert informal_prefix.startswith(
        "/--"), "The informal prefix must start with /--"
    assert informal_prefix.endswith(
        "-/"), "The informal prefix must end with -/"
    informal_prefix = informal_prefix[3:-2]

    start_time = time.time()
    response = client.chat.completions.create(
        model="o1-preview",
        messages=[{"role": "user", "content": informal_prefix}],
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(response.choices[0].message.content)

    problem_statement["natural_language_proof"] = response.choices[0].message.content
    # time taken is a proxy of difficulty
    problem_statement["time_taken"] = end_time - start_time

    # save this to an individual file
    os.makedirs(
        "/home/ubuntu/ohm-illinois/plasma-converter/datasets/nl_proofs", exist_ok=True)
    with open(f"/home/ubuntu/ohm-illinois/plasma-converter/datasets/nl_proofs/{problem_statement['name']}.json", "w") as f:
        json.dump(problem_statement, f)

json.dump(problem_statements, open(
    SAVE_PATH, "w"))
