# Iterate over the problem statements in /home/ubuntu/ohm-illinois/plasma-converter/datasets/minif2f.jsonl
# they have keys "name", "split", "informal_prefix", "formal_statement", "header".
# take the informal prefix, give it to gpt-4o-mini, and obtain a solution.
# save these solutions in /home/ubuntu/ohm-illinois/plasma-converter/datasets/minif2f_nl_proofs.jsonl
# the format is the same as minif2f.jsonl, but with an additional "natural_language_proof" field.

import json
import os

import openai

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# LOAD_PATH = "/home/ubuntu/ohm-illinois/plasma-converter/datasets/minif2f.jsonl"
# SAVE_PATH = "/home/ubuntu/ohm-illinois/plasma-converter/datasets/minif2f_nl_proofs.jsonl"

# Testing
LOAD_PATH = "/home/ubuntu/ohm-illinois/plasma-converter/datasets/small_dataset.jsonl"
SAVE_PATH = "/home/ubuntu/ohm-illinois/plasma-converter/datasets/small_dataset_nl_proofs.jsonl"

# this is a JSONL file!!!
problem_statements = [json.loads(line) for line in open(LOAD_PATH)]

for problem_statement in problem_statements:
    successful_proof = False
    print(problem_statement["name"])
    print()
    print(problem_statement["informal_prefix"])
    print()
    print()
    while not successful_proof:
        informal_prefix = problem_statement["informal_prefix"]
        # trim
        informal_prefix = informal_prefix.strip()
        assert informal_prefix.startswith(
            "/--"), "The informal prefix must start with /--"
        assert informal_prefix.endswith(
            "-/"), "The informal prefix must end with -/"
        informal_prefix = informal_prefix[3:-2]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": informal_prefix}],
        )
        print(response.choices[0].message.content)

        # print(response.choices[0].message.content)

        # problem_statement["natural_language_proof"] = response.choices[0].message.content
        response_text = response.choices[0].message.content

        # ask gpt-4o-mini to find any errors in the proof
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": informal_prefix},
                      {"role": "assistant", "content": response_text},
                      {"role": "user", "content": "Find any errors in the proof. If there are no errors, return 'No errors found'. If there are errors, return the errors and the location of the errors in the proof."}],
        )

        # print(response.choices[0].message.content)

        if response.choices[0].message.content == "No errors found.":
            successful_proof = True
        else:
            print(response.choices[0].message.content)
            print("Errors found in proof. Retrying...")
            continue

        problem_statement["natural_language_proof"] = response_text
        break

json.dump(problem_statements, open(
    SAVE_PATH, "w"))
