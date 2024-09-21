# This test is to measure the speed of the model on a large dataset.
import json
import os
import time

import modal
import torch
import torch.multiprocessing as mp
from vllm import LLM, SamplingParams

app = modal.App("VLLM_speed_test")
image = modal.Image.from_registry("czhang2718/deepseek-lean-ubuntu-py")
image = image.pip_install("pydantic", extra_options="-U")


# currently, __file__ is /src/tests/LLM_speed_test.py
# we want to get to the local root folder.
local_root_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))


@app.function(image=image,
              gpu=modal.gpu.A100(count=1, size="80GB"),
              timeout=86400,
              mounts=[modal.Mount.from_local_dir(
                  os.path.join(local_root_dir, "datasets"), remote_path="/root/datasets")
              ]
              )
def run():

    data = []
    with open(os.path.join('datasets', 'minif2f.jsonl'), 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    # We should concatenate the informal prefix, header, formal statement, and goal together.

    input_data = []
    for d in data:
        input_data.append(d['goal'] + "\n" + d['informal_prefix'] + '```lean4\n' +
                          d['header'] + d['formal_statement'])

    # print the first data point.
    print(input_data[0])
    llm = LLM(model="deepseek-ai/DeepSeek-Prover-V1.5-RL",
              max_num_batched_tokens=8192,
              trust_remote_code=True)

    for test_size in [1, 10, 20, 50, 100, 200, 500]:
        print(f"Testing with {test_size} data points.")
        input_subset = []
        while len(input_subset) < test_size:
            input_subset.extend(input_data)

        input_subset = input_subset[:test_size]

        start = time.time()
        outputs = llm.generate(
            input_subset,
            sampling_params=SamplingParams(
                max_tokens=4096,
                temperature=1.0,
                top_p=1.0,
                n=1
            )
        )
        end = time.time()
        print(f"Time taken for {test_size} data points: {end - start}")

        # print out the first generated output.
        print(outputs[0].outputs[0].text)
