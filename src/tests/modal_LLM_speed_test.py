# This test is to measure the speed of the model on a large dataset.
import json
import os
import time

import modal
from transformers import AutoModelForCausalLM, AutoTokenizer

app = modal.App("LLM_speed_test")
image = modal.Image.from_registry("czhang2718/deepseek-lean-ubuntu-py")
image = image.pip_install("pydantic", extra_options="-U")


@app.cls(gpu="any", image=image)
class Model:
    @modal.build()  # add another step to the image build
    def download_model_to_folder(self):
        from huggingface_hub import snapshot_download

        os.makedirs("model_weights", exist_ok=True)
        snapshot_download("deepseek-ai/DeepSeek-Prover-V1.5-RL",
                          cache_dir="model_weights")
        return "Model downloaded successfully."


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

    # convert each data point to a string for inference.
    # Example:
    # {
    #     "name": "amc12a_2019_p21",
    #     "split": "valid",
    #     "informal_prefix": "/-- Let $z=\\frac{1+i}{\\sqrt{2}}.$What is $\\left(z^{1^2}+z^{2^2}+z^{3^2}+\\dots+z^{{12}^2}\\right) \\cdot \\left(\\frac{1}{z^{1^2}}+\\frac{1}{z^{2^2}}+\\frac{1}{z^{3^2}}+\\dots+\\frac{1}{z^{{12}^2}}\\right)?$\n\n$\\textbf{(A) } 18 \\qquad \\textbf{(B) } 72-36\\sqrt2 \\qquad \\textbf{(C) } 36 \\qquad \\textbf{(D) } 72 \\qquad \\textbf{(E) } 72+36\\sqrt2$ Show that it is \\textbf{(C) }36.-/\n",
    #     "formal_statement": "theorem amc12a_2019_p21 (z : ℂ) (h₀ : z = (1 + Complex.I) / Real.sqrt 2) :\n  ((∑ k : ℤ in Finset.Icc 1 12, z ^ k ^ 2) * (∑ k : ℤ in Finset.Icc 1 12, 1 / z ^ k ^ 2)) = 36 := by\n",
    #     "goal": "z : ℂ\nh₀ : z = (1 + Complex.I) / ↑√2\n⊢ (∑ k ∈ Finset.Icc 1 12, z ^ k ^ 2) * ∑ k ∈ Finset.Icc 1 12, 1 / z ^ k ^ 2 = 36",
    #     "header": "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"
    # }

    # We should concatenate the informal prefix, header, formal statement, and goal together.

    input_data = []
    for d in data:
        input_data.append(d['goal'] + "\n" + d['informal_prefix'] + '```lean4\n' +
                          d['header'] + d['formal_statement'])

    # print the first data point.
    print(input_data[0])

    tokenizer = AutoTokenizer.from_pretrained(
        'deepseek-ai/DeepSeek-Prover-V1.5-RL',
        trust_remote_code=True,
        device_map='auto',
        cache_dir='model_weights'
    )
    tokenizer.pad_token = tokenizer.eos_token
    base_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        'deepseek-ai/DeepSeek-Prover-V1.5-RL',
        trust_remote_code=True,
        device_map='auto',
        cache_dir='model_weights'
    )

    for test_size in [1, 10, 12, 14, 16, 18, 20]:
        print(f"Testing with {test_size} data points.")
        input_subset = []
        while len(input_subset) < test_size:
            input_subset.extend(input_data)

        input_subset = input_subset[:test_size]

        # Tokenize the input data, timing it.
        start = time.time()
        tokenized_input = tokenizer(input_subset, return_tensors='pt', padding=True,
                                    truncation=True, max_length=1024)
        end = time.time()
        print(f"Tokenization took {end-start} seconds.")

        # Move the tokenized_input to the GPU.
        tokenized_input = {k: v.to('cuda') for k, v in tokenized_input.items()}
        print("Tokenized input shape: ", tokenized_input['input_ids'].shape)

        print("Keys")
        print(tokenized_input.keys())

        # Inference on the input data, timing it.
        start = time.time()
        tokenized_output = base_model.generate(
            **tokenized_input, max_new_tokens=1024)
        end = time.time()
        print(f"Inference took {end-start} seconds.")

        print("output size: ", tokenized_output.shape)

        # Decode the tokenized_output.
        start = time.time()
        output = tokenizer.batch_decode(
            tokenized_output, skip_special_tokens=True)
        end = time.time()
        print(f"Decoding took {end-start} seconds.")
        # print(f"Example output: ", output[0])
