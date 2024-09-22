import modal

app = modal.App("lean-game-test")

image = modal.Image.from_registry("czhang2718/deepseek-lean-ubuntu-py")
image = image.pip_install("pydantic", extra_options="-U")

@app.function(image=image,
              gpu=modal.gpu.A100(count=1, size="80GB"),
              timeout=86400,
              mounts=[modal.Mount.from_local_dir("../", remote_path="/root/src"),
                       modal.Mount.from_local_dir("../../prover", remote_path="/root/prover")])
def test_prover_llm():
    from src.networks.prover_llm import ProverLLM
    model = ProverLLM()

    sample_input3 = r'''

    ```lean4
    /-- The second and fourth terms of a geometric sequence are $2$ and $6$. Which of the following is a possible first term?
    Show that it is $\frac{2\sqrt{3}}{3}$.-/
    theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2)
    (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
    --
    ```
    Output a list of 100 possible next comments that would guide the proof:

    '''

    sample_tokens = model.tokenizer(sample_input3, return_tensors='pt')[
        'input_ids'].to("cuda")
    sample_attention_mask = model.tokenizer(sample_input3, return_tensors='pt')[
        'attention_mask'].to("cuda")

    completed_proof = model.complete(
        sample_tokens, sample_attention_mask, max_length=200)
    # policy, value = model.mcts_forward(sample_tokens, sample_attention_mask)

    print("completed_proof")
    print(completed_proof)

    # with open('../sample_data/sample_prover_llm_output3.txt', 'w') as f:
    #     f.write("Completed Proof:\n")
    #     f.write(completed_proof)
        # f.write("\n\nValue:\n")
        # f.write(str(value))
        # f.write("\n\nPolicy (logits):\n")
        # f.write(str(policy))


@app.local_entrypoint()
def main():
    test_prover_llm.remote()
