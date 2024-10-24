'''

Send queries to ChatGPT 4o asking it to translate formal lean code into a natural language description.

'''

import os
from openai import OpenAI
import json
from tqdm import tqdm

# Get the API key from the shell variable $OPENAIKEY
client = OpenAI(api_key=os.getenv("OPENAIKEY"))


def query_gpt4(prompt, model="gpt-4"):
    """
    Sends a prompt to the GPT-4 model and returns the response.
    Args:
        prompt (str): The input text or query.
        model (str): The OpenAI model to use (default is "gpt-4").
    Returns:
        str: The model's response.
    """
    try:
        response = client.chat.completions.create(model=model,
        messages=[
            {"role": "system", "content": "You are ChatGPT4o, an assistant. \
             Your task is to translate formal Lean code into a natural language description.\
             Rather than explaining the code, simply write the solution in natural language.\
             If both a problem statement and a proof are provided, you should output in the following format:\
             BEGINPROBLEMSTATEMENT: <problem statement> ENDPROBLEMSTATEMENT \n\n BEGINPROOF: <proof> ENDPROOF."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,  # Adjust this based on your requirements
        temperature=0.7)

        # Extract and return the assistant's reply
        return response.choices[0].message.content

    except Exception as e:
        return f"An error occurred: {str(e)}"



# load prompts from ./datasets/minif2f_valid_few_shot.jsonl
# as an example, the start of the jsonl file is:
'''
{"name": "mathd_algebra_182", "split": "valid", "informal_prefix": "/-- Expand the following expression: $7(3y+2)$ Show that it is 21y+14.-/\n", "formal_statement": "theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n", "formal_proof": "  /- We apply the distributive property to get\\begin{align*}\n  7(3y+2) &= 7\\cdot 3y+7\\cdot 2\\\\\n  &= 21y+14.\n  \\end{align*}\n  -/\n  ring"}
{"name": "mathd_algebra_116", "split": "valid", "informal_prefix": "/-- For what real value of $k$ is $\\frac{13-\\sqrt{131}}{4}$ a root of $2x^2-13x+k$? Show that it is $\\frac{19}{4}$.-/\n", "formal_statement": "theorem mathd_algebra_116 (k x : ℝ) (h₀ : x = (13 - Real.sqrt 131) / 4)\n    (h₁ : 2 * x ^ 2 - 13 * x + k = 0) : k = 19 / 4 := by\n", "formal_proof": "  /- We could substitute $(13-\\sqrt{131})/4$ for $x$ in the equation, but the quadratic formula suggests a quicker approach. Substituting $2$, $-13$, and $k$ into the quadratic formula gives  \\[\n  \\frac{-(-13)\\pm\\sqrt{(-13)^2-4(2)(k)}}{2(2)}= \\frac{13\\pm\\sqrt{169-8k}}{4}.\n  \\]Setting $(13+\\sqrt{169-8k})/4$ and $(13-\\sqrt{169-8k})/4$ equal to $(13-\\sqrt{131})/4$, we find no solution in the first case and $169-8k=131$ in the second case.  Solving yields $k=(169-131)/8=38/8=\\frac{19}{4}$.\n  -/\n  rw [h₀] at h₁\n  rw [eq_comm.mp (add_eq_zero_iff_neg_eq.mp h₁)]\n  norm_num\n  rw [pow_two]\n  rw [mul_sub]\n  rw [sub_mul, sub_mul]\n  rw [Real.mul_self_sqrt _]\n  ring\n  linarith"}
'''

def get_prompts(num_prompts="all"):
    prompts = []
    with open("./datasets/minif2f_valid_few_shot.jsonl", "r") as f:
        for i, line in enumerate(f):
            if (num_prompts!="all") and (i >= num_prompts):
                break
            prompt = json.loads(line)
            # append the string "informal_prefix: {prompt['informal_prefix']} \n formal_statement: {prompt['formal_statement']} \n formal_proof: {prompt['formal_proof']}" to prompts
            prompts.append(f"informal_prefix: {prompt['informal_prefix']} \n formal_statement: {prompt['formal_statement']} \n formal_proof: {prompt['formal_proof']}")
    return prompts


if __name__ == "__main__":
    NUM_PROMPTS = "all"
    for i, prompt in tqdm(enumerate(get_prompts(NUM_PROMPTS))):
        response = query_gpt4(prompt, model="gpt-4")
        # print(f"GPT-4 Response to prompt number {i+1}:\n\n", response)

        # Extract the problem statement and proof from the response
        problem_statement = "NO_STATEMENT"
        proof = "NO_PROOF"
        
        if "BEGINPROBLEMSTATEMENT:" in response and "ENDPROBLEMSTATEMENT" in response:
            problem_statement = response.split("BEGINPROBLEMSTATEMENT:")[1].split("ENDPROBLEMSTATEMENT")[0].strip()
        
        if "BEGINPROOF:" in response and "ENDPROOF" in response:
            proof = response.split("BEGINPROOF:")[1].split("ENDPROOF")[0].strip()
        
        # Print the extracted problem statement and proof
        # print(f"Problem Statement for prompt number {i+1}:\n\n{problem_statement}\n")
        # print(f"Proof for prompt number {i+1}:\n\n{proof}\n")
        
        with open("src/deformalization/f2f_valid_few_shot_deformalized.jsonl", "a") as outfile:
            json.dump({
                "prompt_number": i+1,
                "problem_statement": problem_statement,
                "proof": proof
            }, outfile)
            outfile.write("\n")