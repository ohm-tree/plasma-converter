from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.games.lean_game import LeanGameState
from src.networks.network import Network


class ProverLLM(Network):
    def __init__(self,
                 base_model: Optional[AutoModelForCausalLM] = None,
                 heads_only: bool = False,
                 llm_only: bool = False,
                 random_flag: bool = False):
        """
        A ProverLLM model that uses a transformer-based language model for proof generation
        and MCTS decision-making. It implements:
        1. Given a game state, generate a prompt (either for the V/P-heads or for completion).
        2. Given this prompt, tokenize it.
        3. Given a tokenized prompt, run it through the base model to get an intermediate state.
        4. Use the intermediate state to get a value estimate and a policy output.

        The worker/MCTS immediately outside of ProverLLM should be agnostic to the LLM-related
        principles including the specific prompting methods etc. To this end,
        all worker/MCTS interations with this class ("public methods") should
        involve LeanGameStates only.

        The controller and training loop necessarily need to know about the internal
        details of the model; the five steps above can be short-circuited in different
        ways by the controller.

        Parameters:
        ----------
        base_model: Optional[AutoModelForCausalLM]
            An optional pre-trained base model. If None, a default model is loaded.
        heads_only: bool
            If True, only the heads of the model are initialized.
        random_flag: bool
            When this flag is true, the policy output is uniform and
            the value output is identically 0.
            This should be set to true by the workers during the very first iteration
            when there is no model to load yet.
            After the first iteration, this should be set to false.

        """
        super(ProverLLM, self).__init__()
        if heads_only and llm_only:
            raise ValueError(
                "Cannot set both heads_only and llm_only to True. (This is useless.)")

        self.heads_only = heads_only
        self.llm_only = llm_only
        if not heads_only:
            self.tokenizer = AutoTokenizer.from_pretrained(
                'deepseek-ai/DeepSeek-Prover-V1.5-RL',
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if base_model is None:
                self.base_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                    'deepseek-ai/DeepSeek-Prover-V1.5-RL',
                    trust_remote_code=True,
                    device_map='auto'
                )
            else:
                self.base_model = base_model

        if not llm_only:
            self.value_head = nn.Sequential(
                nn.Linear(4096, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            ).to("cuda")

            self.policy_head = nn.Sequential(
                nn.Linear(4096, 512),
                nn.ReLU(),
                nn.Linear(512, 100)
            ).to("cuda")

        self.random_flag = random_flag

########################## Utilities ##########################

    def set_random_flag(self, random_flag: bool):
        """
        Set the random flag for the policy.

        Parameters:
        ----------
        random_flag: bool
            A flag to indicate whether to use random actions in the policy.
        """
        self.random_flag = random_flag

    def policy_value_state_dict(self) -> dict:
        """
        Usually, we do not want to be saving/loading the entire model,
        just the policy and value heads.

        Returns:
        -------
        policy_value_state_dict: dict
            The state dictionaries for the policy and value heads.
        """
        if self.llm_only:
            raise ValueError("Cannot get state dicts in llm_only mode.")
        return {
            'policy_head': self.policy_head.state_dict(),
            'value_head': self.value_head.state_dict()
        }

    def load_policy_value_state_dict(self, policy_value_state_dict: dict):
        """
        Set the state dictionaries for the policy and value heads.

        Parameters:
        ----------
        policy_value_state_dict: dict
            The state dictionaries for the policy and value heads.
        """
        if self.llm_only:
            raise ValueError("Cannot set state dicts in llm_only mode.")
        self.policy_head.load_state_dict(
            policy_value_state_dict['policy_head'])
        self.value_head.load_state_dict(policy_value_state_dict['value_head'])

########################## Public Interface ##########################

    def mcts_forward(self, state: LeanGameState) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the model using MCTS.

        Parameters:
        ----------
        state: LeanGameState
            The current game state.

        Returns:
        -------
        policy_output: torch.Tensor
            The policy output from the policy head.
        value_output: torch.Tensor
            The value estimate from the value head.
        """
        if self.heads_only or self.llm_only:
            raise ValueError(
                "MCTS forward is not supported in heads_only or llm_only mode.")
        prompt = self.value_policy_prompter(state)
        input_ids, attention_mask = self.tokenize(prompt)
        intermediate_output = self.get_intermediate_state(
            input_ids, attention_mask)
        policy_output, value_output = self.policy_and_value(
            intermediate_output)
        return policy_output, value_output

    def forward(self, state: LeanGameState) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the forward pass of the network. Returns a policy and a value.

        Parameters:
        ----------
        state: LeanGameState
            The current game state.

        Returns:
        -------
        policy_output: torch.Tensor
            The policy output from the policy head.
        value_output: torch.Tensor
            The value estimate from the value head.
        """
        return self.mcts_forward(state)

    def rollout_forward(self, state: LeanGameState, comment: str, truncate_existing: bool) -> str:
        """
        Perform a forward pass through the model for a rollout.

        Parameters:
        ----------
        state: LeanGameState
            The current game state.
        comment: str
            A comment to include in the prompt.
        truncate_existing: bool
            Whether to truncate existing proof text.

        Returns:
        -------
        str
            The completed proof text.
        """
        if self.heads_only:
            raise ValueError("Rollout is not supported in heads_only mode.")
        prompt = self.rollout_prompter(state, comment)
        input_ids, attention_mask = self.tokenize(prompt)
        completed_proof = self.complete(input_ids, attention_mask)
        assert completed_proof.startswith(prompt)
        if truncate_existing:
            completed_proof = completed_proof[len(prompt):]
        return completed_proof


########################## Part 1: Given a game state, generate a prompt ##########################


    def rollout_prompter(self, state: LeanGameState, comment: str) -> str:
        """
        Generates a prompt for the model based on the current game state and a comment.
        """
        s = ""
        s += state.header
        s += state.problem
        s += state.code
        s += state.tactic_state
        s += comment + "\n"
        return s

    def value_policy_prompter(self, state: LeanGameState) -> str:
        """
        Generates a prompt for the model based on the current game state and a comment.
        """
        s = ""
        s += state.header
        s += state.problem
        s += state.code
        s += state.tactic_state
        return s

########################## Part 2: Given this prompt, tokenize it ##########################

    def tokenize(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenizes the given prompt and returns the input IDs and attention mask.

        Parameters:
        ----------
        prompt: str
            The prompt to tokenize.

        Returns:
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The input IDs and attention mask.
        """
        tokens = self.tokenizer(
            prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
        return tokens['input_ids'], tokens['attention_mask']

########################## Part 3: Given a tokenized prompt, run it through the base model to get a completion ##########################

    def complete(self, input_ids: torch.Tensor, attention_mask=None, max_length=1000) -> str:
        """
        Returns most likely completed proof (max'ed with 1000 tokens)

        Parameters:
        ----------
        input_ids: torch.Tensor
            The input token IDs.
        attention_mask: torch.Tensor
            The attention mask for the input.
        max_length: int
            The maximum length of the generated text.

        Returns:
        -------
        generated_text: str
            The generated proof text.
        """
        if self.heads_only:
            raise ValueError("Completion is not supported in heads_only mode.")

        base_output = self.base_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + max_length
        )
        generated_text = self.tokenizer.decode(
            base_output[0], skip_special_tokens=True)
        return generated_text

########################## Part 4: Given a tokenized prompt, run it through the base model to get an intermediate state ##########################

    def get_intermediate_state(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns the intermediate hidden state output from the base model.

        Parameters:
        ----------
        input_ids: torch.Tensor
            The input token IDs.
        attention_mask: torch.Tensor
            The attention mask for the input.

        Returns:
        -------
        intermediate_output: torch.Tensor
            The hidden state output from the base model.
        """
        if self.heads_only:
            raise ValueError(
                "Intermediate state retrieval is not supported in heads_only mode.")

        base_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = base_output.hidden_states

        intermediate_output = hidden_states[25][0][-1]
        # debug
        # print("Intermediate output shape:", intermediate_output.shape)
        return intermediate_output

########################## Part 5: Use the intermediate state to get a value estimate and a policy output ##########################

    def policy_and_value(self, intermediate_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns policy and value outputs for the given intermediate output.

        Parameters:
        ----------
        intermediate_output: torch.Tensor
            The intermediate output from the base model.

        Returns:
        -------
        policy_output: torch.Tensor
            The policy output from the policy head.
        value_output: torch.Tensor
            The value estimate from the value head.
        """

        if self.random_flag:
            # Return uniform policy and zero value
            value_output = torch.zeros(
                (intermediate_output.shape[0], 1)).to("cuda")
            policy_output = F.softmax(torch.ones(
                (intermediate_output.shape[0], 100)).to("cuda"), dim=-1)

            return value_output, policy_output

        policy_output = self.policy_head(intermediate_output)
        value_output = self.value_head(intermediate_output)

        return policy_output, value_output


def _test_prover_llm():
    model = ProverLLM()
    sample_input1 = '''
        Assumptions:
        h: 0 + succ (succ 0) = succ (succ (succ 0))

        Goal:
        False

        Complete the following Lean 4 proof:

        ```lean4
        /-- 2+2 !=5. -/
        example : succ (succ 0) + succ (succ 0) ≠ succ (succ (succ (succ (succ 0)))) := by
        intro h
        rw [succ_add] at h
        apply succ_inj at h
        rw [succ_add] at h
        apply succ_inj at h
        '''

    sample_input2 = r'''Complete the following Lean 4 code:

    ```lean4
    import Mathlib
    import Aesop

    set_option maxHeartbeats 0

    open BigOperators Real Nat Topology Rat

    /-- The second and fourth terms of a geometric sequence are $2$ and $6$. Which of the following is a possible first term?
    Show that it is $\frac{2\sqrt{3}}{3}$.-/
    theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2)
    (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
    '''

    sample_tokens = model.tokenizer(sample_input1, return_tensors='pt')[
        'input_ids'].to("cuda")
    sample_attention_mask = model.tokenizer(sample_input1, return_tensors='pt')[
        'attention_mask'].to("cuda")

    completed_proof = model.complete(
        sample_tokens, sample_attention_mask, max_length=200)
    # policy, value = model.mcts_forward(sample_tokens, sample_attention_mask)

    with open('../sample_data/sample_prover_llm_output3.txt', 'w') as f:
        f.write("Completed Proof:\n")
        f.write(completed_proof)
        # f.write("\n\nValue:\n")
        # f.write(str(value))
        # f.write("\n\nPolicy (logits):\n")
        # f.write(str(policy))


if __name__ == '__main__':
    _test_prover_llm()
