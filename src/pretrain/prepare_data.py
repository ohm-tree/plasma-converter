'''
Helper functions for loading pre-training data and parsing proof segments.
'''

import json
import os
import re
import subprocess
import tempfile
import time
import traceback
from pprint import pprint
from typing import list

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.games.ast_parser import lean4_parser

HOME_DIR = os.path.expanduser('~')
# 'lake build' should run itself
DEFAULT_LAKE_dir = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = 'mathlib4/'

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"

POLICY_PROMPT = "What is the next step? This is the remainder of the proof:"


class AttrDict(dict):
    """A dictionary that allows attribute-style access."""

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


def verify_lean4_file(code,
                      lake_dir=DEFAULT_LAKE_dir,
                      lean_workspace=DEFAULT_LEAN_WORKSPACE,
                      last_env=None,
                      verbose=False,
                      timeout=300,
                      allTactics=False,
                      ast=False,
                      premises=False,
                      tactics=False):
    command = dict(cmd=code, allTactics=allTactics, ast=ast,
                   tactics=tactics, premises=premises)
    if last_env is not None:
        command.update(env=last_env)
    message_str = json.dumps(command, ensure_ascii=False)
    if verbose:
        print(message_str)
    start_time = time.time()
    system_messages = ''
    try:
        with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as temp_file:
            temp_file.write(message_str + "\r\n\r\n")
            temp_file.seek(0)
            outputs = subprocess.run([lake_dir, "exe", 'repl'], stdin=temp_file,
                                     capture_output=True, text=True, cwd=lean_workspace, timeout=timeout)
        result = json.loads(outputs.stdout)

        ast_results = lean4_parser(
            code, result['ast']) if 'ast' in result and result['ast'] else {}
        result = {
            "sorries": result.get('sorries', []),
            "tactics": result.get('tactics', []),
            "errors": [m for m in result.get('messages', []) if m['severity'] == 'error'],
            "warnings": [m for m in result.get('messages', []) if m['severity'] == 'warning'],
            "infos": [m for m in result.get('messages', []) if m['severity'] == 'info'],
            "system_messages": system_messages,
            "system_errors": None,
            "ast": ast_results,
            "verified_code": code,
            "env": result.get('env', None),
        }
        result['pass'] = not result['errors']
        result['complete'] = result['pass'] and not result['sorries'] and not any(
            "declaration uses 'sorry'" in warning['data'] or 'failed' in warning['data'] for warning in result['warnings'])
    except:
        result = {
            "pass": False,
            "complete": False,
            "system_errors": traceback.format_exc(),
            "system_messages": system_messages
        }
    result['verify_time'] = time.time() - start_time
    return result


class Proof(object):
    def __init__(self,
                 code: list[str],
                 formal_statement,
                 header=LEAN4_DEFAULT_HEADER,
                 segments=None,
                 problem_name: str = "",
                 nl_statement: str = "",
                 ):

        self.code = code
        self.header = header
        self.formal_statement = formal_statement
        self.full_code = ''.join(
            [self.header, self.formal_statement, code.rstrip(' \n')])

        print("verifying")
        self.result = verify_lean4_file(self.full_code, ast=True, tactics=True)
        print("verified")

        self._parse_full_code_lines()

        print("full_code".center(80, '-'))
        print(self.full_code)

        self.segments = self.segmentation()

    def _parse_full_code_lines(self):
        """
        Cache some basic string operations:
        the line offset of each line in the full code,
        and the full code lines.
        """
        self._full_code_lines = self.full_code.split('\n')
        self._line_offset, _offset = [], -1
        for _line in self._full_code_lines:
            _offset += 1  # '\n'
            self._line_offset.append(_offset)
            _offset += len(_line)

    def _get_idx(self, pos_info):
        """
        Convert a (line, column) dict to the index of the character in the full code.
        """
        return self._line_offset[pos_info['line'] - 1] + pos_info['column']

    def segmentation(self):
        result = self.result
        if 'errors' not in self.result:
            # compiler timeout
            return []

        """
        First, we need to find the last valid tactic.

        Unsolved goals also show up as errors, and we ignore them.
        """

        _prefix_len = len(self.header) + len(self.formal_statement)
        truncate_pos = len(self.full_code)
        for info in result['sorries'] + result['errors']:
            if info.get('data', str()).lstrip().startswith('unsolved goals'):
                continue
            info_pos = self._get_idx(info['pos'])
            # if info_pos >= _prefix_len:
            truncate_pos = min(truncate_pos, info_pos)

        if truncate_pos <= _prefix_len:
            # all proof lines are invalid
            return []

        partial_code = self.full_code[:truncate_pos]

        code_lines = partial_code.split('\n')
        pos_last, segments = _prefix_len, []
        for line_idx in range(len(code_lines)):
            if self._line_offset[line_idx] >= _prefix_len:

                def compute_last_valid_char_pos(line):
                    idx, last_non_blank = 0, len(line) + 1
                    while idx < len(line):
                        if line[idx: idx+2] == '--':
                            return last_non_blank
                        elif line[idx: idx+2] == '/-':
                            if '-/' not in line[idx+2:]:
                                # cannot split in this line
                                return len(line) + 1
                            idx = line.find('-/', idx+2) + 1
                        elif line[idx] != ' ':
                            last_non_blank = idx
                        idx += 1
                    return last_non_blank

                line_lastChar = self._line_offset[line_idx] + \
                    compute_last_valid_char_pos(code_lines[line_idx])
                line_endPos = self._line_offset[line_idx] + \
                    len(code_lines[line_idx])

                pos_min, goal = 1e9, None
                for tactic_info in result['ast']['tactics']:
                    pos, endPos = tactic_info['pos'], tactic_info['endPos']
                    if line_lastChar <= endPos and endPos <= line_endPos and pos < pos_min:
                        pos_min = pos
                        goal = tactic_info['stateAfter']
                if goal is None:
                    continue

                for tactic_info in result['ast']['tactics']:
                    pos, endPos = tactic_info['pos'], tactic_info['endPos']
                    if pos_last < endPos and endPos <= line_endPos and pos < pos_min:
                        pos_min = pos

                while pos_min > 0 and partial_code[pos_min - 1] != '\n':
                    pos_min -= 1
                indent_len = 0
                while partial_code[pos_min + indent_len] == ' ':
                    indent_len += 1
                newline_with_indent = '\n' + ' ' * indent_len

                segments.append(AttrDict(
                    tactic_code=partial_code[pos_last: line_endPos] + '\n',
                    state_comment=newline_with_indent.join([
                        ' ' * indent_len + '/- tactic state:',
                        '  ' + goal.replace('\n',
                                            newline_with_indent + '  '),
                        '-/\n'
                    ]),
                    goal=goal,
                    indent=indent_len,
                ))
                pos_last = line_endPos + 1
        if result['complete'] and (len(segments) == 0 or segments[-1].goal != 'no goals' or segments[-1].indent != segments[0].indent):
            indent_len = 2 if len(segments) == 0 else segments[0].indent
            newline_with_indent = '\n' + ' ' * indent_len
            segments.append(AttrDict(
                tactic_code=partial_code[pos_last:].rstrip(' \n') + '\n',
                state_comment=newline_with_indent.join([
                    ' ' * indent_len + '/- tactic state:',
                    '  no goals',
                    '-/\n'
                ]),
                goal='no goals',
                indent=indent_len,
            ))
        segments = [seg for seg in segments if len(
            seg.tactic_code.strip(' \n')) > 0]

        # segments.insert(0, {
        #     "tactic_code": "",
        #     "indent": 2,
        #     "state_comment": "  /- tactic state:\n  " + self.formal_statement[self.formal_statement.find(':') + 2:]
        # })
        return segments


def load_workbook_problems(
        data_dir: str = "/home/ubuntu/ohm-tree-filesys/plasma-converter/datasets/lean_workbook_with_proofs.json",
        save_dir: str = "data/prover-llm_v0/lean_workbook_parsed.jsonl"
):
    with open(data_dir, 'r') as f:
        data = json.load(f)

    num_saved = 0
    with open(save_dir, 'r') as f:
        num_saved = sum(1 for _ in f)
    idx = 0
    for problem in data:
        formal_statement = problem['formal_statement']

        for i, proof in enumerate(problem['proof']):
            idx += 1
            if idx <= num_saved:
                continue
            print("Processing proof", i)
            code = '\n'.join('  ' + line for line in proof.split('\n'))
            formal_statement = problem['formal_statement']
            proof = Proof(code=code,
                          formal_statement=formal_statement,
                          nl_statement=problem['natural_language_statement'])
            if not proof.result['complete']:
                continue
            data = {
                "formal_statement": formal_statement,
                "nl_statement": problem['natural_language_statement'],
                "segments": proof.segments,
                "code_lines": proof._full_code_lines,
                "proof_index": i
            }
            # print("Data", data)
            with open(save_dir, 'ab') as f:
                f.write(json.dumps(data).encode('utf-8'))
                f.write(b'\n')
            break  # OOPS, whatever we can prioritize diverisry
        # break


def load_deepseekv1_problems():
    pass


def load_IMO_problems(file_dir: str = "data/prover-llm_v0/IMO_problems/Bulgaria1998P1.lean"):
    data = []
    with open(file_dir, 'r') as f:
        data = json.load(f)

    def format(raw_proof) -> Proof:
        code = '\n'.join('  ' + line for line in raw_proof['proof'])
        formal_statement = raw_proof['formal_statement']
        return Proof(code=code,
                     formal_statement=formal_statement)

    return [format(raw_proof) for raw_proof in data]


def annotate_proof():
    # first run policy
    pass


def value_policy_prompter(partial_code: list[str], tactic_comment: str) -> str:
    """
    Generates a prompt for the model based on the current game state and a comment.
    """
    return "\n".join(partial_code) + "\n" + tactic_comment


def internlm_prompter(full_code, i: int, comment: str = ""):
    prompt = "```lean4\n"
    prompt += "\n".join(full_code[: i])
    prompt += '\n'
    prompt += comment
    prompt += '```\n'
    prompt += POLICY_PROMPT + '\n'
    prompt += '```\n'
    prompt += '\n'.join(full_code[i:])
    prompt += '\n```'
    print("prompt", prompt)
    return prompt


def prepare_policy_data(
        proof_dir: str = "data/prover-llm_v0/lean_workbook_parsed.jsonl",
        save_dir: str = "data/prover-llm_v0/workbook_policy.jsonl",
        comments_dir: str = "data/prover-llm_v0/comments/comments_v2.txt"):
    tokenizer = AutoTokenizer.from_pretrained(
        "internlm/internlm2-math-7b", trust_remote_code=True)
    # Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
    model = AutoModelForCausalLM.from_pretrained(
        "internlm/internlm2-math-7b", trust_remote_code=True, torch_dtype=torch.float16).cuda()
    model = model.eval()

    with open(comments_dir, 'r') as f:
        comments = f.readlines()

    def get_log_probs(prompt: str, device: str = "cuda"):
        probs = []
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        for target in comments:
            target_ids = tokenizer.encode(
                target, return_tensors='pt').to(device)

            # Concatenate the prompt and target ids
            # Exclude the first token of target_ids if it's a special token like BOS
            input_ids = torch.cat([prompt_ids, target_ids[:, 1:]], dim=-1)

            # Get the length of the prompt to know where the target starts
            prompt_length = prompt_ids.size(1)

            # Run the model to get logits
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits

            # Compute log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)

            # Extract log probabilities for the target tokens
            target_log_probs = []
            for i in range(prompt_length, input_ids.size(1)):
                # The token ID at the current position
                token_id = input_ids[0, i]
                # Log probability of the token given the previous tokens
                token_log_prob = log_probs[0, i - 1, token_id]
                target_log_probs.append(token_log_prob.item())

            # Sum the log probabilities to get the total log probability of the target
            total_log_prob = sum(target_log_probs)
            probs.append(total_log_prob)

            # print("Log probabilities of the target tokens:", target_log_probs)
            # print(f"Cum. log probability of {target}:", total_log_prob)
        return probs

    with open(proof_dir, 'r') as f:
        proof_data = [json.loads(line) for line in f]
    with open(save_dir, 'r') as f:
        num_saved = sum(1 for _ in f)

    idx = 0
    for i, problem in enumerate(proof_data):
        formal_statement = problem['formal_statement']
        nl_statement = problem['nl_statement']
        segments = problem['segments']
        code_lines = problem['code_lines']

        prev_tactic_comment = "  -- " + nl_statement

        k = 0
        while ("by" not in code_lines[k]):
            k += 1
        k += 1
        for j in range(len(segments)):
            idx += 1
            if idx > num_saved:
                # index of segments['tactic_code'] in code_lines
                print("code_lines", code_lines)
                print("k", k)
                full_code = code_lines
                intern_prompt = internlm_prompter(
                    full_code, k, prev_tactic_comment)
                pv_prompt = value_policy_prompter(
                    full_code[:k], prev_tactic_comment)
                # print("prompt", prompt)

                data_point = {
                    "prompt": pv_prompt,
                    "logits": get_log_probs(intern_prompt)
                }

                print("data point")
                pprint(data_point)

                with open(save_dir, 'ab') as f:
                    f.write(json.dumps(data_point).encode('utf-8'))
                    f.write(b'\n')

                prev_tactic_comment = segments[j]['state_comment']
            k += segments[j]['tactic_code'].count('\n')
        # break


def len_to_value(len):
    return np.exp(-len)


def prepare_value_data(
    proof_dir: str = "data/prover-llm_v0/lean_workbook_parsed.jsonl",
    save_dir: str = "data/prover-llm_v0/workbook_value.jsonl"
):

    with open(proof_dir, 'r') as f:
        proof_data = [json.loads(line) for line in f]
    with open(save_dir, 'r') as f:
        num_saved = sum(1 for _ in f)

    idx = 0

    for i, problem in enumerate(proof_data):
        formal_statement = problem['formal_statement']
        nl_statement = problem['nl_statement']
        segments = problem['segments']
        code_lines = problem['code_lines']

        prev_tactic_comment = "  -- " + nl_statement

        k = 0
        while ("by" not in code_lines[k]):
            k += 1
        k += 1
        for j in range(len(segments)):
            idx += 1
            if idx > num_saved:
                print("code_lines", code_lines)
                print("k", k)
                full_code = code_lines
                prompt = value_policy_prompter(
                    full_code[:k], prev_tactic_comment)
                # print("prompt", prompt)

                data_point = {
                    "prompt": prompt,
                    "value": len_to_value(len(code_lines) - k - 1)
                }

                print("data point")
                pprint(data_point)

                with open(save_dir, 'ab') as f:
                    f.write(json.dumps(data_point).encode('utf-8'))
                    f.write(b'\n')

                prev_tactic_comment = segments[j]['state_comment']
            k += segments[j]['tactic_code'].count('\n')


if __name__ == "__main__":
    # load_workbook_problems()
    # # data = [Proof(code=sample_amc["code"], formal_statement=sample_amc["formal_statement"])]
    # for proof in data:
    #     print("Segmentation".center(80, '-'))
    #     pprint(proof.segmentation())
    #     # print("Proof".center(80, '-'))
    #     # pprint(proof.result)

    prepare_policy_data()
    prepare_value_data()

""" 
Output:

-----------------------------------full_code------------------------------------
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

theorem lean_workbook_18 : (29 * 31 + 37 - 41) % 4 = 3  :=  by 
  simp [Nat.mod_lt]
-----------------------------------full_code------------------------------------
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

theorem lean_workbook_26 (x : ℝ) (hx : 0 < x) : x - 1 ≥ Real.log x  :=  by 
  have h1 : 0 ≤ (x - 1)^2 := sq_nonneg (x - 1)
  nlinarith [log_le_sub_one_of_pos hx]
----------------------------------Segmentation----------------------------------
[{'goal': 'no goals',
  'indent': 2,
  'state_comment': '  /- tactic state:\n    no goals\n  -/\n',
  'tactic_code': '  simp [Nat.mod_lt]\n'}]
----------------------------------Segmentation----------------------------------
[{'goal': 'x : ℝ\nhx : 0 < x\nh1 : 0 ≤ (x - 1) ^ 2\n⊢ x - 1 ≥ x.log',
  'indent': 2,
  'state_comment': '  /- tactic state:\n'
                   '    x : ℝ\n'
                   '    hx : 0 < x\n'
                   '    h1 : 0 ≤ (x - 1) ^ 2\n'
                   '    ⊢ x - 1 ≥ x.log\n'
                   '  -/\n',
  'tactic_code': '  have h1 : 0 ≤ (x - 1)^2 := sq_nonneg (x - 1)\n'},
 {'goal': 'no goals',
  'indent': 2,
  'state_comment': '  /- tactic state:\n    no goals\n  -/\n',
  'tactic_code': '  nlinarith [log_le_sub_one_of_pos hx]\n'}]
"""
