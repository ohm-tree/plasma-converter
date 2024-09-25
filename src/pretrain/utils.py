'''
Helper functions for loading pre-training data and parsing proof segments.
'''

import json
import ijson
import os
import subprocess
import tempfile
import time
import traceback
from src.games.ast_parser import lean4_parser
from pprint import pprint

HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = 'mathlib4/'

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"

class AttrDict(dict):
    """A dictionary that allows attribute-style access."""
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

def verify_lean4_file(code,
    lake_path=DEFAULT_LAKE_PATH,
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
            outputs = subprocess.run([lake_path, "exe", 'repl'], stdin=temp_file,
                                     capture_output=True, text=True, cwd=lean_workspace, timeout=timeout)
        result = json.loads(outputs.stdout)

        errors =  [m for m in result.get('messages', []) if m['severity'] == 'error']
        if len(errors) > 0 and errors[0]['data']=='no goals to be solved':
            first_line_after = errors[0]['pos']['line'] - 1
            lines = code.splitlines()[:first_line_after]
            # Join the lines back into a single string with line breaks
            prefix = '\n'.join(lines)
            return prefix

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
                 code,
                 formal_statement,
                 header = LEAN4_DEFAULT_HEADER,
                 tailer = "",
                 ):
        
        self.code = code
        self.header = header
        self.formal_statement = formal_statement
        self.tailer = tailer
        self.full_code = ''.join([self.header, self.formal_statement, code.rstrip(' \n'), self.tailer])
        print("old full_code".center(80, '-'))
        print(self.full_code)

        self.result = verify_lean4_file(self.full_code, ast = True, tactics = True)
        if type(self.result) is not dict:
            new_full_code = self.result
            # print("self.code".center(80, '-'))
            # print(self.code)
            suffix_start = new_full_code.find(self.code)
            code = new_full_code[suffix_start:]
            self.code = code
            self.full_code = new_full_code
            # print("new_full_code".center(80, '-'))
            # print(new_full_code)
            # print("code".center(80, '-'))
            # print(code)
            self.result = verify_lean4_file(self.full_code, ast = True, tactics = True)
            
        self._parse_full_code_lines()

        print("full_code".center(80, '-'))
        print(self.full_code)


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
        truncate_pos = len(self.full_code) - len(self.tailer)
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
        return segments



# Example
"""
{
    "natural_language_statement": "$29\\cdot31+37-41\\equiv 3\\pmod{4}$",
    "answer": "3",
    "tags": [
        "number_theory",
        "equation",
        "modular_arithmetic"
    ],
    "formal_statement": "theorem lean_workbook_18 :\n  (29 * 31 + 37 - 41) % 4 = 3  :=  by sorry",
    "split": "lean_workbook",
    "proof": [
        "simp [Nat.mod_lt]",
        "simp only [Nat.add_mod_left, Nat.mod_eq_zero_of_dvd, dvd_pow]",
        "simp only [Nat.gcd_add_self_right]",
        "simp only [Nat.add_mod, Nat.mul_mod, Nat.mod_mod]",
        "simp only [Nat.mod_add_div]",
        "exact (by norm_num : (29 * 31 + 37 - 41) % 4 = 3)",
        "simp only [Nat.mul_comm, Nat.mul_assoc, Nat.mul_left_comm]",
        "norm_num [Int.add_emod]",
        "simp only [Nat.mul_mod_right]",
        "simp only [add_comm]",
        "simp only [Nat.gcd]",
        "simp only [Nat.mod_eq_zero_of_dvd, dvd_pow]",
        "simp only [Nat.mul_mod, Nat.mod_mod]",
        "simp [Mod.mod]",
        "simp only [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc, Nat.sub_sub]",
        "simp only [Nat.add_comm, Nat.add_left_comm]",
        "simp only [Nat.add_sub_assoc, Nat.mod_self]",
        "simp only [mod_eq_sub_mod]",
        "norm_num [Nat.mod_eq_of_lt]",
        "simp only [Nat.add_mod, Nat.mul_mod, Nat.mod_mod, Nat.mod_eq_zero_of_dvd]",
        "exact Nat.mod_mod 3 4",
        "norm_num [Nat.mul_mod, Nat.add_mod, Nat.mod_mod]",
        "simp only [Nat.mod_mod]",
        "exact rfl",
        "simp only [Nat.add_sub_assoc]"
    ]
},
"""
# fit in to
"""
code,
formal_statement,
header = LEAN4_DEFAULT_HEADER,
tailer = "",
"""


def load_data(
        file_path: str = "data/prover-llm_v0/lean_workbook_with_proofs.json",
        num_samples: int = 1000) -> Proof:
    
    # load only first num_samples entries of json
    data = []
    with open(file_path, 'r') as f:
        # parse the file incrementally, looking for objects
        parser = ijson.items(f, 'item')  # 'item' assumes top-level is a list of items
        for i, item in enumerate(parser):
            if i >= num_samples:
                break
            data.append(item)

    # print(data[0])
    
    def format(raw_proof) -> Proof:
        code = '\n'.join('    ' + line for line in raw_proof['proof'])
        code = "  " + code
        print("code", code)
        formal_statement = raw_proof['formal_statement']
        return Proof(code=code,
                     formal_statement=formal_statement)

    return [format(raw_proof) for raw_proof in data[:num_samples]]


sample_amc = {
    "code": """simp_all only [Nat.one_eq_succ_zero, Nat.zero_eq, zero_add, Nat.add_succ, Nat.add_zero,
    Nat.succ_add]
  have h₁' : a * r = 2 := by simpa [h₀] using h₁
  have h₂' : a * r ^ 3 = 6 := by simpa [h₀] using h₂
  -- Now we can divide the two equations to eliminate $a$ and determine $r$
  have h₃ : r ^ 2 = 3 := by
    nlinarith
  -- Finally, we can substitute back to find $a$
  have h₄ : a = 2 / Real.sqrt 3 ∨ a = -(2 / Real.sqrt 3) := by
    apply eq_or_eq_neg_of_sq_eq_sq <;>
    field_simp <;>
    nlinarith
  simpa [h₀] using h₄
    """,
    "formal_statement": '''theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2)
  (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
  '''
}

if __name__ == "__main__":
    data = load_data(num_samples=2)
    # data = [Proof(code=sample_amc["code"], formal_statement=sample_amc["formal_statement"])]
    for proof in data:
        print("Segmentation".center(80, '-'))
        pprint(proof.segmentation())
        print("Proof".center(80, '-'))
        pprint(proof.result)
