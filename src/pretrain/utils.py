'''
Helper functions for loading pre-training data and parsing proof segments.
'''

import json
import os
import subprocess
import tempfile
import time
import traceback
from typing import List
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
                 code: List[str],
                 formal_statement,
                 header = LEAN4_DEFAULT_HEADER,
                 segments = None,
                 problem_name: str = ""
                 ):
        
        self.code = code
        self.header = header
        self.formal_statement = formal_statement
        self.full_code = ''.join([self.header, self.formal_statement, code.rstrip(' \n'), self.tailer])

        self.result = verify_lean4_file(self.full_code, ast = True, tactics = True)
            
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




def load_IMO_problems(
        file_path: str = "data/prover-llm_v0/IMO_problems/Bulgaria1998P1.lean"
        )
    
    # load only first num_samples entries of json
    data = []
    with open(file_path, 'r') as f:
        data = json.load(f)
    data = [data[i] for i in samples_idx]
    def format(raw_proof) -> Proof:
        code = '\n'.join('  ' + line for line in raw_proof['proof'])
        formal_statement = raw_proof['formal_statement']
        return Proof(code=code,
                     formal_statement=formal_statement)

    return [format(raw_proof) for raw_proof in data]


if __name__ == "__main__":
    data = load_data(samples_idx=[0])
    # data = [Proof(code=sample_amc["code"], formal_statement=sample_amc["formal_statement"])]
    for proof in data:
        print("Segmentation".center(80, '-'))
        pprint(proof.segmentation())
        print("Proof".center(80, '-'))
        pprint(proof.result)
