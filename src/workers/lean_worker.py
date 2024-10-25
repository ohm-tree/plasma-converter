import json
import multiprocessing
import os
import queue
import re
import time
import traceback
from typing import Optional

import pexpect
from numpy import full
from performance_logger import PerformanceLogger

from src.workers.worker import *

HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = 'mathlib4/'

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"

# Useful for debugging
"""
{"cmd" : "import Mathlib\nimport Aesop\n\nopen BigOperators Real Nat Topology Rat\n\n"}

{"cmd" : "theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2) (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by\n", "env": 0}

{"cmd" : "theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2) (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by\n  -- First, we want to re-write the condition about the second\n  -- and fourth terms of the geometric sequence using the definition of a geometric sequence\n  simp_all only [Nat.one_eq_succ_zero, Nat.zero_eq, zero_add, Nat.add_succ, Nat.add_zero,\n    Nat.succ_add]\n  have h₁' : a * r = 2 := by simpa [h₀] using h₁\n  have h₂' : a * r ^ 3 = 6 := by simpa [h₀] using h₂\n  -- Now we can divide the two equations to eliminate $a$ and determine $r$\n  have h₃ : r ^ 2 = 3 := by\n    nlinarith\n  -- Finally, we can substitute back to find $a$\n  have h₄ : a = 2 / Real.sqrt 3 ∨ a = -(2 / Real.sqrt 3) := by\n    apply eq_or_eq_neg_of_sq_eq_sq <;>\n    field_simp <;>\n    nlinarith\n  simpa [h₀] using h₄\n", "env" : 0}
"""


class LeanWorker(Worker):
    def __init__(self,
                 config: dict,
                 run_name: str,
                 task_id: int,
                 queues: dict[str, multiprocessing.Queue],
                 **kwargs  # Unused
                 ):
        super().__init__(
            name="lean" + "_" + str(task_id),
            worker_type="lean",
            worker_idx=task_id,
            queues=queues,
            run_name=run_name,
        )
        self.logger.info(
            f"Global Variables I can see: {globals().keys()}"
        )
        time.sleep(8 * task_id)  # 8 seconds staggered start
        self.setup_repl()
        self.logger.info("Lean4 REPL setup.")
        self.performance_logger = PerformanceLogger()

    def setup_repl(self):
        while True:
            try:
                self.child = pexpect.spawn(
                    f"{DEFAULT_LAKE_PATH} exe repl",
                    cwd=DEFAULT_LEAN_WORKSPACE,
                    timeout=600)

                # Use the unprotected version to avoid error-loops.
                self._send_code_read_json(
                    {
                        "cmd": LEAN4_DEFAULT_HEADER,
                        "allTactics": True,
                        "tactics": True,
                    },
                    timeout_start=600,
                    timeout_cat=600,
                    timeout_finish=600
                )
                self.logger.info("Just initialized Lean4 REPL.")
            except Exception as e:
                self.logger.error(traceback.format_exc())
                time.sleep(10)
                try:
                    self.child.close()
                except:
                    pass
            else:
                break

    def send_code_read_json(self, cmd, timeout_start=30, timeout_cat=30, timeout_finish=30, kill=False):
        try:
            return self._send_code_read_json(cmd, timeout_start=timeout_start, timeout_cat=timeout_cat, timeout_finish=timeout_finish, kill=kill)
        except Exception as e:
            self.logger.error(traceback.format_exc())
            return {'system_error': traceback.format_exc()}

    def _send_code_read_json(self, cmd, timeout_start=30, timeout_cat=30, timeout_finish=30, kill=False):
        """
        Previous thoughts:
        Note that there's actually no reason to make the timeouts super short. Timeouts aren't usually indicative
        of buggy code, they're just due to variance in the time it takes to run the code. So, we can just set them
        to be very long.
        New thoughts (2024-10-10):
        Actually, when the lean code is taking a super long time to run, this is usually because there's something
        about the lean code that's running which is causing it to take a super long time. This is reproducible
        in linear inference on testcases (20, 61, 141). So we're going to nuke all the timeouts to 30 seconds.
        The setup code is *actually* guaranteed to never fail, so we can just set the timeout to be 600 seconds there.
        """
        cmd_json = json.dumps(cmd)
        cmd_json = cmd_json.strip()  # Remove whitespace
        # print(cmd_json)

        self.child.send(cmd_json + "\r\n")
        # self.logger.debug(f"Waiting for start")

        self.child.expect_exact("<START>", timeout=timeout_start)
        # self.logger.debug(f"Found start")

        self.child.expect_exact("<STOP>", timeout=timeout_finish)
        # self.logger.debug(f"Found stop")
        # Read everything before <STOP>
        res: str = self.child.before.decode('utf-8')

        try:
            json_res = json.loads(res)
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Lean4 REPL output was not valid JSON:")
            self.logger.error(res)
            self.logger.error(str(list(res)))  # Show the exact characters
            raise e
        else:
            # self.logger.info("Lean4 REPL output was valid JSON.")
            pass

        # kill
        if kill:
            self.child.close()
            self.logger.error("Lean4 REPL killed.")
        return json_res

    def loop(self):
        input_data = self.deque_task(
            channel="lean",
            timeout=30
        )
        start_time = time.time()

        if input_data is None:
            # self.logger.info("No tasks to complete.")
            # Spinlock, disappointing, but there's nothing to do.
            return

        # self.logger.info(f"Received task: {input_data['task']}")
        result = self.send_code_read_json({
            "cmd": input_data['task'],
            # "allTactics": True,
            # "tactics": True,
            "env": 0
        })
        # self.logger.info(f"Processed task.")

        # if result is error, try restarting the repl.
        if 'system_error' in result:
            try:
                self.child.close()
            except:
                pass
            self.setup_repl()
            result = self.send_code_read_json({
                "cmd": input_data['task'],
                # "allTactics": True,
                # "tactics": True,
                "env": 0
            })

        full_result = input_data
        full_result.update(
            {
                'response': result
            }
        )

        end_time = time.time()

        self.enqueue(
            obj=full_result,
            channel=input_data['channel']  # The response channel.
        )
        # self.logger.info(str(result))

        self.performance_logger.log_query(end_time - start_time)
        self.performance_logger.occasional_log(self.logger)


def test_lean_worker():
    # This is a test function to ensure that the LeanWorker class is working correctly.

    class FakeWorker(LeanWorker):  # A fake worker to simulate the Worker class
        def __init__(self):
            # Do not initialize the parent class
            self.logger = logging.getLogger("FakeWorker")
            self.child = None

    worker = FakeWorker()

    print("Setting up REPL... (may take up to 30 seconds)")
    worker.setup_repl()
    print("REPL setup complete.")
    assert worker.child is not None  # Ensure that the child process is created

    # Test sending a command
    command = {
        "cmd": "theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2) (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by\n", "env": 0}

    command_2 = {
        "cmd": "theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2) (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by\n  -- First, we want to re-write the condition about the second\n  -- and fourth terms of the geometric sequence using the definition of a geometric sequence\n  simp_all only [Nat.one_eq_succ_zero, Nat.zero_eq, zero_add, Nat.add_succ, Nat.add_zero,\n    Nat.succ_add]\n  have h₁' : a * r = 2 := by simpa [h₀] using h₁\n  have h₂' : a * r ^ 3 = 6 := by simpa [h₀] using h₂\n  -- Now we can divide the two equations to eliminate $a$ and determine $r$\n  have h₃ : r ^ 2 = 3 := by\n    nlinarith\n  -- Finally, we can substitute back to find $a$\n  have h₄ : a = 2 / Real.sqrt 3 ∨ a = -(2 / Real.sqrt 3) := by\n    apply eq_or_eq_neg_of_sq_eq_sq <;>\n    field_simp <;>\n    nlinarith\n  simpa [h₀] using h₄\n", "env": 0}

    response = worker.send_code_read_json(command)
    print(response)  # Print the response for inspection

    response_2 = worker.send_code_read_json(command_2)
    print(response_2)  # Print the response for inspection


if __name__ == "__main__":
    test_lean_worker()
