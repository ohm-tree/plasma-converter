import json
import multiprocessing
import os
import time
import traceback

import pexpect

from src.lean.lean_game import LEAN4_DEFAULT_HEADER
from src.workers.performance_logger import PerformanceLogger
from src.workers.worker import *

HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = 'mathlib4/'


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
        self.starting_time = time.time()
        # 15 seconds staggered start. I just want everyone to spawn without dying!
        time.sleep(15 * task_id)

        self.num_procs = config['num_procs']

        self.num_failures = 0

        self.repl_dead = False

        self.setup_repl()
        self.logger.info("Lean4 REPL setup.")
        self.performance_logger = PerformanceLogger()

    def setup_repl(self):
        self.logger.info("Beginning Lean4 REPL setup.")

        # Staggered start
        remaining_time = 15 * self.num_procs - \
            (time.time() - self.starting_time)
        if remaining_time > 0 and self.num_failures > 0:
            self.logger.info(
                f"Other Lean processes have not finished initializing yet. To avoid jamming up the REPL, sleeping for {remaining_time:.2f} seconds.")
            time.sleep(remaining_time)
        while True:
            self.num_failures += 1
            try:
                # self.child = pexpect.spawn(
                #     f"{DEFAULT_LAKE_PATH} exe repl",
                #     cwd=DEFAULT_LEAN_WORKSPACE,
                #     timeout=3600)
                self.child = pexpect.spawn(
                    "/bin/bash",
                    cwd=DEFAULT_LEAN_WORKSPACE,
                    timeout=3600
                )
                self.logger.info("Just started bash.")
                self.child.sendline("stty -icanon")  # Disable canonical mode
                self.logger.info("Just disabled canonical mode.")
                # Start the Lean REPL
                self.child.sendline(
                    f"{DEFAULT_LAKE_PATH} exe repl")
                self.logger.info("Just started REPL.")

                # Shaves off 50 ms, which is actually the main bottlneck for fast queries.
                self.child.delaybeforesend = None

                # Use the unprotected version to avoid error-loops.
                self._send_code_read_json(
                    {
                        "cmd": LEAN4_DEFAULT_HEADER,
                        "allTactics": True,
                        "tactics": True,
                    },
                    timeout_start=3600,
                    timeout_finish=3600
                )
                self.logger.info("Just initialized Lean4 REPL.")
            except Exception as e:
                try:
                    self.child.close()
                except:
                    pass

                self.logger.error(
                    f"Failed to start Lean4 REPL. Attempt {self.num_failures}. Sleeping for {(2 ** self.num_failures)} minutes.")
                self.logger.error(traceback.format_exc())
                time.sleep(60 * (2 ** self.num_failures))
                self.logger.error("Retrying...")
            else:
                break
        self.repl_dead = False

    def send_code_read_json(self, cmd, timeout_start=30, timeout_finish=1200):
        try:
            return self._send_code_read_json(cmd, timeout_start=timeout_start, timeout_finish=timeout_finish)
        except Exception as e:
            self.logger.error(f"Error on command: {cmd}")
            self.logger.error(traceback.format_exc())

            return {'system_error': traceback.format_exc()}

    def _send_code_read_json(self, cmd, timeout_start=30, timeout_finish=1200):
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

        Newer thoughts (2024-10-25):
        Actually, if you inspect the actual logs produced by performance_logger, you'll see that >99% of tasks complete
        in 5 seconds or less, and the 30 second barrier is just nuking all of the performance (because we have to wait
        for the Lean repl to die (30 seconds) plus re-start the lean repl (another 30+ seconds) then wait for the lean
        repl to die again (another 30 seconds) and then re-start it again (another 30+ seconds), leading to worst-case
        behavior of ~160 seconds). I'm not sure what the best way to capitalize on this information is.
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

        # Sometimes, the REPL itself outputs errors, most notably this one:
        # Error in Linarith.normalizeDenominatorsLHS: Expr.rewrite may not produce subgoals.

        # So we just slice until the position of the first curly brace.

        # Get everything after the first curly brace.
        res = res[res.find("{"):]

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

        return json_res

    def loop(self):

        if self.repl_dead:
            # try to start a new REPL
            try:
                self.child.close(force=True)
            except:
                pass
            self.setup_repl()

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
            "env": 0
        })
        # self.logger.info(f"Processed task.")

        latency = input_data.get('latency', 0)

        # if result is error, try restarting the repl.
        if 'system_error' in result:
            self.repl_dead = True

            """
            New thoughts (2024-10-25):
            Inspecting the logs, every single time a lean 4 repl dies, it dies again
            after it's bounced. So the best way to handle this is to never bounce those
            tasks; they're just definitely dead.
            """

            # # Check if the task has already been bounced.
            # # If it has, then we should return the result (error).
            # # If not, then we should try to bounce it once.
            # if input_data.get('bounced', False):
            #     self.logger.error(
            #         f"Task has already been bounced. Returning error.")
            #     full_result = input_data
            #     full_result.update(
            #         {
            #             'response': result
            #         }
            #     )
            #     self.enqueue(
            #         obj=full_result,
            #         channel=input_data['channel']  # The response channel.
            #     )
            #     end_time = time.time()

            #     self.performance_logger.log_query(
            #         end_time - start_time + latency)
            #     self.performance_logger.occasional_log(self.logger)

            #     return
            # else:
            #     self.logger.error(f"Task has not been bounced. Bouncing task.")
            #     input_data['bounced'] = True
            #     # Add latency to input_data
            #     input_data['latency'] = time.time() - start_time
            #     self.enqueue(  # Re-queue the task.
            #         obj=input_data,
            #         channel="lean"  # The global lean channel.
            #     )
            #     return

        # If we get here, we have a valid result.

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

        self.performance_logger.log_query(
            latency=end_time - start_time + latency,
            total_waiting_time=start_time - full_result['enqueue_time']
        )
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
