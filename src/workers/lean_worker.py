import json
import multiprocessing
import os
import queue
import time
import traceback
from typing import Optional

import pexpect

from src.workers.worker import *

HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = 'mathlib4/'

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"


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
        self.setup_repl()
        self.logger.info("Lean4 REPL setup.")

    def setup_repl(self):
        self.child = pexpect.spawn(
            f"{DEFAULT_LAKE_PATH} exe repl",
            cwd=DEFAULT_LEAN_WORKSPACE)

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
        # print(cmd_json)
        self.child.send(cmd_json + "\r\n")
        # Read the input itself.
        # This should be printed instantly, so timeout is set to 1 second.
        self.child.expect_exact(cmd_json + "\r\n", timeout=timeout_cat)
        assert self.child.after.decode('utf-8') == cmd_json + "\r\n"
        # print("Sent code to Lean4 REPL.")

        # Read the output.
        # This code is critical; the repl seems to print out some
        # strange non-json stuff before the actual json output,
        # including characters that delete the previous input,
        # such that it doesn't show up in debug output.
        self.child.expect_exact("{", timeout=timeout_start)
        res = "{"
        # print("Received start of output from Lean4 REPL.")
        # while res is not a valid json string, read lines.
        # All of the lines should print essentially instantly,
        # so there are no timeouts in this loop.
        start_time = time.time()
        while True:
            res = res + self.child.readline().decode('utf-8')
            try:
                # print all chars in res
                json.loads(res.strip())
                break
            except json.JSONDecodeError as e:
                # print(e)
                pass
            if time.time() - start_time > timeout_finish:
                raise TimeoutError("Lean4 REPL timed out.")
            # time.sleep(0.1)

        # kill
        if kill:
            self.child.close()
        return json.loads(res)

    def loop(self):
        input_data = self.deque_task(
            channel="lean",
            timeout=30
        )
        if input_data is None:
            self.logger.info("No tasks to complete.")
            # Spinlock, disappointing, but there's nothing to do.
            return

        self.logger.info(f"Received task: {input_data['task']}")
        result = self.send_code_read_json({
            "cmd": input_data['task'],
            # "allTactics": True,
            # "tactics": True,
            "env": 0
        })
        self.logger.info(f"Processed task.")

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

        self.enqueue(
            response=result,
            channel=input_data['channel']  # The response channel.
        )
        self.logger.info(str(result))
