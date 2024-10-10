import json
import logging
import multiprocessing
import os
import queue
import time
from typing import Dict, Optional

import pexpect

from src.workers.types import LeanTaskType, LeanWorkerType
from src.workers.worker import *

HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = 'mathlib4/'

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"


def send_code_read_json(cmd, timeout_start=600, timeout_cat=600, timeout_finish=600, _child: Optional[pexpect.spawn] = None, kill=False):
    try:
        return _send_code_read_json(cmd, timeout_start=timeout_start, timeout_cat=timeout_cat, timeout_finish=timeout_finish, _child=_child, kill=kill)
    except Exception as e:
        return {'system_error': str(e)}


def _send_code_read_json(cmd, timeout_start=600, timeout_cat=600, timeout_finish=600, _child: Optional[pexpect.spawn] = None, kill=False):
    """
    Note that there's actually no reason to make the timeouts super short. Timeouts aren't usually indicative
    of buggy code, they're just due to variance in the time it takes to run the code. So, we can just set them
    to be very long.
    """
    if _child is None:
        child = pexpect.spawn(
            f"{DEFAULT_LAKE_PATH} exe repl",
            cwd=DEFAULT_LEAN_WORKSPACE)
    else:
        child = _child

    cmd_json = json.dumps(cmd)
    # print(cmd_json)
    child.send(cmd_json + "\r\n")
    # Read the input itself.
    # This should be printed instantly, so timeout is set to 1 second.
    child.expect_exact(cmd_json + "\r\n", timeout=timeout_cat)
    assert child.after.decode('utf-8') == cmd_json + "\r\n"
    # print("Sent code to Lean4 REPL.")

    # Read the output.
    # This code is critical; the repl seems to print out some
    # strange non-json stuff before the actual json output,
    # including characters that delete the previous input,
    # such that it doesn't show up in debug output.
    child.expect_exact("{", timeout=timeout_start)
    res = "{"
    # print("Received start of output from Lean4 REPL.")
    # while res is not a valid json string, read lines.
    # All of the lines should print essentially instantly,
    # so there are no timeouts in this loop.
    start_time = time.time()
    while True:
        res = res + child.readline().decode('utf-8')
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
        child.close()
    return json.loads(res)


def setup_repl():
    child = pexpect.spawn(
        f"{DEFAULT_LAKE_PATH} exe repl",
        cwd=DEFAULT_LEAN_WORKSPACE)

    # Use the unprotected version to avoid error-loops.
    _send_code_read_json(
        {
            "cmd": LEAN4_DEFAULT_HEADER,
            "allTactics": True,
            "tactics": True,
        },
        _child=child
    )
    return child


class LeanWorker(Worker):
    def __init__(self,
                 config: dict,
                 run_name: str,
                 task_id: int,
                 queues: Dict[Union[TaskType, WorkerIdentifer], multiprocessing.Queue],
                 **kwargs  # Unused
                 ):
        super().__init__(
            worker_id=WorkerIdentifer(
                LeanWorkerType, task_id),
            queues=queues,
            run_name=run_name,
        )
        self.child = setup_repl()
        logging.info("Lean4 REPL setup.")

    def loop(self):
        input_data = self.deque_task(
            channel=LeanTaskType,
            timeout=30
        )
        if input_data is None:
            logging.info("No tasks to complete.")
            # Spinlock, disappointing, but there's nothing to do.
            return

        logging.info(f"Received task: {input_data['task']}")
        result = send_code_read_json({
            "cmd": input_data['task'],
            # "allTactics": True,
            # "tactics": True,
            "env": 0
        }, _child=self.child)
        # if result is error, try restarting the repl.
        if 'system_error' in result:
            self.logger.error(
                f"Error in send_code_read_json: {result['system_error']}")
            try:
                self.child.close()
            except:
                pass
            self.child = setup_repl()
            result = send_code_read_json({
                "cmd": input_data['task'],
                # "allTactics": True,
                # "tactics": True,
                "env": 0
            }, _child=self.child)

        self.enqueue_response(
            response=result,
            task=input_data
        )
        self.logger.info(str(result))
