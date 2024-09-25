import json
import multiprocessing
import os
import queue
import time
from typing import Dict

import pexpect

HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = 'mathlib4/'

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"


def send_code_read_json(cmd, timeout_start=30, timeout_finish=1):

    child = pexpect.spawn(
        f"{DEFAULT_LAKE_PATH} exe repl",
        cwd=DEFAULT_LEAN_WORKSPACE)

    cmd_json = json.dumps(cmd)
    print(cmd_json)
    child.send(cmd_json + "\r\n")
    # Read the input itself.
    # This should be printed instantly, so timeout is set to 1 second.
    child.expect_exact(cmd_json + "\r\n", timeout=1)
    assert child.after.decode('utf-8') == cmd_json + "\r\n"
    print("Sent code to Lean4 REPL.")

    # Read the output.
    # This code is critical; the repl seems to print out some
    # strange non-json stuff before the actual json output,
    # including characters that delete the previous input,
    # such that it doesn't show up in debug output.
    child.expect_exact("{", timeout=timeout_start)
    res = "{"
    print("Received start of output from Lean4 REPL.")
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
    child.close()
    return json.loads(res)


def main(
        task_id: int,
        num_tasks: int,
        json_name: str,
        master_queue: multiprocessing.Queue,
        worker_queues: Dict[int, multiprocessing.Queue],
        lean_queue: multiprocessing.Queue,
):
    """
    Entry point for the lean worker process.
    """

    while True:
        # check for kill signals from the master queue.
        try:
            kill_signal = master_queue.get_nowait()
            print(f"Worker {task_id} received kill signal: {kill_signal}")
            if kill_signal == "kill":
                break
        except queue.Empty:
            pass

        try:
            input_data = lean_queue.get(timeout=3)
            # tasks should take the form
            # {
            #   'worker_id': int, # The worker task id that generated this task.
            #   'lean_task_id': int, # The specific lean task id of this task.
            #   'task': str # The task to complete, a string prompt.
            # }

            result = send_code_read_json({
                "cmd": input_data['task'],
                "allTactics": True,
                "tactics": True,
                "ast": True
            })

            worker_queues[input_data['worker_id']].put(
                {
                    'lean_task_id': input_data['lean_task_id'],
                    'result': result
                }
            )

        except queue.Empty:
            time.sleep(1)
