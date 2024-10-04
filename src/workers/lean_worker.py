import multiprocessing
from typing import Dict, Optional


def main(
        config: dict,
        run_name: str,
        task_id: int,
        master_queue: multiprocessing.Queue,
        lean_queue: multiprocessing.Queue,
        worker_queues: Dict[int, multiprocessing.Queue],
        global_lean_queue: multiprocessing.Queue,
):
    """
    Entry point for the lean worker process.
    """
    import json
    import logging
    import os
    import queue
    import time

    import pexpect
        
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



    # I live in src/workers/
    WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.dirname(WORKER_DIR)
    ROOT_DIR = os.path.dirname(SRC_DIR)

    # give myself a custom logging file.
    os.makedirs(f"{ROOT_DIR}/logs/{run_name}", exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"logs/{run_name}/lean_worker_{task_id}.log")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(f"Starting lean worker {task_id}.")

    # To avoid everyone killing everyone else when the processes start,
    # we stagger the start times of the lean workers across 5 minutes.
    # time.sleep((task_id / num_tasks) * 300)
    # Actually, you should never do this it doesn't solve any problems.

    child = setup_repl()

    while True:
        # check for kill signals from the master queue.
        try:
            kill_signal = master_queue.get_nowait()
            logger.fatal(
                f"Worker {task_id} received kill signal: {kill_signal}")
            if kill_signal == "kill":
                break
        except queue.Empty:
            pass

        try:
            input_data = global_lean_queue.get(timeout=30)
            # tasks should take the form
            # {
            #   'mcts_worker_id': int, # The worker task id that generated this task.
            #   'lean_task_id': int, # The specific lean task id of this task.
            #   'task': str # The task to complete, a string prompt.
            #   'type': str # Should be 'lean'
            # }
            assert input_data['type'] == 'lean'

        except queue.Empty:
            continue

        result = send_code_read_json({
            "cmd": input_data['task'],
            "allTactics": True,
            "tactics": True,
            "env": 0
        }, _child=child)
        # if result is error, try restarting the repl.
        if 'system_error' in result:
            logger.error(
                f"Error in send_code_read_json: {result['system_error']}")
            try:
                child.close()
            except:
                pass
            child = setup_repl()
            result = send_code_read_json({
                "cmd": input_data['task'],
                "allTactics": True,
                "tactics": True,
                "env": 0
            }, _child=child)

        result = {
            'mcts_worker_id': input_data['mcts_worker_id'],
            'lean_task_id': input_data['lean_task_id'],
            'result': result,
            'type': 'lean'
        }
        worker_queues[input_data['mcts_worker_id']].put(
            result
        )
        logger.info(str(result))
