import multiprocessing
from typing import Dict, List


def main(
        run_name: str,
        completion_worker_id: int,
        num_completion_workers: int,
        json_name: str,
        gpu_set: List[int],
        master_queue: multiprocessing.Queue,
        completion_queue: multiprocessing.Queue,
        worker_queues: Dict[int, multiprocessing.Queue],
        global_completion_queue: multiprocessing.Queue,
        completion_batch_size: int,
        custom_eos: list
):
    """
    Entry point for the lean worker process.
    """
    import logging
    import os
    import queue

    # I live in src/workers/
    WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.dirname(WORKER_DIR)
    ROOT_DIR = os.path.dirname(SRC_DIR)

    # give myself a custom logging file.
    os.makedirs(f"{ROOT_DIR}/logs/{run_name}", exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(
        f"logs/{run_name}/completion_worker_{completion_worker_id}.log")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(f"Starting completion worker {completion_worker_id}.")

    # detect the type of gpus available.
    # If there is at least one A100 80GB or H100,
    # tensor parallelization is not needed.
    # Else, check that there are at least 4 V100s.
    # and set tensor_parallel_size=4

    # num_devices = torch.cuda.device_count()
    # device_counts = {}
    # for i in range(num_devices):
    #     device_name = torch.cuda.get_device_name(i)
    #     if device_name in device_counts:
    #         device_counts[device_name] += 1
    #     else:
    #         device_counts[device_name] = 1

    # print(f"Worker {completion_worker_id} detected the following devices: {device_counts}")

    # if "A100-SXM4-80GB" in device_counts or "H100-SXM4-80GB" in device_counts:
    #     tensor_parallel_size = 1
    #     llm = LLM(model="deepseek-ai/DeepSeek-Prover-V1.5-RL",
    #               max_num_batched_tokens=8192,
    #               trust_remote_code=True
    #               )
    # elif "Tesla V100-SXM2-16GB" in device_counts:
    #     if device_counts["Tesla V100-SXM2-16GB"] >= 4:
    #         tensor_parallel_size = 4
    #     else:
    #         raise ValueError("Not enough Tesla V100-SXM2-16GB GPUs available.")

    # Set up vllm stuff.
    import gc

    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_set))

    # TODO: stuff all of the configs into a config file.
    # llm = LLM(model="deepseek-ai/DeepSeek-Prover-V1.5-RL",
    #           max_num_batched_tokens=8192,
    #           trust_remote_code=True,
    #           dtype="float16",
    #           tensor_parallel_size=len(gpu_set))
    llm = LLM(model="deepseek-ai/DeepSeek-Prover-V1.5-RL",
              max_num_batched_tokens=8192,
              trust_remote_code=True,
              enforce_eager=True,
              tensor_parallel_size=len(gpu_set))

    # else:
    #     raise ValueError(
    #         "You probably need to add a new device to the list of supported devices.")

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        stop=custom_eos
    )

    while True:
        # check for kill signals from the master queue.
        try:
            kill_signal = completion_queue.get_nowait()
            print(
                f"Worker {completion_worker_id} received kill signal: {kill_signal}")
            if kill_signal == "kill":
                break
        except queue.Empty:
            pass

        my_tasks = []
        # tasks should take the form
        # {
        #   'mcts_worker_id': int, # The worker task id that generated this task.
        #   'completion_task_id': int, # The specific completion task id of this task.
        #   'task': str # The task to complete, a string prompt.
        #   'type': str # Should be 'completion'
        # }
        try:
            new_task = global_completion_queue.get(timeout=30)
        except queue.Empty:
            pass
        else:
            assert new_task['type'] == 'completion'
            my_tasks.append(new_task)

        while len(my_tasks) < completion_batch_size:
            try:
                task = global_completion_queue.get_nowait()
            except queue.Empty:
                break
            assert task['type'] == 'completion'
            my_tasks.append(task)

        logger.info(
            f"Worker {completion_worker_id} received {len(my_tasks)} tasks.")

        if len(my_tasks) == 0:
            # Spinlock, disappointing, but there's nothing to do.
            continue
        # We have tasks to complete.
        input_data = [
            my_tasks[i]['task']
            for i in range(len(my_tasks))
        ]
        outputs = llm.generate(
            input_data,
            sampling_params=sampling_params
        )
        for i in range(len(outputs)):
            result = {
                'mcts_worker_id': my_tasks[i]['mcts_worker_id'],
                'completion_task_id': my_tasks[i]['completion_task_id'],
                'output': outputs[i].outputs[0].text,
                'type': 'completion'
            }
            logger.info(str(result))

            worker_queues[my_tasks[i]['mcts_worker_id']].put(result)

    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm
    gc.collect()
    print("Completion worker is dead.")
    logger.info("Completion worker is dead.")
