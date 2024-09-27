import json
import logging
import multiprocessing
import os
import queue
import time
from typing import Dict

# import torch
from vllm import LLM, SamplingParams


def main(
        task_id: int,
        num_tasks: int,
        json_name: str,
        master_queue: multiprocessing.Queue,
        worker_queues: Dict[int, multiprocessing.Queue],
        completion_queue: multiprocessing.Queue,
        completion_batch_size: int,
        custom_eos: list
):
    """
    Entry point for the lean worker process.
    """

    # give myself a custom logging file.
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"logs/completion_worker_{task_id}.log")
    logger.addHandler(fh)
    logger.info(f"Starting completion worker {task_id}.")

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

    # print(f"Worker {task_id} detected the following devices: {device_counts}")

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

    #     # TODO: stuff all of the configs into a config file.
    llm = LLM(model="deepseek-ai/DeepSeek-Prover-V1.5-RL",
              max_num_batched_tokens=8192,
              trust_remote_code=True,
              dtype="float16",
              tensor_parallel_size=4)
    # else:
    #     raise ValueError(
    #         "You probably need to add a new device to the list of supported devices.")

    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        stop=custom_eos
    )

    while True:
        # check for kill signals from the master queue.
        try:
            kill_signal = master_queue.get_nowait()
            print(f"Worker {task_id} received kill signal: {kill_signal}")
            if kill_signal == "kill":
                break
        except queue.Empty:
            pass

        my_tasks = []
        # tasks should take the form
        # {
        #   'worker_id': int, # The worker task id that generated this task.
        #   'completion_task_id': int, # The specific completion task id of this task.
        #   'task': str # The task to complete, a string prompt.
        #   'type': str # Should be 'completion'
        # }
        while len(my_tasks) < completion_batch_size:
            try:
                my_tasks.append(completion_queue.get_nowait())
            except queue.Empty:
                break

        for task in my_tasks:
            assert task['type'] == 'completion'

        if len(my_tasks) == 0:
            # Spinlock, disappointing, but there's nothing to do.
            time.sleep(1)
        else:
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
                    'completion_task_id': my_tasks[i]['completion_task_id'],
                    'output': outputs[i].outputs[0].text,
                    'type': 'completion'
                }

                worker_queues[my_tasks[i]['worker_id']].put(result)
