import multiprocessing
import os
import sys
import time


def worker_1(b: multiprocessing.Barrier):
    print("Worker 1 started")
    from vllm import LLM, SamplingParams
    print("Worker 1 imported vllm")

    # set the visible GPUs to be 0 through 3.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # V100 settings.
    llm = LLM(model="deepseek-ai/DeepSeek-Prover-V1.5-RL",
              max_num_batched_tokens=8192,
              trust_remote_code=True,
              dtype="float16",
              tensor_parallel_size=2)
    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
    )

    b.wait()

    print("Worker 1 created LLM and sampling params")
    output = llm.generate("Hello, world! 2 + 2 = ",
                          sampling_params=sampling_params)
    print("Worker 1 finished sampling")
    print(output)

    b.wait()
    time.sleep(20)


def worker_2(b: multiprocessing.Barrier):
    print("Worker 2 started")
    from vllm import LLM, SamplingParams
    print("Worker 2 imported vllm")

    # set the visible GPUs to be 4 through 7.
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

    # V100 settings.
    llm = LLM(model="deepseek-ai/deepseek-math-7b-instruct",
              max_num_batched_tokens=8192,
              trust_remote_code=True,
              dtype="float16",
              tensor_parallel_size=2)

    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
    )

    b.wait()

    print("Worker 2 created LLM and sampling params")
    output = llm.generate("Hello, world! 3 + 4 = ",
                          sampling_params=sampling_params)
    print("Worker 2 finished sampling")
    print(output)

    b.wait()
    time.sleep(20)


def main():
    b = multiprocessing.Barrier(2)
    p1 = multiprocessing.Process(target=worker_1, args=(b,))
    p2 = multiprocessing.Process(target=worker_2, args=(b,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()


if __name__ == "__main__":
    main()
