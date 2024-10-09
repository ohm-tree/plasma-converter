from src.workers.types import *


def completion_entrypoint(*args, **kwargs):
    from src.workers.completion_worker import CompletionWorker
    worker = CompletionWorker(*args, **kwargs)
    worker.main()


def policy_value_entrypoint(*args, **kwargs):
    from src.workers.policy_value_worker import PolicyValueWorker
    worker = PolicyValueWorker(*args, **kwargs)
    worker.main()


def context_entrypoint(*args, **kwargs):
    from src.workers.policy_value_worker import ContextWorker
    worker = ContextWorker(*args, **kwargs)
    worker.main()


def fast_policy_value_entrypoint(*args, **kwargs):
    from src.workers.fast_pv_worker import FastPolicyValueWorker
    worker = FastPolicyValueWorker(*args, **kwargs)
    worker.main()


def lean_entrypoint(*args, **kwargs):
    from src.workers.lean_worker import LeanWorker
    worker = LeanWorker(*args, **kwargs)
    worker.main()


def mcts_inference_entrypoint(*args, **kwargs):
    from src.workers.mcts_inference_worker import MCTSWorker
    worker = MCTSWorker(*args, **kwargs)
    worker.main()


def linear_inference_entrypoint(*args, **kwargs):
    from src.workers.linear_inference_worker import LinearInferenceWorker
    worker = LinearInferenceWorker(*args, **kwargs)
    worker.main()
