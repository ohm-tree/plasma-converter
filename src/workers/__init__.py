def completion_entrypoint(*args, **kwargs):
    from src.workers.completion_worker import CompletionWorker
    worker = CompletionWorker(*args, **kwargs)
    worker._main()


def policy_value_entrypoint(*args, **kwargs):
    from src.workers.policy_value_worker import PolicyValueWorker
    worker = PolicyValueWorker(*args, **kwargs)
    worker._main()


def context_entrypoint(*args, **kwargs):
    from src.workers.policy_value_worker import ContextWorker
    worker = ContextWorker(*args, **kwargs)
    worker._main()


def fast_policy_value_entrypoint(*args, **kwargs):
    from src.workers.fast_pv_worker import FastPolicyValueWorker
    worker = FastPolicyValueWorker(*args, **kwargs)
    worker._main()


def lean_entrypoint(*args, **kwargs):
    from src.workers.lean_worker import LeanWorker
    worker = LeanWorker(*args, **kwargs)
    worker._main()
