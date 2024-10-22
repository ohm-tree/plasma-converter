from typing import Callable, Tuple

from src.workers.types import *
from src.workers.worker import *


def lean_entrypoint(*args, **kwargs):
    from src.workers.lean_worker import LeanWorker
    worker = LeanWorker(*args, **kwargs)
    worker.main()


def mcts_inference_entrypoint(*args, **kwargs):
    from src.workers.mcts_inference_worker import MCTSWorker
    worker = MCTSWorker(*args, **kwargs)
    worker.main()


def linear_entrypoint(*args, **kwargs):
    from src.workers.linear_inference_worker import LinearWorker
    worker = LinearWorker(*args, **kwargs)
    worker.main()


def llm_entrypoint(*args, **kwargs):
    from src.workers.llm_worker import LLMWorker
    worker = LLMWorker(*args, **kwargs)
    worker.main()


WORKER_TYPES_AND_STRINGS: Tuple[Tuple[str, Callable, bool]] = (
    ('mcts', mcts_inference_entrypoint, False, True),
    ('linear',
     linear_entrypoint, False, True),
    ('lean', lean_entrypoint, False, False),
    ('context', llm_entrypoint, True, False),
    ('value', llm_entrypoint, True, False),
    ('completion', llm_entrypoint, True, False),
)
