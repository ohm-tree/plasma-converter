from abc import ABC, abstractmethod
from typing import Callable, Iterable

from src.workers.worker import TaskType, WorkerResponse, WorkerTask


def handler(t: TaskType):
    """
    Takes a function f which is an instance method,
    and a TaskType t. Registers this function with the TaskType,
    such that when messages of type t are received by the object,
    function f gets called on t.
    """
    def wrapper(func: Callable[[Any, WorkerResponse], WorkerTask]):
        func._task_type = t
        return func
    return wrapper


def on_startup(func):
    """
    Registers a function to be called on object startup.
    """
    func._on_startup = True
    return func


class ConcurrentClass(ABC):
    def __init__(self):
        self.handlers = {}
        for name in dir(self):
            attr = getattr(self, name)
            if hasattr(attr, '_task_type'):
                if attr._task_type in self.handlers:
                    raise ValueError(
                        f"TaskType {attr._task_type} already registered.")
                self.handlers[attr._task_type] = attr
        for name in dir(self):
            attr = getattr(self, name)
            if hasattr(attr, '_on_startup'):
                attr()

    def tick(self, messages: Iterable[WorkerResponse]) -> Iterable[WorkerTask]:
        """
        Ticks the game state.
        """
        for message in messages:
            if message.task_id.task_type in self.handlers:
                yield self.handlers[message.task_id.task_type](self, message)
            else:
                raise ValueError(
                    f"TaskType {message.task_id.task_type} not registered.")
