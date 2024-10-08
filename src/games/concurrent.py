from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional

from src.workers.worker import (
    TaskType,
    WorkerIdentifer,
    WorkerResponse,
    WorkerTask,
    WorkerTaskId,
)


def handler(t: TaskType):
    """
    Takes a function f which is an instance method,
    and a TaskType t. Registers this function with the TaskType,
    such that when messages of type t are received by the object,
    function f gets called on t.
    """
    def wrapper(func: Callable[[ConcurrentClass, WorkerResponse], Iterable[WorkerTask]]):
        func._task_type = t
        return func
    return wrapper


def on_startup(func: Callable[[], Iterable[WorkerTask]]):
    """
    Registers a function to be called on object startup.
    """
    func._on_startup = True
    return func


def require_ready(func):
    """
    Requires that the object is ready before calling the function.
    """

    def wrapper(self, *args, **kwargs):
        if not self.ready():
            raise ValueError(f"{self} is not ready.")
        return func(self, *args, **kwargs)
    return wrapper


class ConcurrentClass(ABC):
    def __init__(self, worker_id: WorkerIdentifer):
        self.worker_id = worker_id
        self.handlers = {}
        for name in dir(self):
            attr = getattr(self, name)
            if hasattr(attr, '_task_type'):
                if attr._task_type in self.handlers:
                    raise ValueError(
                        f"TaskType {attr._task_type} already registered.")
                self.handlers[attr._task_type] = attr
        # self.startup_on_init = startup_on_init
        # if self.startup_on_init:
        #     for name in dir(self):
        #         attr = getattr(self, name)
        #         if hasattr(attr, '_on_startup'):
        #             attr()

        self._ready = False

    def ready(self) -> bool:
        """
        Returns whether the object is ready.
        """
        return self._ready

    def startup(self, callback: Optional[Callable[[], None]] = None) -> Iterable[WorkerTask]:
        """
        Use to delay the object startup.

        If callback is provided, it will be called when the object is ready.
        """
        if callback is not None:
            self.register_ready_callback(callback)
        for name in dir(self):
            attr = getattr(self, name)
            if hasattr(attr, '_on_startup'):
                for _ in attr():
                    yield _

    def register_ready_callback(self, callback: Callable[[], None]):
        """
        Register a callback to be called when the object is ready.
        """
        self.ready_callback = callback

    def finish(self):
        """
        Call when the object is ready.
        """
        self._ready = True
        if hasattr(self, 'ready_callback'):
            self.ready_callback()

    def tick(self, messages: Iterable[WorkerResponse]) -> Iterable[WorkerTask]:
        """
        Ticks the game state. All instance methods decorated with @handler
        will fire when the corresponding TaskType is received.
        """
        for message in messages:
            if message.task_id.task_type in self.handlers:
                for _ in self.handlers[message.task_id.task_type](message):
                    yield _
            else:
                raise ValueError(
                    f"TaskType {message.task_id.task_type} not registered.")
