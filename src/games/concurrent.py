import time
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

from src.workers.worker import *


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

        self._started = False
        self._ready = False

    def ready(self) -> bool:
        """
        Returns whether the object is ready.
        """
        return self._ready

    def started(self) -> bool:
        """
        Returns whether the object has started.
        """
        return self._started

    def startup(self, callback: Optional[Callable[[], None]] = None) -> Iterable[WorkerTask]:
        """
        Use to delay the object startup.

        If callback is provided, it will be called when the object is ready.
        """
        self._started = True
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
        Ticks the concurrent object. All instance methods decorated with @handler
        will fire when the corresponding TaskType is received.
        """
        for message in messages:
            if message.task_id.task_type in self.handlers:
                for _ in self.handlers[message.task_id.task_type](message):
                    yield _
            else:
                raise ValueError(
                    f"TaskType {message.task_id.task_type} not registered.")


T = TypeVar("T", bound=ConcurrentClass)


class Router(Generic[T]):
    def __init__(self, worker: Worker):
        self.worker = worker
        self.active = {}
        self.active_counts_by_task_type = {}
        self.total_times_by_task_type = {}
        self.starting_times = {}
        self.active_objs = {}
        self.total_active = 0

    def __str__(self):
        res = f"Worker {self.worker}, tasks:"
        for task_id, source in self.active.items():
            res += f"\n{task_id} from {source}"
        return res

    def startup(self, source: T):
        for msg in source.startup():
            self.enqueue_task(msg, source)

    def enqueue_task(self, msg: WorkerTask, source: T):
        self.worker.enqueue_task(msg)
        self.active.update(
            {
                msg.task_id: source
            }
        )
        self.active_objs[source] = msg
        task_type = msg.task_id.task_type
        self.total_active += 1
        self.active_counts_by_task_type[task_type] = self.active_counts_by_task_type.get(
            task_type, 0) + 1

        self.starting_times[msg.task_id] = time.time()

    def dequeue_tasks(self, blocking=False, timeout=None) -> Iterator[Tuple[WorkerResponse, T]]:
        responses: List[WorkerResponse] = self.worker.spin_deque_task(
            task_type=self.worker.worker_id,
            blocking=blocking,
            timeout=timeout,
            batch_size=None
        )

        for response in responses:
            if response.task_id not in self.active:
                raise ValueError(
                    f"Task {response.task_id} not in active tasks.")

            task_type = response.task_id.task_type
            self.total_times_by_task_type[task_type] = self.total_times_by_task_type.get(
                task_type, 0) + time.time() - self.starting_times[response.task_id]

            self.active_counts_by_task_type[task_type] -= 1
            self.total_active -= 1
            source = self.active.pop(response.task_id)
            self.active_objs.pop(source)
            yield response, source

    def contains(self, task_id: TaskIdentifier) -> bool:
        return task_id in self.active

    def tick(self, blocking=False, timeout=None):
        for response, source in self.dequeue_tasks(blocking=blocking, timeout=timeout):
            for msg in source.tick(response):
                self.enqueue_task(msg, source)

    def debug(self):
        print(f"Active tasks: {self.total_active}")
        for task_type, count in self.active_counts_by_task_type.items():
            print(f"TaskType {task_type}: {count} active")
            if task_type in self.total_times_by_task_type:
                if count > 0:
                    print(
                        f"Average time per task: {self.total_times_by_task_type[task_type] / count}")
        print("Active objects:")
        for obj, task in self.active_objs.items():
            print(f"{obj}: {task}")
        print("\n")
