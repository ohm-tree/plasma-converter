import abc
import logging
import multiprocessing
import os
import queue
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass(frozen=True)
class TaskType:
    """
    A type of task.
    """
    task_type: str


@dataclass(frozen=True)
class TaskIdentifier:
    """
    A unique identifier for a task.
    """
    task_idx: int
    task_type: TaskType


KillTaskType = TaskType("kill")


@dataclass(frozen=True)
class WorkerType:
    """
    A type of worker.
    """
    worker_type: str
    consumes: Iterable[TaskType]


@dataclass(frozen=True)
class WorkerIdentifer:
    """
    A unique identifier for a worker.
    """
    worker_type: WorkerType
    worker_idx: int


@dataclass(frozen=True)
class WorkerTask:
    """
    A task for a worker to complete.
    """
    head_id: WorkerIdentifer
    task_id: TaskIdentifier
    task: Any


@dataclass(frozen=True)
class WorkerResponse:
    """
    A response from a worker to a task.
    """
    head_id: WorkerIdentifer
    tail_id: WorkerIdentifer
    task_id: TaskIdentifier
    task: Any
    response: Any


class Worker(abc.ABC):
    def __init__(self,
                 worker_id: WorkerIdentifer,
                 queues: Dict[Union[TaskType, WorkerIdentifer], multiprocessing.Queue],
                 run_name: str):
        self.worker_type = worker_id.worker_type
        self.worker_idx = worker_id.worker_idx
        self.worker_id = worker_id

        self.run_name = run_name

        self.queues = queues

        self.setup_logger()

        self._task_idx = 0

        self.logger.info(
            f"Worker {self.worker_idx} of type {self.worker_type} initialized."
        )

        self._no_inbox = False
        if self.worker_type not in self.queues:
            self._no_inbox = True
            self.logger.warning(
                "No inbox queue detected; worker may never terminate.")

        self._no_scream_queue = False
        if KillTaskType not in self.queues:
            self._no_scream_queue = True
            self.logger.warning(
                "No scream queue detected; when worker terminates, master may not know.")

    def setup_logger(self):
        # I should live in src/workers/
        WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
        SRC_DIR = os.path.dirname(WORKER_DIR)
        self.ROOT_DIR = os.path.dirname(SRC_DIR)

        # give myself a custom logging file.
        os.makedirs(os.path.join(self.ROOT_DIR,
                    "logs", self.run_name), exist_ok=True)

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(
            os.path.join(self.ROOT_DIR, f"logs/{self.run_name}/{self.worker_type}_worker_{self.worker_idx}.log"))

        logging_prefix = f'[{self.worker_type} - {self.worker_idx}] '
        formatter = logging.Formatter(
            logging_prefix + '%(asctime)s - %(levelname)s - %(message)s')

        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.info(
            f"Starting {self.worker_type} worker {self.worker_idx}."
        )

    def enqueue(self,
                obj: Any,
                where: Union[TaskType, WorkerIdentifer]
                ) -> None:
        """
        General-purpose enqueue function.
        """
        self.queues[where].put(obj)

    def enqueue_task(self,
                     obj: Any,
                     task_idx: int,
                     task_type: TaskType
                     ) -> None:
        task = WorkerTask(
            head_id=self.worker_id,
            task_id=TaskIdentifier(
                task_idx=task_idx, task_type=task_type),
            task=obj
        )
        self.enqueue(task, task_type)

    def enqueue_response(self,
                         response: Any,
                         task: Union[WorkerTask, WorkerResponse],
                         ) -> None:

        response_task = WorkerResponse(
            head_id=task.head_id,
            tail_id=self.worker_id,
            task_id=task.task_id,
            task=task.task,
            response=response
        )
        self.enqueue(response_task, task.head_id)

    def spin_deque_task(self,
                        task_type: Union[TaskType, WorkerIdentifer],
                        blocking=True,
                        timeout: Optional[int] = None,
                        batch_size: Optional[int] = None,
                        ) -> List[Union[WorkerTask, WorkerResponse]]:
        """
        If batch_size is None, return all tasks available.
        If batch_size is not None, return a batch of tasks
        of size at most batch_size.

        If timeout is None, block until a task is available.

        If timeout is not None, blocks for up to timeout seconds
        before returning; possibly returns fewer than batch_size tasks.
        """
        if batch_size is None:
            batch_size = float('inf')

        my_tasks: List[Union[WorkerTask, WorkerResponse]] = []
        first = blocking
        task: Union[WorkerTask, WorkerResponse]
        while len(my_tasks) < batch_size:
            try:
                if first:
                    first = False
                    task = self.queues[task_type].get(
                        timeout=timeout)
                else:
                    task = self.queues[task_type].get_nowait()
            except queue.Empty:
                break
            else:
                if task == 'kill':
                    self.logger.info(
                        f"Received kill signal, terminating.")
                    return my_tasks
                if task.task_id.task_type not in self.worker_type.consumes:
                    raise ValueError(
                        f"I am a worker of type {self.worker_type}, got {task.task_id.task_type}"
                    )
            my_tasks.append(task)

        return my_tasks

    def deque_task(self,
                   task_type: TaskType,
                   timeout: Optional[int] = None,
                   ) -> Optional[Union[WorkerTask, WorkerResponse]]:
        try:
            return self.queues[task_type].get(timeout=timeout)
        except queue.Empty:
            return None

    def run(self):
        """
        The default main loop for a worker which
        loops infinitely until a kill signal is received.

        If you want to implement a custom main loop, override this method,
        not the main method.
        """
        # check for kill signals from the master queue.
        while self.no_poison():
            try:
                self.loop()
            except Exception as e:
                self.logger.critical(
                    "An exception occurred in the worker loop!!")
                self.logger.critical(e)
                break

    def main(self):
        """
        The main method for a worker.

        This method should not be overridden.
        """
        self.run()
        try:
            # enqueue a kill signal to the master queue.
            if not self._no_scream_queue:
                self.queues[KillTaskType].put("kill")

            self.shutdown()
            self.logger.info(
                f"Worker {self.worker_idx} of type {self.worker_type} terminated."
            )

        except Exception as e:
            self.logger.critical(
                "An exception occurred in the worker shutdown!!")
            self.logger.critical(e)

    def no_poison(self) -> bool:
        """
        Check for kill signals from the master queue.

        Returns
        -------
        bool
            True if no kill signal has been received, False otherwise.
        """
        if not self._no_inbox:
            try:
                kill_signal = self.queues[self.worker_id].get_nowait()
                print(
                    f"Received kill signal: {kill_signal}")
            except queue.Empty:
                pass
            else:
                return False
        return True

    @abc.abstractmethod
    def loop(self):
        pass

    def shutdown(self):
        pass
