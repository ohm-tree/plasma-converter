import asyncio
import logging
import multiprocessing
import os
import queue
import random
import traceback
from abc import ABC, abstractmethod
from typing import Optional, Union

"""
All Tasks should be dict-like objects.


Event-based query system:
 - listen() is the main entrypoint for the event-based query system. This should run forever in the async loop.
 - query() is the high-level function that sends a query and awaits a response.

Other lower-level functions for synchronous querying:
 - spin_deque_tasks()
 - enqueue()
"""


class KillSignalException(Exception):
    pass


class Worker(ABC):
    def __init__(self,
                 name: str,
                 worker_type: str,
                 worker_idx: int,
                 queues: dict[str, multiprocessing.Queue],
                 run_name: str
                 ):
        self.name = name
        self.worker_type = worker_type
        self.worker_idx = worker_idx

        self.run_name = run_name
        self.queues = queues
        self.setup_logger()

        self.dequeue_events: dict[str, asyncio.Event] = {}
        self.dequeue_results: dict[str, dict] = {}

        self._task_idx = 0

        self.logger.info(
            f"Worker {self.worker_idx} of type {self.worker_type} initialized."
        )
        self.logger.info(
            f"Global Variables I can see: {globals().keys()}"
        )

        self._no_inbox = False
        if self.name not in self.queues:
            self._no_inbox = True
            self.logger.warning(
                "No inbox queue detected; worker may never terminate.")

        self._no_scream_queue = False
        if "kill" not in self.queues:
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
            os.path.join(self.ROOT_DIR, f"logs/{self.run_name}/{self.worker_type.worker_type}_worker_{self.worker_idx}.log"))

        logging_prefix = f'[{self.worker_type.worker_type} {self.worker_idx}] '
        formatter = logging.Formatter(
            logging_prefix + '%(asctime)s - %(levelname)s - %(message)s')

        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.info(
            f"Starting {self.worker_type} worker {self.worker_idx}."
        )

    @abstractmethod
    def validate(self, task: dict) -> None:
        """
        This method should be overridden to validate tasks
        that are dequeued.
        """
        pass

    def enqueue(self,
                obj: dict,
                channel: Optional[str] = None,
                ) -> None:
        """
        General-purpose enqueue function.

        If channel is None, it is inferred from the task
        by checking the "channel" field.
        """
        if channel is None:
            if "channel" not in obj:
                raise ValueError(
                    "No channel specified for enqueue.")
            channel = obj["channel"]
        self.queues[channel].put(obj)

    def enqueue_with_handler(self, obj: dict, channel: str) -> None:
        if "_task_idx" not in obj:
            obj["_task_idx"] = self.name + "_" + \
                str(self._task_idx) + "_" + str(random.randint(0, 1 << 30))
            self._task_idx += 1

        self.dequeue_events[obj["_task_idx"]] = asyncio.Event()
        self.enqueue(obj, channel)

    def spin_deque_tasks(self,
                         channel: str,
                         blocking=True,
                         timeout: Optional[int] = None,
                         batch_size: Optional[int] = None,
                         validate=True,
                         ) -> list[dict]:
        """
        If batch_size is None, return all tasks available.
        If batch_size is not None, return a batch of tasks
        of size at most batch_size.

        If timeout is None, block until a task is available.

        If timeout is not None, blocks for up to timeout seconds
        before returning; possibly returns fewer than batch_size tasks.

        If a kill signal is received, will raise a KillSignalException.

        Parameters
        ----------
        self : Worker
            The worker object.
        channel : str
            The channel to dequeue tasks from.
        blocking : bool
            Whether to block until a task is available.
        timeout : Optional[int]
            The number of seconds to wait before returning.
        batch_size : Optional[int]
            The number of tasks to return.
        validate : bool
            Whether to validate tasks before returning.

        """
        if batch_size is None:
            batch_size = float('inf')

        first = blocking
        num_tasks = 0
        res = []
        while num_tasks < batch_size:
            try:
                if first:
                    first = False
                    task = self.queues[channel].get(
                        block=True,
                        timeout=timeout)
                else:
                    task = self.queues[channel].get_nowait()
            except queue.Empty:
                break
            else:
                if task == 'kill':
                    self.logger.fatal(
                        f"Received kill signal, terminating.")
                    raise KillSignalException()
                if validate:
                    self.validate(task)
                res.append(task)
                num_tasks += 1
        return res

    def spin_deque_tasks_with_handler(self,
                                      channel: str,
                                      blocking=True,
                                      timeout: Optional[int] = None,
                                      batch_size: Optional[int] = None,
                                      validate=True,
                                      ) -> list[dict]:
        """
        If batch_size is None, return all tasks available.
        If batch_size is not None, return a batch of tasks
        of size at most batch_size.

        If timeout is None, block until a task is available.

        If timeout is not None, blocks for up to timeout seconds
        before returning; possibly returns fewer than batch_size tasks.

        If a kill signal is received, will raise a KillSignalException.

        Parameters
        ----------
        channel : str
            The channel to dequeue tasks from.
        blocking : bool
            Whether to block until a task is available.
        timeout : Optional[int]
            The number of seconds to wait before returning.
        batch_size : Optional[int]
            The number of tasks to return.
        validate : bool
            Whether to validate tasks before returning.
        """
        res = self.spin_deque_tasks(
            channel, blocking, timeout, batch_size, validate)
        for task in res:
            if "_task_idx" not in task:
                raise ValueError(
                    "Task does not have a _task_idx field.")
            if task["_task_idx"] not in self.dequeue_events:
                raise ValueError(
                    "Task does not have a corresponding dequeue event.")
            self.dequeue_results[task["_task_idx"]] = task
            self.dequeue_events[task["_task_idx"]].set()
        return res

    async def listen(self,
                     channel: str,
                     blocking=False,
                     timeout: Optional[int] = None,
                     batch_size: Optional[int] = None,
                     validate=True,
                     ) -> list[dict]:
        """
        This method is an async wrapper around spin_deque_tasks_with_handler;
        it is the main entrypoint for the event-based query system.
        """
        while True:
            self.spin_deque_tasks_with_handler(
                channel, blocking, timeout, batch_size, validate)

            await asyncio.sleep(0)

    def deque_task(self,
                   channel: str,
                   timeout: Optional[int] = None,
                   ) -> Optional[dict]:
        try:
            return self.queues[channel].get(timeout=timeout)
        except queue.Empty:
            return None

    async def query(self, task: dict, channel: Optional[str] = None) -> dict:
        """
        Send a query to the master process and wait for a response.
        """
        if channel is None:
            if "channel" not in task:
                raise ValueError(
                    "No channel specified for query.")
            channel = task["channel"]
        self.enqueue_with_handler(task, channel)
        await self.dequeue_events[task["_task_idx"]].wait()
        res = self.dequeue_results[task["_task_idx"]]
        del self.dequeue_results[task["_task_idx"]]
        del self.dequeue_events[task["_task_idx"]]
        return res

    def run(self):
        """
        The default main loop for a worker which
        loops infinitely until a kill signal is received.

        If you want to implement a custom main loop, override this method,
        not the main method.
        """
        # check for kill signals from the master queue.
        while True:
            try:
                self.loop()
            except Exception as e:
                self.logger.critical(
                    "An exception occurred in the worker loop!!")
                self.logger.critical(traceback.format_exc())
                break
        else:
            self.logger.info(
                f"Worker {self.worker_idx} of type {self.worker_type} received kill signal."
            )

    def main(self):
        """
        The main method for a worker.

        This method should not be overridden.
        """
        try:
            self.run()
        except KillSignalException:
            self.logger.info(
                f"Worker {self.worker_idx} of type {self.worker_type} received kill signal.")
        except Exception as e:
            self.logger.critical(
                "An exception occurred in the worker main!!")
            self.logger.critical(traceback.format_exc())
        finally:
            try:
                self.shutdown()
            except Exception as e:
                self.logger.critical(
                    "An exception occurred in the worker shutdown!!")
                self.logger.critical(traceback.format_exc())

            self.logger.info(
                f"Worker {self.worker_idx} of type {self.worker_type} terminated."
            )

    @abstractmethod
    def loop(self):
        pass

    def shutdown(self):
        pass
