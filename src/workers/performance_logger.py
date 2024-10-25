import logging
import time


class PerformanceLogger():
    def __init__(self, min_log_interval=60):
        self.start_time = time.time()
        self.query_count = 0
        self.latencies = []
        self.num_queries_per_minute = []

        self.last_log_time = -float("inf")
        self.min_log_interval = min_log_interval

    def log_query(self, latency, quantity):
        self.query_count += quantity
        self.latencies.extend([latency] * quantity)

        # check the current minute.
        minute = int(time.time() - self.start_time) // 60

        while minute >= len(self.num_queries_per_minute):
            self.num_queries_per_minute.append(0)

        self.num_queries_per_minute[minute] += quantity

    def calculate_diagnostics(self):
        throughput_stats = self.calculate_throughput_stats()
        latency_stats = self.calculate_latency_stats()

        return {
            **throughput_stats,
            **latency_stats
        }

    def occasional_log(self, logger: logging.Logger):
        if time.time() - self.last_log_time < self.min_log_interval:
            return
        self.last_log_time = time.time()
        diagnostics = self.calculate_diagnostics()
        for key, value in diagnostics.items():
            logger.info(f"{key}: {value}")  # Log the diagnostics

    def calculate_throughput_stats(self):
        elapsed_time = time.time() - self.start_time
        elapsed_minutes = elapsed_time / 60
        throughput = self.query_count / elapsed_minutes if elapsed_minutes > 0 else 0

        throughput_volatility = (sum((x - throughput) ** 2 for x in self.num_queries_per_minute) / len(
            self.num_queries_per_minute)) ** 0.5 if self.num_queries_per_minute else 0

        return {
            'absolute count': self.query_count,
            'absolute time (min)': elapsed_time,
            'throughput': throughput,
            'throughput volatility': throughput_volatility
        }

    def calculate_latency_stats(self):
        if not self.latencies:
            return "No latencies recorded."

        mean_latency = sum(self.latencies) / len(self.latencies)
        stdev_latency = (sum((x - mean_latency) **
                         2 for x in self.latencies) / len(self.latencies)) ** 0.5
        sorted_latencies = sorted(self.latencies)
        min_latency = sorted_latencies[0]
        max_latency = sorted_latencies[-1]
        bottom_10 = sorted_latencies[int(len(sorted_latencies) * 0.1)]
        bottom_1 = sorted_latencies[int(len(sorted_latencies) * 0.01)]
        top_10 = sorted_latencies[int(len(sorted_latencies) * 0.9)]
        top_1 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        return {
            'latency mean': mean_latency,
            'latency stdev': stdev_latency,
            'latency top_1%': top_1,
            'latency top_10%': top_10,
            'latency bottom_10%': bottom_10,
            'latency bottom_1%': bottom_1,
            'latency min': min_latency,
            'latency max': max_latency
        }

    def reset(self):
        self.start_time = time.time()
        self.query_count = 0
        self.latencies = []
        self.num_queries_per_minute = []
