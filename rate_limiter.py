"""
Rate limiter for controlling API request throughput
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional
from collections import deque
import threading


class RateLimiter:
    """
    Rate limiter that controls the number of requests per time window
    with support for batch processing with delays
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        """
        Initialize rate limiter

        Args:
            max_requests: Maximum number of requests allowed per window
            window_seconds: Time window in seconds (default: 60 for per-minute limiting)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_times: deque = deque()
        self.lock = threading.Lock()

    def _clean_old_requests(self) -> None:
        """Remove requests outside the current time window"""
        now = datetime.now()
        cutoff_time = now - timedelta(seconds=self.window_seconds)

        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()

    def get_current_usage(self) -> dict:
        """Get current rate limit usage information"""
        with self.lock:
            self._clean_old_requests()

            if self.request_times:
                oldest_request = self.request_times[0]
                window_reset = oldest_request + timedelta(seconds=self.window_seconds)
            else:
                window_reset = datetime.now() + timedelta(seconds=self.window_seconds)

            return {
                "requests_in_current_window": len(self.request_times),
                "max_requests_per_window": self.max_requests,
                "window_reset_at": window_reset,
                "requests_remaining": max(0, self.max_requests - len(self.request_times))
            }

    def can_make_request(self) -> bool:
        """Check if a request can be made without exceeding the rate limit"""
        with self.lock:
            self._clean_old_requests()
            return len(self.request_times) < self.max_requests

    def record_request(self) -> None:
        """Record a request in the rate limiter"""
        with self.lock:
            self._clean_old_requests()
            self.request_times.append(datetime.now())

    def wait_time_until_available(self) -> float:
        """Calculate seconds to wait until a request can be made"""
        with self.lock:
            self._clean_old_requests()

            if len(self.request_times) < self.max_requests:
                return 0.0

            # Calculate when the oldest request will expire
            oldest_request = self.request_times[0]
            expiry_time = oldest_request + timedelta(seconds=self.window_seconds)
            wait_time = (expiry_time - datetime.now()).total_seconds()

            return max(0.0, wait_time)

    async def acquire(self, count: int = 1) -> None:
        """
        Acquire permission to make request(s), waiting if necessary

        Args:
            count: Number of requests to acquire (for batch processing)
        """
        if count > self.max_requests:
            raise ValueError(f"Cannot acquire {count} requests, max is {self.max_requests}")

        # For batch requests, wait until we have enough capacity
        while True:
            with self.lock:
                self._clean_old_requests()
                available = self.max_requests - len(self.request_times)

                if available >= count:
                    # Record all requests
                    now = datetime.now()
                    for _ in range(count):
                        self.request_times.append(now)
                    return

            # Wait a bit before checking again
            wait_time = self.wait_time_until_available()
            if wait_time > 0:
                await asyncio.sleep(wait_time + 0.1)  # Add small buffer
            else:
                await asyncio.sleep(0.1)  # Small polling interval


class BatchRateLimiter:
    """
    Specialized rate limiter for batch processing that processes items
    in chunks with delays between batches
    """

    def __init__(self, max_per_batch: int = 10, batch_delay_seconds: int = 60):
        """
        Initialize batch rate limiter

        Args:
            max_per_batch: Maximum items to process per batch
            batch_delay_seconds: Delay between batches in seconds
        """
        self.max_per_batch = max_per_batch
        self.batch_delay_seconds = batch_delay_seconds
        self.last_batch_time: Optional[datetime] = None
        self.lock = asyncio.Lock()

    async def process_batches(self, items: List, process_func) -> List:
        """
        Process items in rate-limited batches

        Args:
            items: List of items to process
            process_func: Async function to process a single batch

        Returns:
            List of results from all batches
        """
        all_results = []
        total_items = len(items)

        for batch_start in range(0, total_items, self.max_per_batch):
            batch_end = min(batch_start + self.max_per_batch, total_items)
            batch = items[batch_start:batch_end]

            # Wait for delay if this is not the first batch
            async with self.lock:
                if self.last_batch_time is not None:
                    elapsed = (datetime.now() - self.last_batch_time).total_seconds()
                    if elapsed < self.batch_delay_seconds:
                        wait_time = self.batch_delay_seconds - elapsed
                        await asyncio.sleep(wait_time)

                # Process the batch
                batch_results = await process_func(batch)
                all_results.extend(batch_results)

                # Update last batch time
                self.last_batch_time = datetime.now()

        return all_results

    def get_batch_info(self, total_items: int) -> dict:
        """Get information about how items will be batched"""
        num_batches = (total_items + self.max_per_batch - 1) // self.max_per_batch
        estimated_time = (num_batches - 1) * self.batch_delay_seconds if num_batches > 1 else 0

        return {
            "total_items": total_items,
            "items_per_batch": self.max_per_batch,
            "number_of_batches": num_batches,
            "delay_between_batches_seconds": self.batch_delay_seconds,
            "estimated_total_time_seconds": estimated_time
        }
