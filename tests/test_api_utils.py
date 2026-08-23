# tests/test_api_utils.py — rate-limit/backoff wrapper, including a
# permanent regression test for the event-loop-blocking bug: this wrapper
# used to call api_call_func() directly instead of via asyncio.to_thread,
# which stalled the ENTIRE bot (heartbeat, all other symbols' fetches, the
# drawdown check) for the full duration of every single order submission.

import asyncio
import time

import pytest

from api_utils import call_with_rate_limit_handling, call_with_rate_limit_handling_async, RateLimitError
from conftest import run_async


class FakeApiError(Exception):
    def __init__(self, status_code, response=None):
        super().__init__(f"status={status_code}")
        self.status_code = status_code
        self.response = response


def test_does_not_block_the_event_loop():
    """
    Wraps a synchronous 150ms blocking call and runs a concurrent 10ms
    heartbeat alongside it. If the wrapper doesn't actually offload to a
    thread, the heartbeat will show one large gap (~150ms) instead of many
    small ones (~10ms).
    """
    def blocking_call():
        time.sleep(0.15)
        return "ok"

    async def scenario():
        ticks = []
        stop = asyncio.Event()

        async def heartbeat():
            while not stop.is_set():
                ticks.append(time.perf_counter())
                await asyncio.sleep(0.01)

        hb = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.03)
        result = await call_with_rate_limit_handling_async(blocking_call, max_retries=1)
        await asyncio.sleep(0.03)
        stop.set()
        await hb

        gaps = [b - a for a, b in zip(ticks, ticks[1:])]
        return result, (max(gaps) if gaps else 0.0)

    result, max_gap = run_async(scenario())
    assert result == "ok"
    assert max_gap < 0.08, f"event loop was blocked for {max_gap*1000:.0f}ms -- call ran on the loop, not a thread"


def test_retries_on_429_then_succeeds():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise FakeApiError(429)
        return "ok"

    async def scenario():
        return await call_with_rate_limit_handling(flaky, max_retries=5, base_delay=0.001)

    result = run_async(scenario())
    assert result == "ok"
    assert attempts["n"] == 3


def test_does_not_retry_on_non_retryable_4xx():
    attempts = {"n": 0}

    def always_403():
        attempts["n"] += 1
        raise FakeApiError(403)

    async def scenario():
        return await call_with_rate_limit_handling(always_403, max_retries=5, base_delay=0.001)

    with pytest.raises(FakeApiError):
        run_async(scenario())
    assert attempts["n"] == 1  # no retries wasted on a client error


def test_raises_after_exhausting_retries():
    def always_429():
        raise FakeApiError(429)

    async def scenario():
        return await call_with_rate_limit_handling(always_429, max_retries=3, base_delay=0.001)

    with pytest.raises(FakeApiError):
        run_async(scenario())


def test_async_wrapper_passes_args_and_kwargs_through():
    def add(a, b, c=0):
        return a + b + c

    async def scenario():
        return await call_with_rate_limit_handling_async(add, 1, 2, c=3, max_retries=1)

    assert run_async(scenario()) == 6
