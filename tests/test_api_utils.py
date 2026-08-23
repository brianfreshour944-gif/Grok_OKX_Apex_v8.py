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
    Wraps a synchronous blocking call and runs a concurrent heartbeat
    alongside it, then compares the heartbeat gap against a deliberately-
    blocking control measured in the SAME run/environment -- rather than a
    fixed absolute-ms threshold, which is exactly the kind of test that
    flakes under CI/system scheduling jitter (this one did, once, on a
    loaded local machine: 454ms vs an 80ms threshold, on code that is
    correct). Comparing relative to a same-run control cancels out overall
    system speed/jitter, since it affects both measurements alike.
    """
    def blocking_call():
        time.sleep(0.2)
        return "ok"

    async def measure_max_heartbeat_gap(offload: bool) -> float:
        ticks = []
        stop = asyncio.Event()

        async def heartbeat():
            while not stop.is_set():
                ticks.append(time.perf_counter())
                await asyncio.sleep(0.01)

        hb = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.03)
        if offload:
            await call_with_rate_limit_handling_async(blocking_call, max_retries=1)
        else:
            blocking_call()  # deliberately blocks the loop -- the control
        await asyncio.sleep(0.03)
        stop.set()
        await hb

        gaps = [b - a for a, b in zip(ticks, ticks[1:])]
        return max(gaps) if gaps else 0.0

    real_gap    = run_async(measure_max_heartbeat_gap(offload=True))
    control_gap = run_async(measure_max_heartbeat_gap(offload=False))

    assert control_gap > 0.15, (
        f"control (deliberately blocking) only showed a {control_gap*1000:.0f}ms gap -- "
        f"test setup itself is broken, this should have blocked for ~200ms"
    )
    assert real_gap < control_gap / 2, (
        f"event loop was blocked for {real_gap*1000:.0f}ms vs a {control_gap*1000:.0f}ms "
        f"control -- call likely ran on the loop, not a thread"
    )


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
