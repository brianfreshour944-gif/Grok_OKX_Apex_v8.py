# api_utils.py — Rate-limit-aware API call wrapper with exponential backoff.
# Handles HTTP 429 (Too Many Requests) and transient failures gracefully.

import asyncio
import time

from config import logger


class RateLimitError(Exception):
    """Raised when API returns 429 and retry-after is not available."""
    pass


async def call_with_rate_limit_handling(api_call_func, max_retries: int = 5, base_delay: float = 1.0):
    """
    Wraps an API call function with exponential backoff and rate-limit handling.
    
    - Retries on HTTP 429 with Retry-After header or exponential backoff
    - Retries on 502/503/504 transient errors
    - Does NOT retry on 400/401/403 (client errors)
    - Uses exponential backoff: base_delay * (2 ** attempt)
    
    Args:
        api_call_func: A callable that makes the API call (no args)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff (seconds)
    
    Returns:
        The result of api_call_func()
    
    Raises:
        The original exception if all retries fail
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            return await asyncio.to_thread(api_call_func)
        except Exception as e:
            last_exception = e
            
            # Check if it's a rate limit error (HTTP 429)
            status_code = getattr(e, 'status_code', None)
            
            # Retry-After header parsing
            retry_after = None
            if hasattr(e, 'response') and e.response is not None:
                retry_after = e.response.headers.get('Retry-After')
            
            if status_code == 429:
                # Rate limited
                if retry_after:
                    try:
                        wait_time = float(retry_after)
                    except (ValueError, TypeError):
                        wait_time = base_delay * (2 ** attempt)
                else:
                    wait_time = base_delay * (2 ** attempt) + (0.5 * attempt)  # jitter
                
                logger.warning(
                    f"Rate limited (429). Waiting {wait_time:.1f}s before retry "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait_time)
                continue
            
            # Retry on transient server errors
            if status_code in (502, 503, 504):
                wait_time = base_delay * (2 ** attempt) + (0.5 * attempt)
                logger.warning(
                    f"Server error {status_code}. Retrying in {wait_time:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait_time)
                continue
            
            # Non-retryable error (4xx except 429)
            logger.error(f"Non-retryable API error: {e}")
            raise
    
    # All retries exhausted
    logger.error(f"All {max_retries} retry attempts exhausted for API call")
    raise last_exception


async def call_with_rate_limit_handling_async(api_call_func, *args, max_retries: int = 5, base_delay: float = 1.0, **kwargs):
    """
    Async version that wraps a function that takes arguments.
    
    Usage:
        await call_with_rate_limit_handling_async(
            trading_client.get_order_by_id, order_id,
            max_retries=3
        )
    """
    return await call_with_rate_limit_handling(
        lambda: api_call_func(*args, **kwargs),
        max_retries=max_retries,
        base_delay=base_delay
    )