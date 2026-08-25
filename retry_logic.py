"""
Retry logic with exponential backoff for resilient operation retries.

Provides a decorator for automatically retrying operations with
configurable delays and exponential backoff.
"""

import time
import logging
from typing import Callable, TypeVar, Tuple, Type
from functools import wraps

from constants import RetryConstants
from exceptions import HikrobotError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryableOperationError(HikrobotError):
    """Raised when an operation fails after all retry attempts exhausted."""
    pass


def retry_with_backoff(
    max_retries: int = RetryConstants.DEFAULT_MAX_RETRIES,
    initial_delay: float = RetryConstants.DEFAULT_INITIAL_DELAY,
    backoff_factor: float = RetryConstants.DEFAULT_BACKOFF_FACTOR,
    max_delay: float = RetryConstants.DEFAULT_MAX_DELAY,
    exceptions: Tuple[Type[Exception], ...] = RetryConstants.RETRYABLE_EXCEPTIONS
) -> Callable:
    """Decorator for retrying operations with exponential backoff.
    
    Automatically retries a function call if it raises specified exceptions.
    Delays between retries using exponential backoff with a maximum cap.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 0.1)
        backoff_factor: Multiplier for delay after each retry (default: 2.0)
        max_delay: Maximum delay between retries in seconds (default: 30.0)
        exceptions: Tuple of exception types to catch and retry on
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry_with_backoff(max_retries=3, initial_delay=0.5)
        def connect_to_plc():
            # Connection logic here
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    logger.debug(f"Attempt {attempt + 1}/{max_retries} for {func.__name__}")
                    result = func(*args, **kwargs)
                    
                    if attempt > 0:
                        logger.info(f"{func.__name__} succeeded on attempt {attempt + 1}")
                    
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries - 1:
                        # Last attempt failed
                        logger.error(
                            f"{func.__name__} failed after {max_retries} attempts: {e}"
                        )
                        raise RetryableOperationError(
                            f"{func.__name__} failed after {max_retries} attempts"
                        ) from e
                    
                    # Not the last attempt - wait and retry
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}, "
                        f"retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
                
                except Exception as e:
                    # Non-retryable exception
                    logger.error(f"{func.__name__} raised non-retryable exception: {e}")
                    raise
            
            # Should not reach here, but just in case
            if last_exception:
                raise RetryableOperationError(
                    f"{func.__name__} failed after {max_retries} attempts"
                ) from last_exception
        
        return wrapper
    return decorator


# ============================================================================
# Example usage patterns
# ============================================================================

if __name__ == "__main__":
    # Example 1: Simple retry with defaults
    @retry_with_backoff()
    def unreliable_operation():
        """Operation that might fail."""
        import random
        if random.random() < 0.7:  # 70% failure rate
            raise TimeoutError("Connection timeout")
        return "Success!"
    
    # Example 2: Custom retry parameters
    @retry_with_backoff(
        max_retries=5,
        initial_delay=0.2,
        backoff_factor=1.5,
        max_delay=10.0
    )
    def network_operation():
        """Network operation with custom retry config."""
        # Network logic here
        pass
    
    # Example 3: Retry on specific exceptions only
    @retry_with_backoff(
        max_retries=3,
        exceptions=(ConnectionError, TimeoutError)
    )
    def plc_communication():
        """PLC communication with specific exception handling."""
        # PLC logic here
        pass
