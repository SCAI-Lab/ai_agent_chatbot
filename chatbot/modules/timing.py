"""Simple performance timing utilities."""
import time
import functools
from typing import Callable, Dict, Any


# Simple dict to store current session timings
_timings: Dict[str, float] = {}


def clear_timings():
    """Clear all timing data."""
    _timings.clear()


def get_timings() -> Dict[str, float]:
    """Get current timing data."""
    return _timings.copy()


def print_timings(title: str = "Performance Summary"):
    """Print timing summary."""
    if not _timings:
        return

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"{'Operation':<30} {'Duration':>10}")
    print("-" * 60)

    total = 0.0
    for name in sorted(_timings.keys()):
        duration = _timings[name]
        print(f"{name:<30} {duration:>9.4f}s")
        total += duration

    print("-" * 60)
    print(f"{'TOTAL':<30} {total:>9.4f}s")
    print("=" * 60)


def timing(name: str = None):
    """Simple decorator to time function execution."""
    def decorator(func: Callable) -> Callable:
        op_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            _timings[op_name] = time.perf_counter() - start
            return result

        return wrapper
    return decorator


class timing_context:
    """Simple context manager for timing code blocks."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _timings[self.name] = time.perf_counter() - self.start
