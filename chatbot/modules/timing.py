"""Simple performance timing utilities."""
import time
import functools
from typing import Callable, Dict, Any, List, Tuple


# Store timings with order: [(name, duration, order_index), ...]
_timings: List[Tuple[str, float, int]] = []
_timing_order = 0


def clear_timings():
    """Clear all timing data."""
    global _timings, _timing_order
    _timings = []
    _timing_order = 0


def get_timings() -> Dict[str, float]:
    """Get current timing data as a dict (for backwards compatibility)."""
    return {name: duration for name, duration, _ in _timings}


def _record_timing(name: str, duration: float):
    """Record a timing with order tracking."""
    global _timing_order
    _timings.append((name, duration, _timing_order))
    _timing_order += 1


def print_timings(title: str = "Performance Summary"):
    """Print timing summary in execution order."""
    if not _timings:
        return

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"{'Operation':<40} {'Duration':>10}")
    print("-" * 60)

    # Sort by order index to show execution order
    sorted_timings = sorted(_timings, key=lambda x: x[2])

    total = 0.0
    for name, duration, _ in sorted_timings:
        # Add indentation for sub-operations
        indent = "  " if name.startswith("  ") or "_sub_" in name else ""
        display_name = name.replace("_sub_", "").strip()
        print(f"{indent}{display_name:<38} {duration:>9.4f}s")
        total += duration

    print("-" * 60)
    print(f"{'TOTAL':<40} {total:>9.4f}s")
    print("=" * 60)


def timing(name: str = None):
    """Simple decorator to time function execution."""
    def decorator(func: Callable) -> Callable:
        op_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            _record_timing(op_name, duration)
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
        duration = time.perf_counter() - self.start
        _record_timing(self.name, duration)
