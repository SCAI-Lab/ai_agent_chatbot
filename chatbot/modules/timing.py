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

    # Build list of sub-tasks to exclude from total
    # These are tasks included in parent operations
    excluded_from_total = set()

    # LLM sub-tasks (llm_total includes these)
    llm_subtasks = {"llm_memory_fetch", "llm_first_token", "llm_inference"}
    excluded_from_total.update(llm_subtasks)

    # Exclude response TTS (output phase, not processing latency)
    excluded_from_total.add("response_tts")

    # Find parallel analysis block for runtime timings
    for name, duration, _ in sorted_timings:
        if name.startswith("[Parallel] Analysis"):
            # Mark individual analysis tasks as subtasks
            excluded_from_total.update(["personality_analysis", "text2emotion_analysis", "speech2emotion_analysis"])
            break

    total = 0.0
    for name, duration, _ in sorted_timings:
        # Add indentation for sub-operations
        is_subtask = (
            name.startswith("  ├─") or
            "_sub_" in name or
            name in llm_subtasks or
            name in excluded_from_total
        )

        indent = "  " if is_subtask else ""
        display_name = name.replace("_sub_", "").strip()

        # Add visual indicator for sub-tasks
        if is_subtask and not display_name.startswith("├─"):
            display_name = f"├─ {display_name}"

        print(f"{indent}{display_name:<38} {duration:>9.4f}s")

        # Only add to total if not a sub-task
        if name not in excluded_from_total and not is_subtask:
            total += duration

    print("-" * 60)
    print(f"{'TOTAL (wall clock time)':<40} {total:>9.4f}s")
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
