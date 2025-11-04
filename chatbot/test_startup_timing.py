"""Test startup timing with emotion model."""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_startup_timing():
    """Test model loading times at startup."""
    print("=" * 60)
    print("Startup Timing Test")
    print("=" * 60)

    timings = {}

    # Test emotion model loading
    print("\n1. Loading emotion model...")
    start = time.perf_counter()
    from modules.emotion import load_emotion_model
    load_emotion_model()
    timings['Emotion model'] = time.perf_counter() - start
    print(f"   Time: {timings['Emotion model']:.3f}s")

    # Test personality model loading
    print("\n2. Loading personality model...")
    start = time.perf_counter()
    from modules.personality import load_personality_model
    load_personality_model()
    timings['Personality model'] = time.perf_counter() - start
    print(f"   Time: {timings['Personality model']:.3f}s")

    # Test Whisper model loading
    print("\n3. Loading Whisper model...")
    import torch
    start = time.perf_counter()
    from modules.speech import load_whisper_pipeline
    whisper_pipeline = load_whisper_pipeline(torch.cuda.is_available())
    timings['Whisper model'] = time.perf_counter() - start
    print(f"   Time: {timings['Whisper model']:.3f}s")

    # Summary
    total = sum(timings.values())
    print("\n" + "=" * 60)
    print("Startup Summary")
    print("=" * 60)
    for name, duration in timings.items():
        print(f"{name:<30} {duration:>8.3f}s ({duration/total*100:>5.1f}%)")
    print("-" * 60)
    print(f"{'Total startup time':<30} {total:>8.3f}s")
    print("=" * 60)

    # Check if acceptable
    print("\nPerformance Assessment:")
    if timings['Emotion model'] < 5.0:
        print("✓ Emotion model load time is acceptable (<5s)")
    else:
        print("⚠ Emotion model load time is high (>5s)")

    if total < 30.0:
        print("✓ Total startup time is acceptable (<30s)")
    else:
        print("⚠ Total startup time is high (>30s)")

if __name__ == "__main__":
    test_startup_timing()
