"""Test emotion recognition integration with main app."""
import sys
import os
import tempfile
import numpy as np
import scipy.io.wavfile as wav

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly."""
    print("=" * 60)
    print("Testing Emotion Recognition Integration")
    print("=" * 60)

    print("\n1. Testing imports...")
    try:
        from modules.emotion import load_emotion_model, predict_emotion
        print("   ✓ emotion module imported")

        from main import analyze_emotion
        print("   ✓ analyze_emotion function imported from main")

        from main import build_prompt_context
        print("   ✓ build_prompt_context imported")

        return True
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_emotion_analysis():
    """Test emotion analysis function."""
    print("\n2. Testing emotion analysis...")

    try:
        from main import analyze_emotion

        # Create test audio
        sample_rate = 16000
        duration = 1.0
        audio_array = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        wav.write(tmp_path, sample_rate, audio_array)

        # Test analysis
        emotion_dict = analyze_emotion(tmp_path)

        # Verify result
        assert isinstance(emotion_dict, dict), "Result should be a dictionary"
        assert len(emotion_dict) == 7, "Should have 7 emotions"

        dominant = max(emotion_dict, key=emotion_dict.get)
        print(f"   ✓ Emotion analysis successful")
        print(f"   Dominant emotion: {dominant} ({emotion_dict[dominant]:.4f})")

        os.unlink(tmp_path)
        return True

    except Exception as e:
        print(f"   ✗ Emotion analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_building():
    """Test context building with emotion data."""
    print("\n3. Testing context building with emotion...")

    try:
        from main import build_prompt_context
        import pandas as pd

        # Mock data
        text = "Hello, how are you today?"
        personality_df = pd.DataFrame({
            "r": [0.5, 0.4, 0.6, 0.7, 0.5],
            "theta": ["EXT", "NEU", "AGR", "CON", "OPN"]
        })
        emotion_dict = {
            'happy': 0.6,
            'neutral': 0.2,
            'sad': 0.1,
            'angry': 0.05,
            'surprise': 0.03,
            'fear': 0.01,
            'disgust': 0.01
        }
        history = []
        preferences = {}

        # Build context
        messages = build_prompt_context(
            text, personality_df, emotion_dict, history, preferences, 5
        )

        # Verify
        assert len(messages) == 2, "Should have system and user messages"
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'

        system_content = messages[0]['content']
        assert 'EMOTIONAL STATE' in system_content, "Should contain emotion section"
        assert 'happy' in system_content.lower(), "Should mention dominant emotion"

        print("   ✓ Context building successful")
        print(f"   System prompt length: {len(system_content)} chars")
        print(f"   Contains emotion section: {'EMOTIONAL STATE' in system_content}")

        return True

    except Exception as e:
        print(f"   ✗ Context building failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test that emotion model loads at startup."""
    print("\n4. Testing model loading...")

    try:
        import modules.emotion as emotion_module

        # Load model
        emotion_module.load_emotion_model()

        # Verify loaded
        assert emotion_module._model is not None, "Model should be loaded"

        print("   ✓ Model loaded successfully")
        print(f"   Model parameters: {sum(p.numel() for p in emotion_module._model.parameters()):,}")

        return True

    except Exception as e:
        print(f"   ✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("\n")

    tests = [
        test_imports,
        test_model_loading,
        test_emotion_analysis,
        test_context_building,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    if all(results):
        print("✓ All integration tests passed!")
        print("=" * 60)
        print("\nEmotion recognition is successfully integrated!")
        print("The chatbot will now:")
        print("- Analyze emotion from audio at startup")
        print("- Display detected emotion before transcription")
        print("- Include emotion context in LLM prompts")
        print("- Enable more empathetic responses")
        return 0
    else:
        print(f"✗ {sum(not r for r in results)}/{len(results)} tests failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
