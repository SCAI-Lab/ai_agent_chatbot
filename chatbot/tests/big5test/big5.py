from transformers import BertTokenizer, BertForSequenceClassification
import torch
import time

# Load tokenizer and model once at module level (one-time overhead)
print("Loading tokenizer and model (one-time setup)...")
start_load = time.time()
tokenizer = BertTokenizer.from_pretrained("Minej/bert-base-personality")
model = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality")
load_time = time.time() - start_load
print(f"Model loading completed in {load_time:.4f} seconds\n")

def personality_detection(text):
    """Perform personality detection on input text.

    Returns scores between 0 and 1 for each trait:
    - 0.0-0.3: Low
    - 0.3-0.7: Moderate
    - 0.7-1.0: High
    """

    # Tokenization
    start_time = time.time()
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    tokenization_time = time.time() - start_time

    # Inference
    start_time = time.time()
    outputs = model(**inputs)
    # Apply sigmoid to convert logits to probabilities (0-1 range)
    logits = outputs.logits.squeeze()
    predictions = torch.sigmoid(logits).numpy()
    inference_time = time.time() - start_time

    label_names = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    result = {label_names[i]: float(predictions[i]) for i in range(len(label_names))}

    print(f"Tokenization: {tokenization_time:.4f}s | Inference: {inference_time:.4f}s | Total: {tokenization_time + inference_time:.4f}s")

    return result

# Test with multiple calls to show the speed improvement
if __name__ == "__main__":
    test_texts = [
        "Hi I'm your boy Ethan! Hahaha",
        "I love spending time with friends and going to parties!",
        "I prefer staying home and reading books quietly."
    ]

    print("=" * 60)
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: \"{text}\"")
        start_time = time.time()
        result = personality_detection(text)
        total_time = time.time() - start_time
        print(f"Results: {result}")
        print(f"Call execution time: {total_time:.4f} seconds")
        print("-" * 60)