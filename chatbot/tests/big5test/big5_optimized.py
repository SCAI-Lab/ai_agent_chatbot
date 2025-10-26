from transformers import BertTokenizer, BertForSequenceClassification
import torch
import time

print("=" * 60)
print("Big5 Personality Detection - Optimized Version")
print("=" * 60)

# Load tokenizer and model once (one-time setup)
print("\n[1] Loading model...")
start_load = time.time()
tokenizer = BertTokenizer.from_pretrained("Minej/bert-base-personality")
model = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality")

# Optimizations
model.eval()  # Set to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

load_time = time.time() - start_load
print(f"Device: {device}")
print(f"Model loading time: {load_time:.2f}s")

def personality_detection(text):
    """Analyze personality from text.

    Returns scores between 0 and 1 for each trait:
    - 0.0-0.3: Low
    - 0.3-0.7: Moderate
    - 0.7-1.0: High
    """
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Apply sigmoid to convert logits to probabilities (0-1 range)
    logits = outputs.logits.squeeze()
    predictions = torch.sigmoid(logits).cpu().numpy()

    label_names = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    return {label_names[i]: float(predictions[i]) for i in range(len(label_names))}

if __name__ == "__main__":
    # Warmup GPU
    print("\n[2] Warming up GPU...")
    warmup_text = "This is a warmup"
    if torch.cuda.is_available():
        warmup_start = time.time()
        _ = personality_detection(warmup_text)
        warmup_time = time.time() - warmup_start
        print(f"Warmup time: {warmup_time:.4f}s")

    # Test inference
    print("\n[3] Testing inference speed...")
    text = "Hi I'm your boy Ethan! Hahaha"
    print(f"Text: \"{text}\"")

    num_runs = 5
    times = []
    for i in range(num_runs):
        start = time.time()
        result = personality_detection(text)
        elapsed = time.time() - start
        times.append(elapsed)
        if i == 0:
            print(f"Run 1: {elapsed:.4f}s")

    avg_time = sum(times) / len(times)
    print(f"Average ({num_runs} runs): {avg_time:.4f}s")
    print(f"Range: {min(times):.4f}s - {max(times):.4f}s")

    # Show results
    print("\n[4] Results:")
    for trait, score in result.items():
        level = "Low" if score < 0.3 else ("High" if score > 0.7 else "Moderate")
        print(f"  {trait:20s}: {score:.3f} ({level})")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model loading: {load_time:.2f}s (one-time)")
    if torch.cuda.is_available():
        print(f"GPU warmup:    {warmup_time:.4f}s (one-time)")
    print(f"Inference:     {avg_time:.4f}s (per text)")
    print("=" * 60)
