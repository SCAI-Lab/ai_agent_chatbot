"""
Simple comparison: normal vs optimized
Both use IDENTICAL implementation, only difference is optimization flags
"""
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import time

test_text = "Hi I'm your boy Ethan! Hahaha"
num_runs = 10

print("=" * 60)
print("Big5 Performance Comparison")
print("=" * 60)
print(f"Test: \"{test_text}\"")
print(f"Runs: {num_runs}\n")

# ============================================================
# [1] Normal version (no optimizations)
# ============================================================
print("[1] Normal (baseline)")
print("-" * 60)

tokenizer1 = BertTokenizer.from_pretrained("Minej/bert-base-personality")
model1 = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality")

times1 = []
for i in range(num_runs):
    start = time.time()
    inputs = tokenizer1(test_text, truncation=True, padding=True, return_tensors="pt")
    outputs = model1(**inputs)
    # Apply sigmoid (same as optimized version)
    logits = outputs.logits.squeeze()
    predictions = torch.sigmoid(logits).detach().numpy()
    times1.append(time.time() - start)

avg1 = sum(times1) / len(times1)
print(f"Average: {avg1:.4f}s")
print(f"Range: {min(times1):.4f}s - {max(times1):.4f}s")

# ============================================================
# [2] Optimized version
# ============================================================
print("\n[2] Optimized (eval + no_grad + GPU + TF32)")
print("-" * 60)

tokenizer2 = BertTokenizer.from_pretrained("Minej/bert-base-personality")
model2 = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality")
model2.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2 = model2.to(device)
print(f"Device: {device}")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Warmup GPU
    inputs = tokenizer2(test_text, truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        _ = model2(**inputs)

times2 = []
for i in range(num_runs):
    start = time.time()
    inputs = tokenizer2(test_text, truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model2(**inputs)
    # Apply sigmoid to match big5_optimized.py behavior
    logits = outputs.logits.squeeze()
    predictions = torch.sigmoid(logits).cpu().numpy()
    times2.append(time.time() - start)

avg2 = sum(times2) / len(times2)
print(f"Average: {avg2:.4f}s")
print(f"Range: {min(times2):.4f}s - {max(times2):.4f}s")

# ============================================================
# Summary
# ============================================================
speedup = avg1 / avg2
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Normal:      {avg1:.4f}s  (baseline)")
print(f"Optimized:   {avg2:.4f}s  ({speedup:.2f}x faster)")
print("=" * 60)
print(f"\n✓ Use big5_optimized.py for best performance")
print(f"  Inference time: ~{avg2:.4f}s per text")
