=== OLLAMA MULTI-STAGE SIMULATION REPORT (Dual-Phase Evaluation) ===
Reference: Qwen2.5:7b-instruct (easy)
Standard: concise structured Summary + hierarchical Extract with mention dates
──────────────────────────────────────────────────────────────────────────────

[MODEL] gemma3:4b
  easy:
    Summary → ✅ matches reference (clear bullet events, concise)
    Extract → ✅ strong (TOPIC fields, correct timestamps)
    total_time: 93.17s
  medium:
    Summary → ⚠️ semi-structured (adds commentary, slight drift)
    Extract → ✅ clean structure, well aligned with Summary
    total_time: 83.36s
  hard:
    Summary → ❌ failed (essay / dataset analysis, no structure)
    Extract → ⚠️ partly recovered (contains profile fields but verbose)
    total_time: 164.18s
  overall → ★★★★☆ (Summary weak at hard stage but Extract solid)

[MODEL] mistral:7b-instruct
  easy:
    Summary → ✅ structured and concise (mirrors Qwen2.5 easy)
    Extract → ✅ clean TOPIC layout
    total_time: 109.24s
  medium:
    Summary → ✅ highly detailed yet stays structured
    Extract → ✅ well formed (temporal breakdown per day)
    total_time: 276.93s
  hard:
    Summary → ⚠️ reflective prose (no bullet structure)
    Extract → ⚠️ minimal structure but partial compliance
    total_time: 116.93s
  overall → ★★★★☆ (excellent through medium; hard drifted)

[MODEL] qwen2.5:7b-instruct
  easy:
    Summary → ✅ reference baseline (perfect bullet structure)
    Extract → ✅ reference baseline (ideal TOPIC hierarchy)
    total_time: 135.87s
  medium:
    Summary → ⚠️ verbose narrative (not strictly bulletized)
    Extract → ✅ correct structured profile
    total_time: 209.64s
  hard:
    Summary → ❌ guide-like output (not summary)
    Extract → ⚠️ partial structure maintained
    total_time: 146.17s
  overall → ★★★★☆ (reference only at easy; later summaries drift)

[MODEL] phi3:3.8b
  easy:
    Summary → ⚠️ verbose reasoning, semi-structured
    Extract → ⚠️ overly inferential but still hierarchical
    total_time: 231.65s
  medium:
    Summary → ⚠️ narrative mix, weak structure
    Extract → ⚠️ hybrid of reasoning + structure
    total_time: 139.32s
  hard:
    Summary → ❌ turned into weekly essay plan
    Extract → ❌ unstructured long text
    total_time: 171.41s
  overall → ★★☆☆☆ (both phases verbose and inconsistent)

[MODEL] llama3.1:8b
  easy:
    Summary → ❌ refused content (no usable data)
    Extract → ❌ replaced by safety message
    total_time: 20.82s
  medium:
    Summary → ⚠️ acceptable structured bullets
    Extract → ⚠️ replaced with disclaimer, partial profile recovery
    total_time: 163.37s
  hard:
    Summary → ⚠️ good narrative but not formatted
    Extract → ⚠️ partial structured list recovered
    total_time: 162.25s
  overall → ★★☆☆☆ (safety filter blocks full compliance)


