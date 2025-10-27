# Evaluation Summary

| Model | Precision | Recall | F1 | Exact Match | LLM Judge |
| --- | --- | --- | --- | --- | --- |
| llama_flash_kv-16.txt | 0.377 | 0.580 | 0.457 | 0.580 | 0.320 |
| mistral_flash_kv-16.txt | 0.453 | 0.480 | 0.466 | 0.480 | 0.100 |
| qwen_flash_kv-16.txt | 0.592 | 0.900 | 0.714 | 0.900 | 0.340 |

## Per-topic F1

### llama_flash_kv-16.txt
| Topic | Precision | Recall | F1 |
| --- | --- | --- | --- |
| basic_info | 1.000 | 1.000 | 1.000 |
| interests | 0.227 | 0.333 | 0.270 |
| mental_state | 0.643 | 0.450 | 0.529 |

### mistral_flash_kv-16.txt
| Topic | Precision | Recall | F1 |
| --- | --- | --- | --- |
| basic_info | 1.000 | 0.267 | 0.421 |
| interests | 0.556 | 0.667 | 0.606 |
| mental_state | 0.909 | 0.500 | 0.645 |

### qwen_flash_kv-16.txt
| Topic | Precision | Recall | F1 |
| --- | --- | --- | --- |
| basic_info | 1.000 | 1.000 | 1.000 |
| interests | 1.000 | 0.800 | 0.889 |
| mental_state | 1.000 | 0.900 | 0.947 |

## Latency (seconds)

| Model | Mean | Median | P95 | Total |
| --- | --- | --- | --- | --- |
| llama_flash_kv-16.txt | 460.414 | 398.406 | 560.330 | 2302.072 |
| mistral_flash_kv-16.txt | 258.388 | 250.363 | 306.543 | 1291.940 |
| qwen_flash_kv-16.txt | 349.893 | 310.632 | 393.218 | 1749.467 |

## Extraneous Output

| Model | Slots | Values |
| --- | --- | --- |
| llama_flash_kv-16.txt | 48 | 77 |
| mistral_flash_kv-16.txt | 29 | 40 |
| qwen_flash_kv-16.txt | 31 | 53 |

