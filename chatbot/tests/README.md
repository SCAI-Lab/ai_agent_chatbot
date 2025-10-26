# Chatbot Test Suite

This directory contains a comprehensive test suite for the chatbot system, covering multiple testing dimensions: conversation quality testing, prompt engineering testing, speech recognition testing, and cache performance testing.

## Directory Structure

```
tests/
├── chat4test/          # Conversation quality assessment tests
├── prompts4test/       # Prompt engineering experiment tests
├── whisper4test/       # Whisper speech recognition benchmark tests
└── cache4test/         # Cache and performance benchmark tests
```

---

## 1. chat4test - Conversation Quality Assessment Tests

### Overview
Tests the accuracy and quality of the chatbot's ability to extract user information from conversations, evaluating system performance across different conversational scenarios.

### Directory Structure
```
chat4test/
├── chats/
│   ├── mock_user/      # Mock user conversation data (JSON format)
│   ├── ground_truth/   # Ground truth data (TXT format)
│   └── output/         # Model output results
├── extract.py          # Information extraction script
└── evaluate.py         # Evaluation script
```

### Testing Workflow

#### Step 1: Information Extraction (extract.py)
Extracts user information from conversations and inserts it into the MemoBase system.

**Usage:**
```bash
cd chat4test
python extract.py --user 54 \
                  --project_url http://localhost:8019 \
                  --project_token secret \
                  --rounds-per-chunk 5 \
                  [--skip-profile]
```

**Parameters:**
- `--user`: User ID, corresponds to subdirectory under `chats/mock_user/`
- `--project_url`: MemoBase project URL
- `--project_token`: API access token
- `--rounds-per-chunk`: Number of user turns per data chunk (default: 5)
- `--skip-profile`: Skip user profile retrieval (optional)

**Test Coverage:**
1. Read conversation data from `chats/{user}/*.json`
2. Chunk conversations by specified number of rounds
3. Insert conversation data using MemoBase Client
4. Measure insertion time and performance
5. Extract user profile features

**Output Format:**
```
File: ./chats/54/1.json
  Chunks:
  Chunk 1: 5 user turns, 5 assistant replies
  Chunk 2: 3 user turns, 3 assistant replies
Total chats: 2
Cost time(s) 12.34
* basic_info: name - Ethan
* basic_info: age - 20
* interests: music - electronic, ambient
```

#### Step 2: Evaluation (evaluate.py)
Compares extraction results with ground truth, calculating precision, recall, and F1 score.

**Usage:**
```bash
cd chat4test
python evaluate.py [--output-dir chats/output] \
                   [--ground-truth-dir chats/ground_truth]
```

**Evaluation Metrics:**
- **Precision**: TP / (TP + FP) - Accuracy of extracted information
- **Recall**: TP / (TP + FN) - Completeness of extracted information
- **F1 Score**: Harmonic mean of precision and recall
- **Redundancy Rate**: FP / (TP + FP) - Ratio of redundant information
- **Time Metrics**: Total time, average time, standard deviation

**Output Example:**
```
Model: baseline
  Overall -> precision: 0.892, recall: 0.856, F1: 0.874 (TP: 145, FP: 18, FN: 24)
  Redundancy rate: 0.110
  Time -> total: 45.23s, mean: 9.05s, std: 2.31s
  Per topic:
    basic_info: P=0.950, R=0.920, F1=0.935 (TP=46, FP=2, FN=4)
    interests: P=0.875, R=0.840, F1=0.857 (TP=42, FP=6, FN=8)
    mental_state: P=0.860, R=0.810, F1=0.834 (TP=57, FP=10, FN=12)
```
*** METRICS TO BE UPDATED ***

### Data Formats

#### Conversation Data Format (mock_user/*.json)
```json
[
  {"role": "user", "content": "Hi, I'm Ethan."},
  {"role": "assistant", "content": "Nice to meet you, Ethan!"},
  {"role": "user", "content": "I'm 20 years old and love coding."}
]
```

#### Ground Truth Format (ground_truth/*.txt)
```
basic_info::name::Ethan
basic_info::age::20
interests::activities::coding, piano
interests::music::electronic, ambient
mental_state::mood::anxious, calm
```

Format rule: `topic::subtopic::value1, value2, value3`

---

## 2. prompts4test - Prompt Engineering Experiment Tests

### Overview
Tests the impact of different system prompts and chunk sizes. 
*** The result of these tests are not included in the current version due to the instability of compressed prompts. ***
```
### Testing Workflow

#### Main Test: Automated Experiments (run_experiments.py)

**Functionality:**
Automatically runs multiple experiment groups to test the impact of different chunk sizes (rounds) on performance. Each experiment includes:
1. Change the system prompts in memobase
2. Restart memobase
3. Run extract.py test
4. Record performance metrics 
5. Save experiment results

**Usage:**
```bash
cd prompts4test/test
python3 run_experiments.py --rounds 5 10 20 \
                           --user-prefix bench \
                           [--skip-eval]
```

**Parameters:**
- `--rounds`: List of chunk sizes to test (e.g., 5, 10, 20 rounds)
- `--user-prefix`: Experiment user ID prefix
- `--skip-eval`: Skip evaluation step (optional)

```

#### Legacy Test: Two-Stage Prompt Testing (old_test/prompt_test.py)

**Functionality:**
Tests a "summarize + extract" two-stage processing workflow, comparing different models' performance at different difficulty levels.

**Two-Stage Process:**
1. **Summary Stage**: Use summary prompt to summarize conversations
2. **Extract Stage**: Use extract prompt to extract structured information from summary

**Difficulty Levels:**
- Easy: 10 conversation rounds
- Medium: 30 conversation rounds
- Hard: 50 conversation rounds

**Usage:**
```bash
cd prompts4test/old_test
python prompt_test.py
```

**Test Models (configurable):**
- mistral:7b-instruct
- qwen2.5:7b-instruct
- phi3:3.8b
- gemma3:4b
- llama3.1:8b

**Output Metrics:**
- Processing time per stage
- Number of input tokens
- Total time and total tokens for both stages

---

## 3. cache4test - Cache and Performance Benchmark Tests

### Overview
Tests the impact of different Ollama configurations (Flash Attention, KV Cache) on inference performance.

### Directory Structure
```
cache4test/
├── memobase/
│   ├── chats/          # Test conversation data
│   ├── main.py         # MemoBase main test script
│   ├── extract.py      # Information extraction script
│   └── evaluate.py     # Evaluation script
├── benchmark_extract.py    # Benchmark main script
└── benchmark_utils.py      # Benchmark utility functions
```

### Testing Workflow

#### Benchmark Testing (benchmark_extract.py)

**Functionality:**
Runs extraction benchmark tests under specific configurations, measuring performance metrics.

**Usage:**
```bash
cd cache4test
python benchmark_extract.py --config baseline --user mock_user
python benchmark_extract.py --config flash --user mock_user
python benchmark_extract.py --config kvcache --user mock_user
python benchmark_extract.py --config both --user mock_user
```

**Configuration Options:**
- `baseline`: Baseline configuration (no optimizations)
- `flash`: Enable Flash Attention
- `kvcache`: Enable KV Cache
- `both`: Enable both optimizations

**Test Metrics:**
- **TTFT (Time To First Token)**: Time to generate first token
- **Prompt Eval Duration**: Prompt evaluation time
- **Eval Duration**: Generation evaluation time
- **Generation Speed**: Generation speed (tokens/second)
- **Total Time**: Total processing time

**Output Results:**
Saved to `outputs/extract_benchmark_<config>_<timestamp>.json`:
```json
{
  "config": "flash",
  "model": "qwen2.5:7b-instruct",
  "chunk_metrics": [
    {
      "ttft": 0.123,
      "prompt_eval_duration": 1.234,
      "eval_duration": 0.567,
      "generation_speed": 45.6,
      "total_time": 2.456,
      "chunk_index": 1,
      "user_turns": 5,
      "assistant_turns": 5
    }
  ],
  "summary": {
    "avg_ttft": 0.128,
    "avg_prompt_duration": 1.245,
    "avg_eval_duration": 0.578,
    "avg_generation_speed": 44.8,
    "total_time": 24.56
  }
}
```

#### Utility Functions (benchmark_utils.py)

Provides the following core functionalities:
1. **load_system_prompt()**: Load system prompts
2. **chunk_messages()**: Chunk conversations
3. **load_chat_chunks()**: Load conversation chunks from disk
4. **build_prompt_input()**: Build prompt input format
5. **measure_streaming_performance()**: Measure streaming generation performance

**Prompt Format:**
```
### Already Logged
None
### Input Chats
[CHAT_01] User: Hi, I'm Ethan.
[CHAT_02] Assistant: Nice to meet you!
[CHAT_03] User: I'm 20 years old.
```

---

## 4. whisper4test - Whisper Speech Recognition Benchmark Tests

### Overview
Comprehensive testing of Whisper model speech recognition performance under different configurations, including accuracy and inference speed.

### Directory Structure
```
whisper4test/
├── audios/
│   ├── easy/
│   │   ├── files/      # Easy audio files
│   │   └── texts.txt   # Transcription texts
│   ├── medium/
│   │   ├── files/
│   │   └── texts.txt
│   └── hard/
│       ├── files/
│       └── texts.txt
├── whisper_test.py                 # Main test script
└── compute_whisper_averages.py     # Results analysis script
```

### Testing Workflow

#### Main Test Script (whisper_test.py)

**Functionality:**
Systematically tests multiple Whisper model variants under different precision, batch size, and attention mechanism configurations.

**Test Models:**
- whisper-large-v3-turbo
- whisper-large-v3
- whisper-medium
- whisper-small
- whisper-base
- whisper-tiny

**Configuration Matrix:**
- **Precision**: FP16, FP32
- **Batch Size**: 8, 16, 24
- **Attention Mechanisms**:
  - SDPA (Scaled Dot-Product Attention)
  - Flash Attention 2 (FP16 only)
- **Audio Chunk Length**: 10 seconds

**Configuration Examples:**
- `fp16_batch8_flash`: FP16 precision + 8 batch + Flash Attention
- `fp16_batch16_sdpa`: FP16 precision + 16 batch + SDPA
- `fp32_batch24_sdpa`: FP32 precision + 24 batch + SDPA

**Usage:**
```bash
cd whisper4test
python whisper_test.py
```

**Test Process:**
1. Load audio dataset (easy/medium/hard)
2. Load corresponding transcription ground truth
3. For each model and configuration combination:
   - Load model and measure loading time
   - Perform warm-up inference
   - For each difficulty level audio file:
     - Perform speech recognition
     - Measure inference time
     - Calculate Word Error Rate (WER)
   - Record performance metrics
4. Save results to `results.txt` (JSON Lines format)

**Evaluation Metrics:**
- **Accuracy**: Recognition accuracy = max(0, 1 - WER)
- **Word Error Rate (WER)**: Word error rate (based on Levenshtein distance)
- **Load Time**: Model loading time
- **Inference Time**: Total inference time
- **Total Words**: Total word count
- **Word Errors**: Word error count

**Output Example:**
```
============================================================
Benchmarking: openai/whisper-medium | Config: fp16_batch16_flash
============================================================
Model loading time: 12.34 seconds
[easy] total time: 5.23s, total words: 450, accuracy: 98.50%
[medium] total time: 15.67s, total words: 1350, accuracy: 96.20%
[hard] total time: 25.89s, total words: 2250, accuracy: 92.80%

Appended 3 records to /home/user/ai_agent/whisper4test/results.txt
```

**Final Summary:**
```
================================================================================
BENCHMARK SUMMARY
================================================================================
Model                          Config                         Difficulty   Total Time      Accuracy     Word Errs
--------------------------------------------------------------------------------
openai/whisper-medium          fp16_batch16_flash             easy              5.23s      98.50%            7
openai/whisper-medium          fp16_batch16_flash             medium           15.67s      96.20%           51
openai/whisper-medium          fp16_batch16_flash             hard             25.89s      92.80%          162
```

### Data Formats

#### Transcription Text Format (texts.txt)
```
audio_001 This is the transcription of the first audio file.
audio_002 This is the transcription of the second audio file.
audio_003 Another example transcription text.
```

Each line format: `<audio_id> <transcription_text>`

### Evaluation Method

**Word Error Rate Calculation:**
Uses Levenshtein edit distance algorithm to calculate the difference between reference and recognized text:
- Deletion operations
- Insertion operations
- Substitution operations

**Text Normalization:**
For fair comparison, all text undergoes normalization:
1. Convert to uppercase
2. Remove punctuation
3. Normalize whitespace
4. Tokenize

**Variant Generation:**
Considers different word variations (singular/plural, tense, etc.) to improve matching accuracy.

---

## Common Testing Tools and Methods

### Data Chunking Strategy

All conversation tests use a unified chunking strategy:
```python
def chunk_messages(messages, rounds_per_chunk=5):
    """
    Chunk conversations by user turns
    - Each chunk contains specified number of complete user-assistant conversation rounds
    - Ensures each chunk ends with an assistant reply
    """
```

### Performance Measurement Metrics

1. **Time Metrics:**
   - Cost Time: Total processing time
   - TTFT: Time to first token
   - Prompt Eval Duration: Prompt evaluation time
   - Eval Duration: Generation evaluation time

2. **Accuracy Metrics:**
   - Precision: Precision rate
   - Recall: Recall rate
   - F1 Score: F1 score
   - Accuracy: Accuracy rate
   - WER: Word error rate

3. **Efficiency Metrics:**
   - Generation Speed: Generation speed (tokens/second)
   - Redundancy Rate: Redundancy rate
   - Token Count: Token count

### Results Output Formats

- **Text Format**: For manual review (.txt)
- **JSON Format**: For programmatic analysis (.json)
- **JSON Lines Format**: For streaming append (.txt with JSONL)

---

## Quick Start

### 1. Run Complete Conversation Quality Test
```bash
cd chat4test
python extract.py --user 54 --skip-profile
python evaluate.py
```

### 2. Run Automated Prompt Experiments
```bash
cd prompts4test/test
python3 run_experiments.py --rounds 5 10 20 --user-prefix exp
```

### 3. Run Cache Performance Tests
```bash
cd cache4test
for config in baseline flash kvcache both; do
    python benchmark_extract.py --config $config --user mock_user
done
```

### 4. Run Whisper Benchmark Tests
```bash
cd whisper4test
python whisper_test.py
```

---

## Dependencies

### Python Packages
```
- memobase
- ollama
- transformers
- torch
- httpx
- rich
- tiktoken (optional, for token counting)
```

### External Services
- **MemoBase Server**: For chat4test (default: http://localhost:8019)
- **Ollama**: For prompts4test and cache4test (default: http://localhost:11434)
- **CUDA**: For whisper4test (GPU inference, optional)

---

## Notes

1. **Resource Requirements:**
   - Whisper tests require significant GPU memory (recommended 16GB+ for large models)
   - Automated experiments will restart Ollama service multiple times, requiring root privileges

2. **Data Preparation:**
   - Ensure conversation data and ground truth data correspond one-to-one
   - Audio file and transcription text IDs must match

3. **Configuration Validation:**
   - Flash Attention requires hardware support
   - FP16 not available on CPU

4. **Results Analysis:**
   - All tests generate detailed logs and result files
   - JSON format facilitates subsequent data analysis and visualization

---

## Extension and Customization

### Adding New Test Models
Modify model list in respective test scripts:
```python
models_to_benchmark = [
    "your-model-name",
    # ...
]
```

### Adjusting Test Parameters
Modify constants in scripts or use command-line arguments:
```python
ROUNDS_PER_CHUNK = 5  # Conversation rounds per chunk
MAX_TOKENS = 100      # Maximum generation tokens
```

### Custom Evaluation Metrics
Extend evaluation logic in `evaluate.py` to add new metric calculations.

---

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   - Confirm Ollama service is running
   - Check port configuration (default 11434)

2. **Model Loading Failed**
   - Confirm model is downloaded: `ollama pull <model-name>`
   - Check available memory

3. **Abnormal Evaluation Results**
   - Verify ground truth format is correct
   - Check data file encoding (should be UTF-8)

4. **GPU Out of Memory**
   - Reduce batch size
   - Use smaller model variants
   - Lower precision (use FP16)

---

## Contributing Guidelines

When adding new tests, follow existing directory structure and naming conventions:
- Use descriptive directory names (e.g., `<feature>4test`)
- Provide complete data samples and ground truth
- Include clear usage instructions and parameter descriptions
- Output structured result files

---

## References

- MemoBase API Documentation
- Ollama API Documentation
- Whisper Model Documentation
- Transformers Library Documentation
