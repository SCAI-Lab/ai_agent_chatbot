import json
import re
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List

import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

# List of models to benchmark
models_to_benchmark: List[str] = [
    "openai/whisper-large-v3-turbo",
    "openai/whisper-large-v3",
    "openai/whisper-medium",
    "openai/whisper-small",
    "openai/whisper-base",
    "openai/whisper-tiny",
]

RESULT_OUTPUT_PATH = Path("/home/user/ai_agent/whisper4test/results.txt")
AUDIO_DATA_ROOT = Path(__file__).with_name("audios")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Represents a single precision/batch/attention combination."""

    precision_name: str
    torch_dtype: torch.dtype
    batch_size: int
    chunk_length_s: int
    use_flash_attention: bool

    @property
    def id(self) -> str:
        attn = "flash" if self.use_flash_attention else "sdpa"
        return f"{self.precision_name}_batch{self.batch_size}_{attn}"


@dataclass(frozen=True)
class DatasetSpec:
    """Container describing a single evaluation difficulty bucket."""

    name: str
    audio_files: List[Path]
    transcripts: Dict[str, str]


def _normalise_text(text: str) -> List[str]:
    """Uppercase text and strip punctuation so WER comparisons are fair."""

    if not text:
        return []
    clean = re.sub(r"[^A-Z0-9\s]", " ", text.upper())
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean.split()


def _normalise_transcript_key(name: str) -> str:
    """Normalise audio/transcript identifiers for reliable matching."""

    if not name:
        return ""
    stem = Path(name).stem

    def _strip(match) -> str:
        digits = match.group(0)
        stripped = digits.lstrip("0")
        return stripped if stripped else "0"

    return re.sub(r"\d+", _strip, stem)


def _word_error_distance(reference: List[str], hypothesis: List[str]) -> int:
    """Compute total Levenshtein distance between reference and hypothesis."""

    n = len(reference)
    m = len(hypothesis)

    if n == 0:
        return m
    if m == 0:
        return n

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        ref_token = reference[i - 1]
        for j in range(1, m + 1):
            hyp_token = hypothesis[j - 1]
            cost = 0 if ref_token == hyp_token else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution / match
            )
    return dp[n][m]


def build_configurations() -> List[BenchmarkConfig]:
    """Generate the grid of configurations to evaluate."""

    dtype_options = [
        ("fp16", torch.float16),
        ("fp32", torch.float32),
    ]
    batch_sizes = [8, 16, 24]
    flash_options = [True, False]
    chunk_length_s = 10

    configs: List[BenchmarkConfig] = []
    for (precision_name, torch_dtype), batch_size, use_flash in product(
        dtype_options, batch_sizes, flash_options
    ):
        if precision_name == "fp32" and use_flash:
            # Flash attention not supported in fp32 mode
            continue

        configs.append(
            BenchmarkConfig(
                precision_name=precision_name,
                torch_dtype=torch_dtype,
                batch_size=batch_size,
                chunk_length_s=chunk_length_s,
                use_flash_attention=use_flash,
            )
        )
    return configs


def load_datasets(root: Path = AUDIO_DATA_ROOT) -> List[DatasetSpec]:
    """Load audio paths and reference transcripts for each difficulty level."""

    if not root.exists():
        raise FileNotFoundError(f"Audio dataset directory not found: {root}")

    datasets: List[DatasetSpec] = []
    for difficulty_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        transcript_path = difficulty_dir / "texts.txt"
        audio_dir = difficulty_dir / "files"

        if not transcript_path.exists():
            raise FileNotFoundError(f"Missing transcript file: {transcript_path}")
        if not audio_dir.exists():
            raise FileNotFoundError(f"Missing audio directory: {audio_dir}")

        transcripts: Dict[str, str] = {}
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                key, text_value = line.split(" ", 1)
            except ValueError as exc:
                raise ValueError(f"Malformed transcript line: {line!r}") from exc

            normalised_key = _normalise_transcript_key(key.strip())
            if normalised_key in transcripts:
                raise ValueError(
                    "Duplicate transcript id after normalisation: "
                    f"{key!r} conflicts with another entry in {transcript_path}"
                )

            transcripts[normalised_key] = text_value.strip()

        audio_files = sorted(
            p for p in audio_dir.iterdir() if p.is_file() and p.stat().st_size > 0
        )
        if not audio_files:
            raise ValueError(f"No audio files discovered in {audio_dir}")

        missing = [
            audio.stem
            for audio in audio_files
            if _normalise_transcript_key(audio.stem) not in transcripts
        ]
        if missing:
            raise ValueError(
                "Missing transcripts for files in"
                f" {difficulty_dir}: {', '.join(missing[:3])}"
            )

        datasets.append(
            DatasetSpec(
                name=difficulty_dir.name,
                audio_files=audio_files,
                transcripts=transcripts,
            )
        )

    if not datasets:
        raise ValueError(f"No difficulty folders found in {root}")

    return datasets


def _build_error_result(
    model_name: str,
    config: BenchmarkConfig,
    error: str,
    dataset_names: List[str],
) -> List[dict]:
    """Return one error record per dataset placeholder when evaluation skipped."""

    records = []
    for dataset_name in dataset_names:
        records.append(
            {
                "model": model_name,
                "config": config.id,
                "precision": config.precision_name,
                "batch_size": config.batch_size,
                "flash_attention": config.use_flash_attention,
                "load_time": 0.0,
                "difficulty": dataset_name,
                "total_inference_time": 0.0,
                "total_words": 0,
                "word_errors": 0,
                "accuracy": None,
                "word_error_rate": None,
                "sample_transcript": "",
                "error": error,
            }
        )
    return records


def benchmark_model(
    model_name: str,
    config: BenchmarkConfig,
    datasets: List[DatasetSpec],
) -> List[dict]:
    """Benchmark a single model/config pair and return timing results."""

    separator = "=" * 60
    print()
    print(separator)
    print(f"Benchmarking: {model_name} | Config: {config.id}")
    print(separator)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    flash_available = is_flash_attn_2_available()
    dataset_names = [dataset.name for dataset in datasets]

    if device == "cpu" and config.torch_dtype == torch.float16:
        error = "Skipping: float16 not supported on CPU"
        print(error)
        return _build_error_result(model_name, config, error, dataset_names)

    if config.use_flash_attention and not flash_available:
        error = "Skipping: flash attention not available"
        print(error)
        return _build_error_result(model_name, config, error, dataset_names)

    attn_impl = "flash_attention_2" if config.use_flash_attention else "sdpa"

    try:
        start_load = time.time()
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=config.torch_dtype,
            device=device,
            model_kwargs={"attn_implementation": attn_impl},
        )
        load_time = time.time() - start_load
        print(f"Model loading time: {load_time:.2f} seconds")

        warmup_batch = min(config.batch_size, 16)
        first_audio = datasets[0].audio_files[0]
        try:
            pipe(
                str(first_audio),
                chunk_length_s=config.chunk_length_s,
                batch_size=warmup_batch,
                return_timestamps=False,
            )
        except Exception:
            pass

        dataset_results: List[dict] = []

        for dataset in datasets:
            total_inference_time = 0.0
            total_words = 0
            total_errors = 0
            sample_outputs: List[str] = []

            for audio_path in dataset.audio_files:
                start_inference = time.time()
                outputs = pipe(
                    str(audio_path),
                    chunk_length_s=config.chunk_length_s,
                    batch_size=config.batch_size,
                    return_timestamps=False,
                )
                inference_time = time.time() - start_inference
                total_inference_time += inference_time

                predicted_text = outputs.get("text", "")
                lookup_key = _normalise_transcript_key(audio_path.stem)
                reference_text = dataset.transcripts.get(lookup_key, "")

                hyp_tokens = _normalise_text(predicted_text)
                ref_tokens = _normalise_text(reference_text)

                total_words += len(ref_tokens)
                total_errors += _word_error_distance(ref_tokens, hyp_tokens)
                if predicted_text:
                    sample_outputs.append(predicted_text)

            accuracy = None
            word_error_rate = None
            if total_words > 0:
                word_error_rate = total_errors / total_words
                accuracy = max(0.0, 1.0 - word_error_rate)

            accuracy_text = "N/A"
            if accuracy is not None:
                accuracy_text = f"{accuracy * 100:.2f}%"

            print(
                f"[{dataset.name}] total time: {total_inference_time:.2f}s, "
                f"total words: {total_words}, accuracy: {accuracy_text}"
            )

            dataset_results.append(
                {
                    "model": model_name,
                    "config": config.id,
                    "precision": config.precision_name,
                    "batch_size": config.batch_size,
                    "flash_attention": config.use_flash_attention,
                    "load_time": load_time,
                    "difficulty": dataset.name,
                    "total_inference_time": total_inference_time,
                    "total_words": total_words,
                    "word_errors": total_errors,
                    "accuracy": accuracy,
                    "word_error_rate": word_error_rate,
                    "sample_transcript": sample_outputs[0][:200] if sample_outputs else "",
                    "error": None,
                }
            )

        return dataset_results

    except Exception as exc:  # noqa: BLE001 - surface errors clearly to CLI
        print(f"Error with model {model_name}: {str(exc)}")
        return _build_error_result(model_name, config, str(exc), dataset_names)


def initialise_results_output(file_path: Path = RESULT_OUTPUT_PATH) -> None:
    """Create or truncate the results file before streaming records."""

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("", encoding="utf-8")



def write_results_to_file(records: List[dict], file_path: Path = RESULT_OUTPUT_PATH) -> None:
    """Append the latest benchmark records to disk in JSON Lines format."""

    if not records:
        return

    with file_path.open("a", encoding="utf-8") as handle:
        for item in records:
            serialisable = {
                "model": item["model"],
                "config": item.get("config"),
                "precision": item.get("precision"),
                "batch_size": item.get("batch_size"),
                "flash_attention": item.get("flash_attention"),
                "load_time": item.get("load_time"),
                "difficulty": item.get("difficulty"),
                "total_inference_time": item.get("total_inference_time"),
                "total_words": item.get("total_words"),
                "word_errors": item.get("word_errors"),
                "accuracy": item.get("accuracy"),
                "word_error_rate": item.get("word_error_rate"),
                "sample_transcript": item.get("sample_transcript", ""),
                "error": item.get("error"),
            }
            handle.write(json.dumps(serialisable, ensure_ascii=False))
            handle.write("\n")

    print(f"Appended {len(records)} records to {file_path.resolve()}")


def main():
    print("Speech-to-Text Model Benchmark")
    print("=" * 50)

    results: List[dict] = []
    configs = build_configurations()
    datasets = load_datasets()

    initialise_results_output()

    for model_name in models_to_benchmark:
        for config in configs:
            records = benchmark_model(model_name, config, datasets)
            results.extend(records)
            write_results_to_file(records)

    print()
    print()
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    header = (
        f"{'Model':<30} {'Config':<30} {'Difficulty':<12} "
        f"{'Total Time':<15} {'Accuracy':<12} {'Word Errs':<10}"
    )
    print(header)
    print(f"{'-' * 80}")

    for result in results:
        if result.get("error"):
            print(
                f"{result['model']:<30} {result['config']:<30} "
                f"{result.get('difficulty', '-'):>12} {'ERROR':<15} {'-':<12} {'-':<10}"
            )
            continue

        accuracy_display = "N/A"
        if result["accuracy"] is not None:
            accuracy_display = f"{result['accuracy'] * 100:>8.2f}%"

        print(
            f"{result['model']:<30} {result['config']:<30} "
            f"{result.get('difficulty', '-'):>12} {result['total_inference_time']:>8.2f}s "
            f"{accuracy_display:<12} {result['word_errors']:>10}"
        )


if __name__ == "__main__":
    main()
