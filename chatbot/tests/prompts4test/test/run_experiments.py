#!/usr/bin/env python3
"""
Automate extract + benchmark runs across different chunk sizes (N).

The script performs the following for each provided N:
1. Stops any running Ollama service.
2. Starts a fresh Ollama instance with Flash Attention and KV cache enabled.
3. Prefetches the model (warm-up) to remove load time from measurements.
4. Runs `extract.py` with a unique user id, capturing cost time and writing logs.
5. Executes `evaluate.py` to assess output quality (precision, recall, F1).
6. Records prompt/generation metrics by reading the tail of the Ollama log.
7. Saves a JSON summary under `results/experiment_summary_<timestamp>.json`.

Usage example:
    python3 run_experiments.py --user-prefix bench --rounds 5 10 20
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
OLLAMA_LOG_PATH = LOG_DIR / "ollama_experiment.log"
EXTRACT_SCRIPT = BASE_DIR / "extract.py"
EVALUATE_SCRIPT = BASE_DIR / "evaluate.py"
CHAT_DATA_DIR = "mock_user"  # Fixed directory containing test chat data


def generate_random_id(length: int = 4) -> str:
    """Generate a random alphanumeric ID for MemoBase user storage."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def run_cmd(
    cmd: List[str],
    check: bool = True,
    env: dict | None = None,
    capture_output: bool = False
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        env=env,
        capture_output=capture_output,
        text=True
    )


# def stop_ollama() -> None:
#     print("  [1/6] Stopping existing Ollama service...")
#     subprocess.run(["sudo", "systemctl", "stop", "ollama"], check=False)
#     subprocess.run(["pkill", "-9", "ollama"], check=False)
#     time.sleep(2)
#     print("  ✓ Ollama stopped")


# def start_ollama_with_features(log_path: Path) -> subprocess.Popen:
#     print("  [2/6] Starting Ollama with Flash Attention and KV Cache...")
#     env = {
#         **dict(OLLAMA_FLASH_ATTENTION="1", OLLAMA_KV_CACHE_TYPE="q8_0"),
#         **dict(PATH=str(Path.home() / ".local/bin") + ":" + Path("/usr/local/bin").as_posix()),
#     }
#     log_path.parent.mkdir(parents=True, exist_ok=True)
#     log_file = log_path.open("w", encoding="utf-8")
#     process = subprocess.Popen(
#         ["ollama", "serve"],
#         stdout=log_file,
#         stderr=subprocess.STDOUT,
#         env={**env, **dict(PATH="/usr/local/bin:" + env["PATH"])},
#     )
#     wait_for_ollama()
#     print("  ✓ Ollama started (Flash Attention=1, KV Cache=q8_0)")
#     return process


def wait_for_ollama(timeout: int = 60) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            subprocess.run(
                ["curl", "-s", "http://127.0.0.1:11434/api/tags"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except subprocess.CalledProcessError:
            time.sleep(1)
    raise RuntimeError("Failed to start Ollama within timeout.")


def warm_up_model(model: str = "qwen2.5:7b-instruct") -> None:
    print(f"  [3/6] Warming up model {model}...")
    subprocess.run(
        ["ollama", "run", model, "Warm-up request."],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("  ✓ Model warmed up")


def run_extract(rounds: int, user_id: str, skip_profile: bool = True) -> subprocess.CompletedProcess:
    """Run extract.py with specified rounds and user ID for MemoBase storage."""
    cmd = ["python3", str(EXTRACT_SCRIPT), "--user-id", user_id, "--rounds-per-chunk", str(rounds)]
    if skip_profile:
        cmd.append("--skip-profile")
    return run_cmd(cmd, check=True, capture_output=True)


def extract_duration_from_output(output: str) -> float | None:
    for line in output.splitlines():
        if line.strip().startswith("Cost time(s)"):
            try:
                return float(line.strip().split()[-1])
            except (IndexError, ValueError):
                continue
    return None


def tail_ollama_metrics(log_path: Path) -> Tuple[float | None, float | None, float | None, float | None]:
    if not log_path.exists():
        return None, None, None, None
    ttft = prompt = eval_duration = total = None
    with log_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()[-200:]
    for line in reversed(lines):
        if "prompt_eval_duration" in line and prompt is None:
            try:
                prompt = float(line.split("prompt_eval_duration\":")[1].split(",")[0]) / 1e9
                ttft = float(line.split("ttft\":")[1].split(",")[0])
                eval_duration = float(line.split("eval_duration\":")[1].split(",")[0]) / 1e9
                total = float(line.split("total_duration\":")[1].split("}")[0]) / 1e9
                break
            except (IndexError, ValueError):
                continue
    return ttft, prompt, eval_duration, total


def parse_evaluation_metrics(output: str) -> Dict[str, float | None]:
    """Parse precision, recall, F1, and redundancy from evaluate.py output."""
    metrics = {
        "precision": None,
        "recall": None,
        "f1": None,
        "redundancy_rate": None,
        "tp": None,
        "fp": None,
        "fn": None,
    }

    # Look for the "Overall ->" line
    # Example: "  Overall -> precision: 0.892, recall: 0.856, F1: 0.874 (TP: 145, FP: 18, FN: 24)"
    overall_pattern = r"Overall.*?precision:\s*([\d.]+).*?recall:\s*([\d.]+).*?F1:\s*([\d.]+)"
    match = re.search(overall_pattern, output, re.IGNORECASE)
    if match:
        metrics["precision"] = float(match.group(1))
        metrics["recall"] = float(match.group(2))
        metrics["f1"] = float(match.group(3))

    # Extract TP, FP, FN
    counts_pattern = r"TP:\s*(\d+),\s*FP:\s*(\d+),\s*FN:\s*(\d+)"
    match = re.search(counts_pattern, output)
    if match:
        metrics["tp"] = int(match.group(1))
        metrics["fp"] = int(match.group(2))
        metrics["fn"] = int(match.group(3))

    # Extract redundancy rate
    redundancy_pattern = r"Redundancy rate:\s*([\d.]+)"
    match = re.search(redundancy_pattern, output)
    if match:
        metrics["redundancy_rate"] = float(match.group(1))

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Automated extract benchmark runner.")
    parser.add_argument("--rounds", type=int, nargs="+", required=True, help="List of chunk sizes (N) to test.")
    parser.add_argument("--user-prefix", type=str, default="exp", help="Prefix for synthetic user ids.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip calling evaluate.py after extract.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AUTOMATED EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Configurations: Flash Attention=1, KV Cache=q8_0")
    print(f"Chat data source: chats/{CHAT_DATA_DIR}/")
    print(f"Rounds to test: {args.rounds}")
    print(f"Experiment prefix: {args.user_prefix}")
    print(f"Skip evaluation: {args.skip_eval}")
    print("=" * 80)
    print()

    summary: Dict[str, Dict[str, float | str | int | None]] = {}

    # Warm up the model once at the beginning
    print("=" * 80)
    print("Warming up model...")
    print("=" * 80)
    try:
        warm_up_model()
        print("✓ Model warmed up and ready\n")
    except Exception as e:
        print(f"⚠ Warning: Model warm-up failed: {e}")
        print("Continuing with experiments...\n")

    for idx, rounds in enumerate(args.rounds, start=1):
        # Generate a unique MemoBase user ID (random 4-char string)
        memobase_user_id = generate_random_id(4)
        experiment_id = f"{args.user_prefix}_n{rounds}_{memobase_user_id}"

        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT {idx}/{len(args.rounds)}: N={rounds} rounds per chunk")
        print(f"Experiment ID: {experiment_id}")
        print(f"MemoBase User ID: {memobase_user_id}")
        print(f"{'=' * 80}\n")

        # No need to restart Ollama for each experiment
        # Just run the extraction directly

        print(f"  [1/3] Running extraction (N={rounds})...")
        print(f"      Processing all chat files in chats/{CHAT_DATA_DIR}/")
        try:
            result = run_extract(rounds, user_id=memobase_user_id, skip_profile=True)
            duration = extract_duration_from_output(result.stdout)
            print(f"  ✓ Extraction completed in {duration:.2f}s" if duration else "  ✓ Extraction completed")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Extraction failed with error code {e.returncode}")
            print("=" * 80)
            print("STDOUT:")
            print(e.stdout if e.stdout else "(no stdout)")
            print("\nSTDERR:")
            print(e.stderr if e.stderr else "(no stderr)")
            print("=" * 80)
            raise

        # Run evaluation
        eval_metrics = {}
        if not args.skip_eval:
            print("  [2/3] Running evaluation...")
            eval_result = run_cmd(
                ["python3", str(EVALUATE_SCRIPT)],
                check=True,
                capture_output=True
            )
            eval_output = eval_result.stdout
            eval_metrics = parse_evaluation_metrics(eval_output)

            # Print evaluation results
            if eval_metrics.get("precision") is not None:
                print(f"  ✓ Evaluation completed:")
                print(f"    - Precision: {eval_metrics['precision']:.3f}")
                print(f"    - Recall: {eval_metrics['recall']:.3f}")
                print(f"    - F1 Score: {eval_metrics['f1']:.3f}")
                if eval_metrics.get("redundancy_rate") is not None:
                    print(f"    - Redundancy: {eval_metrics['redundancy_rate']:.3f}")
                print(f"    - TP={eval_metrics['tp']}, FP={eval_metrics['fp']}, FN={eval_metrics['fn']}")
            else:
                print("  ⚠ Evaluation completed but metrics not found")
        else:
            print("  [2/3] Skipping evaluation")

        print("  [3/3] Recording Ollama metrics...")
        ttft, prompt, eval_duration_ollama, total_duration = tail_ollama_metrics(OLLAMA_LOG_PATH)

        if ttft is not None:
            print(f"  ✓ Ollama metrics recorded:")
            print(f"    - TTFT: {ttft:.3f}s")
            print(f"    - Prompt eval: {prompt:.3f}s")
            print(f"    - Eval duration: {eval_duration_ollama:.3f}s")
            print(f"    - Total duration: {total_duration:.3f}s")
        else:
            print("  ⚠ Ollama metrics not found (check if Ollama is logging to the expected location)")

        summary[experiment_id] = {
            "experiment_id": experiment_id,
            "rounds": rounds,
            "memobase_user_id": memobase_user_id,
            "chat_data_source": f"chats/{CHAT_DATA_DIR}/",
            "duration": duration,
            "ttft": ttft,
            "prompt_eval_duration": prompt,
            "eval_duration": eval_duration_ollama,
            "total_duration": total_duration,
            **eval_metrics,  # Add all evaluation metrics
        }

        print(f"\n✓ Experiment {idx}/{len(args.rounds)} completed")
        print(f"  - Experiment ID: {experiment_id}")
        print(f"  - MemoBase User ID: {memobase_user_id}\n")

    # Save results
    output_file = RESULTS_DIR / f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)
    print(f"\n📊 Results Summary (Data source: chats/{CHAT_DATA_DIR}/):")
    print(f"{'User ID':>10} | {'N':>5} | {'Duration':>10} | {'TTFT':>8} | {'Precision':>9} | {'Recall':>9} | {'F1':>9}")
    print("-" * 90)

    for exp_id, data in summary.items():
        user_id = data["memobase_user_id"]
        n = data["rounds"]
        dur = f"{data['duration']:.2f}s" if data.get("duration") else "N/A"
        ttft_val = f"{data['ttft']:.3f}s" if data.get("ttft") else "N/A"
        prec = f"{data['precision']:.3f}" if data.get("precision") is not None else "N/A"
        rec = f"{data['recall']:.3f}" if data.get("recall") is not None else "N/A"
        f1 = f"{data['f1']:.3f}" if data.get("f1") is not None else "N/A"
        print(f"{user_id:>10} | {n:>5} | {dur:>10} | {ttft_val:>8} | {prec:>9} | {rec:>9} | {f1:>9}")

    print("\n" + "=" * 80)
    print(f"📁 Detailed results saved to: {output_file}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
