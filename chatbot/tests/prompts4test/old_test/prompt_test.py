import os
import time
from typing import Any, Dict, List, Tuple

from ollama import chat


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def safe_write(log, text: str) -> None:
    log.write(text)
    log.flush()
    os.fsync(log.fileno())


def try_load_tokenizer():
    try:
        import tiktoken  # type: ignore
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


_TOKENIZER = try_load_tokenizer()


def count_tokens(text: str) -> int:
    if _TOKENIZER:
        return len(_TOKENIZER.encode(text))
    return len(text.split())


def extract_chat_lines(chat_text: str) -> List[str]:
    lines: List[str] = []
    for raw_line in chat_text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("[") and "]" in stripped:
            lines.append(stripped)
    return lines


def build_conversation_segments(
    chat_lines: List[str], rounds: Dict[str, int]
) -> Tuple[Dict[str, str], Dict[str, int]]:
    segments: Dict[str, str] = {}
    actual_rounds: Dict[str, int] = {}
    total_entries = len(chat_lines)

    for label, round_count in rounds.items():
        required_entries = round_count * 2
        available_entries = min(total_entries, required_entries)
        segments[label] = "\n".join(chat_lines[:available_entries])
        actual_rounds[label] = available_entries // 2

    return segments, actual_rounds


def indent_block(text: str, indent: int = 6) -> str:
    prefix = " " * indent
    lines = text.splitlines() or [""]
    return "\n".join(f"{prefix}{line}" if line else prefix for line in lines)


def run_chat(
    model: str, messages: List[Dict[str, str]]
) -> Tuple[str, float, Dict[str, Any]]:
    start = time.perf_counter()
    response = chat(model=model, messages=messages, stream=False)
    duration = time.perf_counter() - start
    content = response.get("message", {}).get("content", "")
    return content, duration, response


def main() -> None:
    base_dir = "/home/user/ai_agent/prompts4test"
    summary_prompt_path = os.path.join(base_dir, "compressed_summary_sys_prompt.txt")
    extract_prompt_path = os.path.join(base_dir, "compressed_extract_sys_prompt.txt")
    chat_log_path = os.path.join(base_dir, "fake_chats.txt")
    output_log_path = os.path.join(base_dir, "simulation_log.txt")

    summary_prompt = read_file(summary_prompt_path)
    extract_prompt = read_file(extract_prompt_path)
    chat_text = read_file(chat_log_path)
    chat_lines = extract_chat_lines(chat_text)

    if not chat_lines:
        raise ValueError("No chat messages found in fake_chats.txt.")

    difficulties = {"easy": 10, "medium": 30, "hard": 50}
    conversation_segments, actual_rounds = build_conversation_segments(chat_lines, difficulties)

    summary_token_counts: Dict[str, int] = {}
    for label in difficulties:
        combined_text = f"{summary_prompt}\n\n{conversation_segments[label]}"
        summary_token_counts[label] = count_tokens(combined_text)

    models = [
        # "phi3:3.8b",
        # "gemma3:4b",
        "mistral:7b-instruct",
        "qwen2.5:7b-instruct",
        # "llama3.1:8b",
    ]

    with open(output_log_path, "w", encoding="utf-8") as log:
        safe_write(log, "=== Ollama Multi-Stage Simulation ===\n\n")
        safe_write(log, "Summary input tokens (system prompt + chat):\n")
        for label, requested_rounds in difficulties.items():
            used_rounds = actual_rounds[label]
            warning = ""
            if used_rounds < requested_rounds:
                warning = f" (only {used_rounds} rounds available)"
            safe_write(log, f"- {label}: {summary_token_counts[label]} tokens{warning}\n")
        safe_write(log, "\n")

        for model in models:
            print(f"Running simulations for {model}...")
            safe_write(log, f"[MODEL] {model}\n")

            load_start = time.perf_counter()
            try:
                chat(model=model, messages=[{"role": "user", "content": ""}], stream=False)
                load_duration = time.perf_counter() - load_start
                safe_write(log, f"  Load time (empty prompt warm-up): {load_duration:.2f}s\n")
            except Exception as exc:
                safe_write(log, f"  [ERROR] Warm-up failed: {exc}\n\n")
                print(f"Warm-up failed for {model}: {exc}")
                continue

            for label, requested_rounds in difficulties.items():
                used_rounds = actual_rounds[label]
                safe_write(log, f"  Difficulty: {label} ({used_rounds} rounds)\n")
                conversation_text = conversation_segments[label]

                try:
                    summary_output, summary_time, summary_response = run_chat(
                        model,
                        [
                            {"role": "system", "content": summary_prompt},
                            {"role": "user", "content": conversation_text},
                        ],
                    )
                except Exception as exc:
                    safe_write(log, f"    [ERROR] Summary step failed: {exc}\n\n")
                    print(f"Summary step failed for {model} ({label}): {exc}")
                    continue

                safe_write(log, f"    Summary time: {summary_time:.2f}s\n")
                summary_tokens = summary_token_counts[label]
                safe_write(log, f"    Summary input tokens: {summary_tokens}\n")
                safe_write(log, "    Summary output:\n")
                safe_write(log, f"{indent_block(summary_output)}\n")

                try:
                    extract_output, extract_time, extract_response = run_chat(
                        model,
                        [
                            {"role": "system", "content": extract_prompt},
                            {"role": "user", "content": summary_output},
                        ],
                    )
                except Exception as exc:
                    safe_write(log, f"    [ERROR] Extract step failed: {exc}\n\n")
                    print(f"Extract step failed for {model} ({label}): {exc}")
                    continue

                extract_input = f"{extract_prompt}\n\n{summary_output}"
                extract_tokens = count_tokens(extract_input)

                safe_write(log, f"    Extract time: {extract_time:.2f}s\n")
                safe_write(log, f"    Extract input tokens: {extract_tokens}\n")
                safe_write(log, "    Extract output:\n")
                safe_write(log, f"{indent_block(extract_output)}\n")

                total_time = summary_time + extract_time
                total_tokens = summary_tokens + extract_tokens
                safe_write(log, f"    Summary + Extract total tokens: {total_tokens}\n")
                safe_write(log, f"    Summary + Extract total time: {total_time:.2f}s\n\n")

            safe_write(log, "-" * 60 + "\n")

        safe_write(log, "\nSimulation complete.\n")

    print(f"\nAll results saved to: {output_log_path}")


if __name__ == "__main__":
    main()
