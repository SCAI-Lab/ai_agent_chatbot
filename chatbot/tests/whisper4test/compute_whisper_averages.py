#!/usr/bin/env python3
"""
Compute per-model averages from whisper benchmark summary.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple


def parse_summary_line(line: str) -> Tuple[str, str, float, float, float]:
    """Parse a data row from the summary table."""
    parts = re.split(r"\s{2,}", line.strip())
    if len(parts) != 6:
        raise ValueError(f"Unexpected column count ({len(parts)}): {line!r}")

    model, config, _difficulty, time_raw, accuracy_raw, errs_raw = parts

    time_value = float(time_raw.rstrip("s"))
    accuracy_value = float(accuracy_raw.rstrip("%"))
    errs_value = float(errs_raw)

    return model, config, time_value, accuracy_value, errs_value


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    summary_path = base_dir / "summary.md"
    output_path = base_dir / "avg.txt"

    if not summary_path.is_file():
        raise FileNotFoundError(f"Input file not found: {summary_path}")

    aggregates: Dict[Tuple[str, str], Dict[str, float]] = {}

    with summary_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line or line.startswith(("=", "-", "Model")):
                continue

            try:
                model, config, time_value, accuracy_value, errs_value = parse_summary_line(line)
            except ValueError:
                # Skip any malformed rows without stopping the whole job.
                continue

            key = (model, config)
            record = aggregates.setdefault(
                key,
                {"time_sum": 0.0, "accuracy_sum": 0.0, "errs_sum": 0.0, "count": 0},
            )
            record["time_sum"] += time_value
            record["accuracy_sum"] += accuracy_value
            record["errs_sum"] += errs_value
            record["count"] += 1

    lines = [
        "Model\tConfig\tAvg Time (s)\tAvg Accuracy (%)\tAvg Word Errs",
    ]
    for (model, config) in sorted(aggregates.keys()):
        record = aggregates[(model, config)]
        count = record["count"] or 1  # Defensive: avoid divide-by-zero.
        avg_time = record["time_sum"] / count
        avg_accuracy = record["accuracy_sum"] / count
        avg_errs = record["errs_sum"] / count
        lines.append(
            f"{model}\t{config}\t{avg_time:.2f}\t{avg_accuracy:.2f}\t{avg_errs:.2f}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
