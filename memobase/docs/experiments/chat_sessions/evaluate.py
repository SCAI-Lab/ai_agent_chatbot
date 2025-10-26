import argparse
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass

BASE_DIR = os.path.dirname(__file__)
CHATS_DIR = os.path.join(BASE_DIR, "chats")
GROUND_TRUTH_DIR = os.path.join(CHATS_DIR, "ground_truth")
OUTPUT_DIR = os.path.join(CHATS_DIR, "output")


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, tp: int, fp: int, fn: int) -> None:
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def merge(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0


def load_ground_truth(directory: str) -> dict[str, dict[tuple[str, str], set[str]]]:
    sessions: dict[str, dict[tuple[str, str], set[str]]] = {}
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".txt"):
            continue
        session_id = filename.split(".")[0]
        facts: dict[tuple[str, str], set[str]] = defaultdict(set)
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or "::" not in line:
                    continue
                try:
                    topic, subtopic, values = line.split("::", 2)
                except ValueError:
                    continue
                pieces = [v.strip() for v in values.split(",") if v.strip()]
                for piece in pieces:
                    facts[(topic.strip(), subtopic.strip())].add(piece)
        sessions[session_id] = facts
    return sessions


def clean_prediction_text(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = text.replace("；", ";").replace("，", ",").replace("、", ",")
    text = text.replace("：", ":")
    return " ".join(text.strip().split())


def split_prediction_values(text: str) -> list[str]:
    if not text:
        return []
    segments = [seg.strip(" .") for seg in text.split(";")]
    results: list[str] = []
    for segment in segments:
        if not segment:
            continue
        # Keep both the full segment and optional comma-based splits to catch multi-fact bullets.
        results.append(segment)
        extra_parts = [
            part.strip(" .")
            for part in re.split(r",|\band\b", segment)
            if part.strip()
        ]
        if len(extra_parts) > 1:
            results.extend(extra_parts)
    return results


def generate_variants(value: str) -> set[str]:
    value = value.lower().strip()
    value = value.replace("-", " ")
    variants = {value}
    if "_" in value:
        variants.add(value.replace("_", " "))
    if " " in value:
        variants.add(value.replace(" ", ""))
    token = value.replace("_", " ")
    single_token = token if " " not in token else None
    if single_token:
        stem_candidates = {single_token}
        if single_token.endswith("ing") and len(single_token) > 3:
            stem = single_token[:-3]
            stem_candidates.add(stem)
            stem_candidates.add(stem + "e")
        if single_token.endswith("ies") and len(single_token) > 3:
            stem_candidates.add(single_token[:-3] + "y")
        if single_token.endswith("s") and len(single_token) > 1:
            stem_candidates.add(single_token[:-1])
        stem_candidates.add(single_token + "s")
        stem_candidates.add(single_token + "es")
        for stem in stem_candidates:
            variants.add(stem)
            variants.add(stem.replace("_", " "))
    return {v for v in variants if v}


def normalize_segment(text: str) -> str:
    lowered = text.lower()
    lowered = lowered.replace("-", " ")
    lowered = re.sub(r"[^a-z0-9\s_/]", " ", lowered)
    return " ".join(lowered.split())


def matches_value(value: str, segment: str) -> bool:
    variants = generate_variants(value)
    normalized_segment = normalize_segment(segment)
    if not normalized_segment:
        return False
    for variant in variants:
        if variant in normalized_segment:
            return True
    seg_tokens = set(normalized_segment.split())
    for variant in variants:
        variant_tokens = set(variant.split())
        if variant_tokens and variant_tokens <= seg_tokens:
            return True
        if variant_tokens:
            overlap = len(seg_tokens & variant_tokens)
            if overlap and overlap / len(variant_tokens) >= 0.6:
                return True
    return False


def load_predictions(path: str) -> tuple[
    dict[str, dict[tuple[str, str], list[str]]], dict[str, float]
]:
    sessions: dict[str, dict[tuple[str, str], list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    durations: dict[str, float] = {}
    current_session: str | None = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            file_match = re.match(r"File:\s.*?/(\d+)\.json", line)
            if file_match:
                current_session = file_match.group(1)
                continue
            if current_session is None:
                continue
            if line.startswith("Cost time(s)"):
                try:
                    durations[current_session] = float(line.split()[-1])
                except (ValueError, IndexError):
                    pass
                continue
            if not line.startswith("*"):
                continue

            content = line.lstrip("*").strip()
            if ":" not in content or "-" not in content:
                continue
            topic_part, rest = content.split(":", 1)
            if "-" not in rest:
                continue
            subtopic_part, value_part = rest.split("-", 1)
            topic = topic_part.strip()
            subtopic = subtopic_part.strip()
            values_text = clean_prediction_text(value_part)
            for value in split_prediction_values(values_text):
                if value:
                    sessions[current_session][(topic, subtopic)].append(value)

    return sessions, durations


def evaluate_model(
    name: str,
    predictions: dict[str, dict[tuple[str, str], list[str]]],
    durations: dict[str, float],
    ground_truth: dict[str, dict[tuple[str, str], set[str]]],
) -> None:
    overall = Counts()
    topic_stats: dict[str, Counts] = defaultdict(Counts)
    session_stats: dict[str, Counts] = defaultdict(Counts)
    all_durations: list[float] = []

    for session_id, gt_facts in ground_truth.items():
        preds = predictions.get(session_id, {})
        session_count = Counts()
        keys = set(gt_facts.keys()) | set(preds.keys())
        for key in keys:
            topic, subtopic = key
            gt_values = set(gt_facts.get(key, set()))
            pred_segments = preds.get(key, [])

            unmatched_gt = set(gt_values)
            tp = 0
            predicted_facts = 0

            for segment in pred_segments:
                matched_values = [
                    value
                    for value in list(unmatched_gt)
                    if matches_value(value, segment)
                ]
                if matched_values:
                    tp += len(matched_values)
                    predicted_facts += len(matched_values)
                    for value in matched_values:
                        unmatched_gt.discard(value)
                else:
                    predicted_facts += 1

            fn = len(unmatched_gt)
            fp = predicted_facts - tp

            session_count.update(tp, fp, fn)
            topic_stats[topic].update(tp, fp, fn)

        overall.merge(session_count)
        session_stats[session_id].merge(session_count)

        if session_id in durations:
            all_durations.append(durations[session_id])

    print(f"Model: {name}")
    print(
        f"  Overall -> precision: {overall.precision:.3f}, "
        f"recall: {overall.recall:.3f}, F1: {overall.f1:.3f} "
        f"(TP: {overall.tp}, FP: {overall.fp}, FN: {overall.fn})"
    )
    if overall.tp + overall.fp:
        redundancy = overall.fp / (overall.tp + overall.fp)
        print(f"  Redundancy rate: {redundancy:.3f}")
    if all_durations:
        mean = sum(all_durations) / len(all_durations)
        variance = sum((d - mean) ** 2 for d in all_durations) / len(all_durations)
        std = math.sqrt(variance)
        print(
            f"  Time -> total: {sum(all_durations):.2f}s, "
            f"mean: {mean:.2f}s, std: {std:.2f}s"
        )
    print("  Per topic:")
    for topic, counts in sorted(topic_stats.items()):
        print(
            f"    {topic}: P={counts.precision:.3f}, "
            f"R={counts.recall:.3f}, F1={counts.f1:.3f} "
            f"(TP={counts.tp}, FP={counts.fp}, FN={counts.fn})"
        )
    print("  Per session:")
    for session_id, counts in sorted(session_stats.items()):
        print(
            f"    {session_id}: P={counts.precision:.3f}, "
            f"R={counts.recall:.3f}, F1={counts.f1:.3f} "
            f"(TP={counts.tp}, FP={counts.fp}, FN={counts.fn})"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model outputs against ground truth.")
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory containing model output .txt files.",
    )
    parser.add_argument(
        "--ground-truth-dir",
        default=GROUND_TRUTH_DIR,
        help="Directory containing ground truth .txt files.",
    )
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth_dir)
    if not ground_truth:
        raise ValueError(f"No ground truth files found in {args.ground_truth_dir}")

    output_files = [
        f for f in sorted(os.listdir(args.output_dir)) if f.endswith(".txt")
    ]
    if not output_files:
        raise ValueError(f"No output files found in {args.output_dir}")

    for filename in output_files:
        model_name = os.path.splitext(filename)[0]
        predictions, durations = load_predictions(os.path.join(args.output_dir, filename))
        evaluate_model(model_name, predictions, durations, ground_truth)


if __name__ == "__main__":
    main()
