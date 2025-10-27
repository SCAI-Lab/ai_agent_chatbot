import argparse
import math
import os
import re
import statistics
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple


SessionId = str
Topic = str
Field = str
SlotKey = Tuple[Topic, Field]


@dataclass
class SlotMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    exact_match: int = 0
    count: int = 0

    def update(self, tp: int, fp: int, fn: int, exact: bool) -> None:
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.count += 1
        if exact:
            self.exact_match += 1

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
        if not p and not r:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def exact_rate(self) -> float:
        return self.exact_match / self.count if self.count else 0.0


@dataclass
class TokenMetrics:
    matched: int = 0
    pred_total: int = 0
    gt_total: int = 0

    @property
    def precision(self) -> float:
        return self.matched / self.pred_total if self.pred_total else 0.0

    @property
    def recall(self) -> float:
        return self.matched / self.gt_total if self.gt_total else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        if not p and not r:
            return 0.0
        return 2 * p * r / (p + r)


@dataclass
class SlotEntry:
    values: List[str]
    canonical: Set[str]

    def add(self, value: str) -> None:
        stripped = value.strip()
        if not stripped:
            return
        normalized = canonicalize(stripped)
        if normalized not in self.canonical:
            self.values.append(stripped)
            if normalized:
                self.canonical.add(normalized)

    @property
    def text(self) -> str:
        return ", ".join(self.values)


@dataclass
class SlotRecord:
    session: SessionId
    topic: Topic
    field: Field
    gt_values: List[str]
    pred_values: List[str]
    gt_text: str
    pred_text: str


def read_ground_truth(directory: str) -> Dict[SessionId, Dict[SlotKey, SlotEntry]]:
    sessions: Dict[SessionId, Dict[SlotKey, SlotEntry]] = {}
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".txt"):
            continue
        session_id = os.path.splitext(filename)[0]
        slots: Dict[SlotKey, SlotEntry] = OrderedDict()
        path = os.path.join(directory, filename)
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or "::" not in line:
                    continue
                parts = line.split("::", 2)
                if len(parts) != 3:
                    continue
                topic = parts[0].strip()
                field = parts[1].strip()
                value_text = parts[2].strip()
                values = split_values(value_text)
                key = (topic, field)
                entry = slots.get(key)
                if entry is None:
                    entry = SlotEntry(values=[], canonical=set())
                    slots[key] = entry
                if values:
                    for value in values:
                        entry.add(value)
                else:
                    slots.setdefault(key, entry)
        sessions[session_id] = slots
    return sessions


FILE_PATTERN = re.compile(r"File:\s.*?/(\d+)\.json")
COST_PATTERN = re.compile(r"Cost time\(s\)\s+([0-9.]+)")


def read_predictions(
    path: str,
) -> Tuple[Dict[SessionId, Dict[SlotKey, SlotEntry]], Dict[SessionId, float]]:
    sessions: Dict[SessionId, Dict[SlotKey, SlotEntry]] = OrderedDict()
    durations: Dict[SessionId, float] = {}
    current_session: Optional[SessionId] = None

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            file_match = FILE_PATTERN.match(line)
            if file_match:
                current_session = file_match.group(1)
                if current_session not in sessions:
                    sessions[current_session] = OrderedDict()
                continue

            if current_session is None:
                continue

            cost_match = COST_PATTERN.match(line)
            if cost_match:
                try:
                    durations[current_session] = float(cost_match.group(1))
                except ValueError:
                    pass
                continue

            if "::" not in line:
                continue

            parts = line.split("::", 2)
            if len(parts) != 3:
                continue
            topic = parts[0].strip()
            field = parts[1].strip()
            value = parts[2].strip()

            key = (topic, field)
            session_slots = sessions.setdefault(current_session, OrderedDict())
            entry = session_slots.get(key)
            if entry is None:
                entry = SlotEntry(values=[], canonical=set())
                session_slots[key] = entry
            entry.add(value)

    return sessions, durations


VALUE_DELIM_RE = re.compile(r"\band\b", re.IGNORECASE)


def split_values(raw: str) -> List[str]:
    if not raw:
        return []
    pieces: List[str] = []
    buffer: List[str] = []
    depth = 0
    for char in raw:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        if char in {",", ";"} and depth == 0:
            segment = "".join(buffer).strip()
            if segment:
                pieces.append(segment)
            buffer = []
            continue
        buffer.append(char)
    tail = "".join(buffer).strip()
    if tail:
        pieces.append(tail)

    expanded: List[str] = []
    for piece in pieces:
        stripped = piece.strip(" .")
        if not stripped:
            continue
        if "(" in stripped and ")" in stripped:
            expanded.append(stripped)
            continue
        splits = [seg.strip(" .") for seg in VALUE_DELIM_RE.split(stripped) if seg.strip()]
        if splits:
            expanded.extend(splits)
        else:
            expanded.append(stripped)

    return [segment for segment in expanded if segment]


NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
PUNCT_RE = re.compile(r"[^\w\s/]")


def canonicalize(value: str) -> str:
    text = NON_ASCII_RE.sub("", value)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = PUNCT_RE.sub(" ", text)
    return text.lower().strip()


TOKEN_SPLIT_RE = re.compile(r"[^\w]+")


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens = [tok for tok in TOKEN_SPLIT_RE.split(text.lower()) if tok]
    return tokens


def bleu1_score(pred_tokens: Sequence[str], ref_tokens: Sequence[str]) -> float:
    if not pred_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    precision = overlap / sum(pred_counts.values()) if pred_counts else 0.0
    if not ref_tokens:
        return 0.0
    bp = 1.0
    ref_len = len(ref_tokens)
    pred_len = len(pred_tokens)
    if pred_len <= ref_len:
        bp = math.exp(1 - ref_len / pred_len) if pred_len else 0.0
    return precision * bp


def evaluate_model(
    name: str,
    predictions: Dict[SessionId, Dict[SlotKey, SlotEntry]],
    durations: Dict[SessionId, float],
    ground_truth: Dict[SessionId, Dict[SlotKey, SlotEntry]],
) -> None:
    slot_records: List[SlotRecord] = []
    metrics = SlotMetrics()
    per_topic: Dict[Topic, SlotMetrics] = defaultdict(SlotMetrics)
    token_metrics = TokenMetrics()
    bleu_scores: List[float] = []
    duration_values: List[float] = []
    extra_slot_counts: Counter[str] = Counter()
    extra_value_counts: Counter[str] = Counter()

    session_ids = sorted(set(ground_truth.keys()) | set(predictions.keys()))
    for session in session_ids:
        gt_slots = ground_truth.get(session, {})
        pred_slots = predictions.get(session, {})
        for (topic, field), gt_entry in gt_slots.items():
            pred_entry = pred_slots.get((topic, field))
            pred_has_values = bool(pred_entry and pred_entry.values)
            if pred_has_values:
                metrics.update(1, 0, 0, True)
                per_topic[topic].update(1, 0, 0, True)
            else:
                metrics.update(0, 0, 1, False)
                per_topic[topic].update(0, 0, 1, False)

            gt_text = gt_entry.text
            pred_text = pred_entry.text if pred_entry else ""

            gt_tokens = tokenize(gt_text)
            pred_tokens = tokenize(pred_text)

            token_metrics.matched += sum((Counter(gt_tokens) & Counter(pred_tokens)).values())
            token_metrics.gt_total += len(gt_tokens)
            token_metrics.pred_total += len(pred_tokens)

            if gt_tokens or pred_tokens:
                bleu_scores.append(bleu1_score(pred_tokens, gt_tokens))

            slot_records.append(
                SlotRecord(
                    session=session,
                    topic=topic,
                    field=field,
                    gt_values=list(gt_entry.values),
                    pred_values=list(pred_entry.values if pred_entry else []),
                    gt_text=gt_text,
                    pred_text=pred_text,
                )
            )

        for (topic, field), pred_entry in pred_slots.items():
            if (topic, field) not in gt_slots and pred_entry.values:
                extra_slot_counts[topic] += 1
                extra_value_counts[topic] += len(pred_entry.values)
                metrics.fp += 1
                per_topic[topic].fp += 1

        if session in durations:
            duration_values.append(durations[session])

    overall_bleu = statistics.mean(bleu_scores) if bleu_scores else 0.0

    print("=" * 72)
    print(f"MODEL: {name}")
    print("=" * 72)
    print(f"Slots evaluated: {metrics.count}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall:    {metrics.recall:.3f}")
    print(f"F1 Score:  {metrics.f1:.3f}")
    print(f"Exact-Match Coverage: {metrics.exact_rate:.3f}")
    print()
    print("Token-level metrics:")
    print(f"  Token Precision: {token_metrics.precision:.3f}")
    print(f"  Token Recall:    {token_metrics.recall:.3f}")
    print(f"  Token F1:        {token_metrics.f1:.3f}")
    print(f"  BLEU-1 (macro):  {overall_bleu:.3f}")
    print()

    print("Per-topic breakdown:")
    for topic, topic_metrics in sorted(per_topic.items()):
        if topic_metrics.count == 0:
            continue
        print(f"  {topic:20s} P={topic_metrics.precision:.3f} R={topic_metrics.recall:.3f} F1={topic_metrics.f1:.3f} EM={topic_metrics.exact_rate:.3f}")
    print()

    if duration_values:
        duration_values.sort()
        total = sum(duration_values)
        median = statistics.median(duration_values)
        p95_index = max(int(len(duration_values) * 0.95) - 1, 0)
        p95 = duration_values[p95_index]
        print("Latency (seconds):")
        print(f"  Mean:   {total / len(duration_values):.3f}")
        print(f"  Median: {median:.3f}")
        print(f"  P95:    {p95:.3f}")
        print(f"  Total:  {total:.3f}")
        print()

    mismatches = []
    for rec in slot_records:
        gt_norm = {norm for v in rec.gt_values if (norm := canonicalize(v))}
        pred_norm = {norm for v in rec.pred_values if (norm := canonicalize(v))}
        if gt_norm != pred_norm:
            mismatches.append(rec)
    if mismatches:
        print("Mismatched slots:")
        for rec in mismatches:
            print(f"  session {rec.session} :: {rec.topic}::{rec.field}")
            print(f"    GT : {rec.gt_text or '(empty)'}")
            print(f"    Pred: {rec.pred_text or '(empty)'}")
    print()

    extra_total_slots = sum(extra_slot_counts.values())
    extra_total_values = sum(extra_value_counts.values())
    if extra_total_slots or extra_total_values:
        print("Additional slots without GT reference:")
        print(f"  Slot count:  {extra_total_slots}")
        print(f"  Value count: {extra_total_values}")
        for topic in sorted(extra_slot_counts):
            print(
                f"    {topic}: {extra_slot_counts[topic]} slot(s), {extra_value_counts[topic]} value(s)"
            )
        print()


def collect_output_files(directory: str, only: Optional[Sequence[str]]) -> List[str]:
    files = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".txt"):
            continue
        if only and filename not in only and os.path.join(directory, filename) not in only:
            continue
        files.append(os.path.join(directory, filename))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extracted memories against ground truth.")
    parser.add_argument(
        "--ground-truth",
        default=os.path.join("chats", "ground_truth"),
        help="Directory with ground truth .txt files.",
    )
    parser.add_argument(
        "--outputs",
        default=os.path.join("chats", "clean_output"),
        help="Directory containing model output .txt files.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional list of output filenames to evaluate.",
    )
    args = parser.parse_args()

    ground_truth = read_ground_truth(args.ground_truth)
    files = collect_output_files(args.outputs, args.only)
    if not files:
        raise SystemExit("No output files found.")

    for path in files:
        name = os.path.basename(path)
        predictions, durations = read_predictions(path)
        evaluate_model(
            name=name,
            predictions=predictions,
            durations=durations,
            ground_truth=ground_truth,
        )


if __name__ == "__main__":
    main()
