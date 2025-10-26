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

# 定义核心主题
CORE_TOPICS = {"basic_info", "interests", "mental_state"}


@dataclass
class TokenStats:
    """Token 级别的统计数据"""
    gt_tokens: int = 0           # Ground Truth tokens
    matched_tokens: int = 0      # 匹配的 tokens
    core_output_tokens: int = 0  # 核心主题输出 tokens
    redundant_tokens: int = 0    # 冗余 tokens (额外主题)

    @property
    def core_efficiency(self) -> float:
        """核心命中效率"""
        return self.matched_tokens / self.core_output_tokens if self.core_output_tokens else 0.0

    @property
    def total_efficiency(self) -> float:
        """总命中效率"""
        total = self.core_output_tokens + self.redundant_tokens
        return self.matched_tokens / total if total else 0.0

    @property
    def redundancy_ratio(self) -> float:
        """冗余比例"""
        total = self.core_output_tokens + self.redundant_tokens
        return self.redundant_tokens / total if total else 0.0


@dataclass
class FactStats:
    """事实级别的统计数据"""
    gt_facts: int = 0        # Ground Truth 事实数
    matched_facts: int = 0   # 匹配的事实数
    additional_items: int = 0  # 额外条目数

    @property
    def coverage(self) -> float:
        """覆盖率"""
        return self.matched_facts / self.gt_facts if self.gt_facts else 0.0


def count_tokens(text: str) -> int:
    """计算文本中的 token 数量（简单空格分割）"""
    return len(text.split())


def load_ground_truth(directory: str) -> dict[str, dict[tuple[str, str], set[str]]]:
    """加载 Ground Truth 数据"""
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
    """清洗预测文本"""
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = text.replace("；", ";").replace("，", ",").replace("、", ",")
    text = text.replace("：", ":")
    return " ".join(text.strip().split())


def split_prediction_values(text: str) -> list[str]:
    """分割预测值"""
    if not text:
        return []
    segments = [seg.strip(" .") for seg in text.split(";")]
    results: list[str] = []
    for segment in segments:
        if not segment:
            continue
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
    """生成文本变体（处理时态、单复数等）"""
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
    """标准化文本片段"""
    lowered = text.lower()
    lowered = lowered.replace("-", " ")
    lowered = re.sub(r"[^a-z0-9\s_/]", " ", lowered)
    return " ".join(lowered.split())


def matches_value(value: str, segment: str) -> bool:
    """判断预测片段是否匹配真实值（宽松匹配）"""
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
    """加载预测结果"""
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
    """评估模型性能"""
    overall_token_stats = TokenStats()
    overall_fact_stats = FactStats()
    topic_token_stats: dict[str, TokenStats] = defaultdict(TokenStats)
    topic_fact_stats: dict[str, FactStats] = defaultdict(FactStats)
    additional_by_topic: dict[str, int] = defaultdict(int)
    additional_tokens_by_topic: dict[str, int] = defaultdict(int)
    all_durations: list[float] = []

    for session_id, gt_facts in ground_truth.items():
        preds = predictions.get(session_id, {})

        # 统计 Ground Truth tokens 和 facts
        for key, values in gt_facts.items():
            topic, subtopic = key
            if topic not in CORE_TOPICS:
                continue

            fact_count = len(values)
            token_count = sum(count_tokens(v) for v in values)

            overall_fact_stats.gt_facts += fact_count
            overall_token_stats.gt_tokens += token_count
            topic_fact_stats[topic].gt_facts += fact_count
            topic_token_stats[topic].gt_tokens += token_count

        # 处理预测
        all_pred_keys = set(preds.keys())
        for key in all_pred_keys:
            topic, subtopic = key
            pred_values = preds[key]

            if topic in CORE_TOPICS:
                # 核心主题
                pred_tokens = sum(count_tokens(v) for v in pred_values)
                overall_token_stats.core_output_tokens += pred_tokens
                topic_token_stats[topic].core_output_tokens += pred_tokens

                if key in gt_facts:
                    # 有对应的 GT，计算匹配
                    gt_values = gt_facts[key]
                    matched_gt_values = set()

                    for pred_value in pred_values:
                        for gt_value in gt_values:
                            if gt_value not in matched_gt_values and matches_value(gt_value, pred_value):
                                # 匹配成功
                                matched_gt_values.add(gt_value)
                                gt_tokens = count_tokens(gt_value)

                                overall_fact_stats.matched_facts += 1
                                overall_token_stats.matched_tokens += gt_tokens
                                topic_fact_stats[topic].matched_facts += 1
                                topic_token_stats[topic].matched_tokens += gt_tokens
                                break
            else:
                # 额外主题（非核心）
                additional_items = len(pred_values)
                additional_tokens = sum(count_tokens(v) for v in pred_values)

                overall_fact_stats.additional_items += additional_items
                overall_token_stats.redundant_tokens += additional_tokens
                additional_by_topic[topic] += additional_items
                additional_tokens_by_topic[topic] += additional_tokens

        if session_id in durations:
            all_durations.append(durations[session_id])

    # 输出报告
    print("=" * 65)
    print(f"MODEL: {name}")
    print("=" * 65)
    print()

    # 核心主题性能
    print("┌" + "─" * 63 + "┐")
    print("│ CORE TOPIC PERFORMANCE" + " " * 40 + "│")
    print("└" + "─" * 63 + "┘")
    print(f"  Ground Truth Facts:    {overall_fact_stats.gt_facts}")
    print(f"  Matched Facts:         {overall_fact_stats.matched_facts}")
    print(f"  Coverage:              {overall_fact_stats.coverage:.1%} {'✅' if overall_fact_stats.coverage >= 0.7 else '⚠️'}")
    print()
    print(f"  Ground Truth Tokens:   {overall_token_stats.gt_tokens}")
    print(f"  Core Output Tokens:    {overall_token_stats.core_output_tokens}")
    print(f"  Matched Tokens:        {overall_token_stats.matched_tokens}")
    print(f"  Core Hit Efficiency:   {overall_token_stats.core_efficiency:.1%} {'✅' if overall_token_stats.core_efficiency >= 0.5 else '⚠️'}")
    print()
    print("  Per-topic Breakdown:")
    for topic in sorted(CORE_TOPICS):
        stats_t = topic_token_stats[topic]
        facts_t = topic_fact_stats[topic]
        if facts_t.gt_facts == 0:
            continue
        emoji = "✨" if facts_t.coverage >= 0.8 else "⚠️" if facts_t.coverage >= 0.5 else "❌"
        print(f"    {topic}:")
        print(f"      Facts:    {facts_t.matched_facts}/{facts_t.gt_facts} ({facts_t.coverage:.0%})  {emoji}")
        print(f"      Tokens:   {stats_t.matched_tokens}/{stats_t.core_output_tokens} ({stats_t.core_efficiency:.0%})  {emoji}")
    print()

    # 额外输出分析
    print("┌" + "─" * 63 + "┐")
    print("│ ADDITIONAL OUTPUT ANALYSIS" + " " * 36 + "│")
    print("└" + "─" * 63 + "┘")
    print(f"  Additional Items:      {overall_fact_stats.additional_items}")
    print(f"  Redundant Tokens:      {overall_token_stats.redundant_tokens}")
    print()
    if additional_by_topic:
        print("  Breakdown by topic:")
        for topic in sorted(additional_by_topic.keys()):
            items = additional_by_topic[topic]
            tokens = additional_tokens_by_topic[topic]
            print(f"    {topic}: {items} items, {tokens} tokens")
    print()

    # 总体效率
    print("┌" + "─" * 63 + "┐")
    print("│ OVERALL EFFICIENCY" + " " * 44 + "│")
    print("└" + "─" * 63 + "┘")
    total_output = overall_token_stats.core_output_tokens + overall_token_stats.redundant_tokens
    print(f"  Total Output Tokens:   {total_output} ({overall_token_stats.core_output_tokens} core + {overall_token_stats.redundant_tokens} redundant)")
    print(f"  Matched GT Tokens:     {overall_token_stats.matched_tokens}")
    print(f"  Total Hit Efficiency:  {overall_token_stats.total_efficiency:.1%}")
    print(f"  Redundancy Ratio:      {overall_token_stats.redundancy_ratio:.1%}")
    print()

    # 时间性能
    if all_durations:
        total_time = sum(all_durations)
        print("┌" + "─" * 63 + "┐")
        print("│ TIME PERFORMANCE" + " " * 46 + "│")
        print("└" + "─" * 63 + "┘")
        print(f"  Total Time:   {total_time:.2f}s")
        print()

    # 总结
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)

    # 评分
    coverage_score = overall_fact_stats.coverage
    core_eff_score = overall_token_stats.core_efficiency
    total_eff_score = overall_token_stats.total_efficiency

    print("  ✅ Strengths:")
    if coverage_score >= 0.7:
        print(f"    - Good fact coverage ({coverage_score:.1%})")
    if core_eff_score >= 0.5:
        print(f"    - Good core efficiency ({core_eff_score:.1%})")
    if overall_fact_stats.additional_items > 0:
        print(f"    - Rich information extraction ({overall_fact_stats.additional_items} additional items)")

    print()
    print("  ⚠️  Areas for Improvement:")
    if coverage_score < 0.7:
        print(f"    - Low fact coverage ({coverage_score:.1%}) - target: ≥70%")
    if core_eff_score < 0.5:
        print(f"    - Low core efficiency ({core_eff_score:.1%}) - target: ≥50%")
    if total_eff_score < 0.3:
        print(f"    - Low total efficiency ({total_eff_score:.1%}) - reduce redundant output")

    # 主题级别建议
    for topic in sorted(CORE_TOPICS):
        facts_t = topic_fact_stats[topic]
        if facts_t.gt_facts > 0 and facts_t.coverage < 0.6:
            print(f"    - Improve {topic} extraction (coverage: {facts_t.coverage:.1%})")

    print()
    overall_grade = int((coverage_score * 40 + core_eff_score * 30 + total_eff_score * 30) * 100)
    print(f"  Overall Grade: {overall_grade}/100")
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
