import argparse
import json
from itertools import islice, permutations
from typing import Dict, Iterable, List, Optional

from path_nlp_parser import PathNLPParser


BIO_TAGS = (
    "O",
    "B-START",
    "I-START",
    "B-END",
    "I-END",
    "B-WAYPOINT",
    "I-WAYPOINT",
)


REPRESENTATIVE_ALIASES = {
    "red": ["红", "红色点", "赤色点"],
    "orange": ["橙", "橙色点"],
    "yellow": ["黄", "黄色点"],
    "green": ["绿", "绿色点"],
    "cyan": ["青", "青色点"],
    "blue": ["蓝", "蓝色点"],
    "purple": ["紫", "紫色点"],
}


def generate_bio_training_data(limit: Optional[int] = None) -> List[Dict[str, object]]:
    samples = list(_iter_bio_samples())
    if limit is not None:
        return samples[:limit]
    return samples


def generate_intent_training_data() -> List[Dict[str, str]]:
    samples = []
    for sample in generate_bio_training_data(limit=200):
        samples.append({"text": sample["text"], "intent": sample["intent"]})

    samples.extend(
        [
            {"text": "再经过黄色点", "intent": "add_waypoint"},
            {"text": "帮我加一个青色点作为途经点", "intent": "add_waypoint"},
            {"text": "当前路线怎么走", "intent": "query_route"},
            {"text": "查询当前路径", "intent": "query_route"},
            {"text": "今天天气怎么样", "intent": "unknown"},
            {"text": "讲个笑话", "intent": "unknown"},
            {"text": "打开音乐", "intent": "unknown"},
            {"text": "北京现在几点", "intent": "unknown"},
        ]
    )
    return samples


def _iter_bio_samples() -> Iterable[Dict[str, object]]:
    colors = list(PathNLPParser.COLOR_ALIASES.keys())
    for start, end, waypoint_a, waypoint_b in islice(permutations(colors, 4), 400):
        start_text = REPRESENTATIVE_ALIASES[start][0]
        end_text = REPRESENTATIVE_ALIASES[end][1]
        waypoint_a_text = REPRESENTATIVE_ALIASES[waypoint_a][1]
        waypoint_b_text = REPRESENTATIVE_ALIASES[waypoint_b][0]

        yield _sample(
            f"从{start_text}到{end_text}",
            "navigation",
            start_text=start_text,
            end_text=end_text,
        )
        yield _sample(
            f"我要从{start_text}去{end_text}",
            "navigation",
            start_text=start_text,
            end_text=end_text,
        )
        yield _sample(
            f"从{start_text}出发，经过{waypoint_a_text}，到{end_text}",
            "navigation",
            start_text=start_text,
            waypoint_texts=[waypoint_a_text],
            end_text=end_text,
        )
        yield _sample(
            f"从{start_text}到{end_text}，途径{waypoint_a_text}和{waypoint_b_text}",
            "navigation",
            start_text=start_text,
            waypoint_texts=[waypoint_a_text, waypoint_b_text],
            end_text=end_text,
        )
        yield _sample(
            f"{start_text}到{end_text}，先经过{waypoint_a_text}，再经过{waypoint_b_text}",
            "navigation",
            start_text=start_text,
            waypoint_texts=[waypoint_a_text, waypoint_b_text],
            end_text=end_text,
        )
        yield _sample(
            f"去{end_text}，从{start_text}出发",
            "navigation",
            start_text=start_text,
            end_text=end_text,
        )


def _sample(
    text: str,
    intent: str,
    start_text: Optional[str] = None,
    waypoint_texts: Optional[List[str]] = None,
    end_text: Optional[str] = None,
) -> Dict[str, object]:
    tags = ["O"] * len(text)
    _tag_span(text, tags, start_text, "START")
    for waypoint_text in waypoint_texts or []:
        _tag_span(text, tags, waypoint_text, "WAYPOINT")
    _tag_span(text, tags, end_text, "END")
    return {"text": text, "intent": intent, "tags": tags}


def _tag_span(text: str, tags: List[str], span_text: Optional[str], slot: str) -> None:
    if not span_text:
        return

    start = text.find(span_text)
    if start < 0:
        raise ValueError(f"Span {span_text!r} not found in {text!r}.")

    tags[start] = f"B-{slot}"
    for index in range(start + 1, start + len(span_text)):
        tags[index] = f"I-{slot}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate path NLP training data.")
    parser.add_argument("--output", default="training_data.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    samples = generate_bio_training_data(limit=args.limit)
    with open(args.output, "w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
