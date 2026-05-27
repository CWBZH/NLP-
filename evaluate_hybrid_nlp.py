import json
from typing import Dict, List

from hybrid_path_nlp import HybridPathNLP


EVALUATION_SAMPLES: List[Dict[str, object]] = [
    _ for _ in [
        {
            "text": "从蓝色点到绿色点",
            "intent": "navigation",
            "start": "blue",
            "waypoints": [],
            "end": "green",
            "is_complete": True,
            "missing_slots": [],
        },
        {
            "text": "我要从蓝点去绿点",
            "intent": "navigation",
            "start": "blue",
            "waypoints": [],
            "end": "green",
            "is_complete": True,
            "missing_slots": [],
        },
        {
            "text": "蓝到绿",
            "intent": "navigation",
            "start": "blue",
            "waypoints": [],
            "end": "green",
            "is_complete": True,
            "missing_slots": [],
        },
        {
            "text": "从紫色点到绿色点",
            "intent": "navigation",
            "start": "purple",
            "waypoints": [],
            "end": "green",
            "is_complete": True,
            "missing_slots": [],
        },
        {
            "text": "帮我规划从橙点到蓝点的路线",
            "intent": "navigation",
            "start": "orange",
            "waypoints": [],
            "end": "blue",
            "is_complete": True,
            "missing_slots": [],
        },
        {
            "text": "从蓝点出发，经过青点，最后到绿点",
            "intent": "navigation",
            "start": "blue",
            "waypoints": ["cyan"],
            "end": "green",
            "is_complete": True,
            "missing_slots": [],
        },
        {
            "text": "从黄色点出发，经过紫色点，到红色点",
            "intent": "navigation",
            "start": "yellow",
            "waypoints": ["purple"],
            "end": "red",
            "is_complete": True,
            "missing_slots": [],
        },
        {
            "text": "从绿点经过蓝点到紫点",
            "intent": "navigation",
            "start": "green",
            "waypoints": ["blue"],
            "end": "purple",
            "is_complete": True,
            "missing_slots": [],
        },
        {
            "text": "从赤色点出发，途经橙色点，去青色点",
            "intent": "navigation",
            "start": "red",
            "waypoints": ["orange"],
            "end": "cyan",
            "is_complete": True,
            "missing_slots": [],
        },
        {
            "text": "从蓝到红，途径黄和紫",
            "intent": "navigation",
            "start": "blue",
            "waypoints": ["yellow", "purple"],
            "end": "red",
            "is_complete": True,
            "missing_slots": [],
        },
        {
            "text": "蓝色点到红色点，先经过橙色点，再经过黄色点",
            "intent": "navigation",
            "start": "blue",
            "waypoints": ["orange", "yellow"],
            "end": "red",
            "is_complete": True,
            "missing_slots": [],
        },
        {
            "text": "从紫到绿，经过青和蓝",
            "intent": "navigation",
            "start": "purple",
            "waypoints": ["cyan", "blue"],
            "end": "green",
            "is_complete": True,
            "missing_slots": [],
        },
        {
            "text": "从红点到蓝点，途经橙点和黄点",
            "intent": "navigation",
            "start": "red",
            "waypoints": ["orange", "yellow"],
            "end": "blue",
            "is_complete": True,
            "missing_slots": [],
        },
        {
            "text": "去绿色点，从蓝色点出发",
            "intent": "navigation",
            "start": "blue",
            "waypoints": [],
            "end": "green",
            "is_complete": True,
            "missing_slots": [],
        },
        {
            "text": "到紫点，从青点出发",
            "intent": "navigation",
            "start": "cyan",
            "waypoints": [],
            "end": "purple",
            "is_complete": True,
            "missing_slots": [],
        },
        {
            "text": "去红色点，从赤点出发",
            "intent": "navigation",
            "start": "red",
            "waypoints": [],
            "end": "red",
            "is_complete": True,
            "missing_slots": [],
        },
        {
            "text": "帮我规划到绿色点",
            "intent": "navigation",
            "start": None,
            "waypoints": [],
            "end": "green",
            "is_complete": False,
            "missing_slots": ["start"],
        },
        {
            "text": "到黄色点",
            "intent": "navigation",
            "start": None,
            "waypoints": [],
            "end": "yellow",
            "is_complete": False,
            "missing_slots": ["start"],
        },
        {
            "text": "去青色点",
            "intent": "navigation",
            "start": None,
            "waypoints": [],
            "end": "cyan",
            "is_complete": False,
            "missing_slots": ["start"],
        },
        {
            "text": "从蓝色点出发",
            "intent": "navigation",
            "start": "blue",
            "waypoints": [],
            "end": None,
            "is_complete": False,
            "missing_slots": ["end"],
        },
        {
            "text": "从赤色点出发",
            "intent": "navigation",
            "start": "red",
            "waypoints": [],
            "end": None,
            "is_complete": False,
            "missing_slots": ["end"],
        },
        {
            "text": "我要从紫点出发",
            "intent": "navigation",
            "start": "purple",
            "waypoints": [],
            "end": None,
            "is_complete": False,
            "missing_slots": ["end"],
        },
        {
            "text": "从赤点到青点",
            "intent": "navigation",
            "start": "red",
            "waypoints": [],
            "end": "cyan",
            "is_complete": True,
            "missing_slots": [],
        },
        {
            "text": "从红色点到赤色点",
            "intent": "navigation",
            "start": "red",
            "waypoints": [],
            "end": "red",
            "is_complete": True,
            "missing_slots": [],
        },
        {
            "text": "从青点到蓝点",
            "intent": "navigation",
            "start": "cyan",
            "waypoints": [],
            "end": "blue",
            "is_complete": True,
            "missing_slots": [],
        },
        {
            "text": "从蓝点到青点",
            "intent": "navigation",
            "start": "blue",
            "waypoints": [],
            "end": "cyan",
            "is_complete": True,
            "missing_slots": [],
        },
        {
            "text": "先去青色点，再到紫色点，最后到红色点",
            "intent": "navigation",
            "start": None,
            "waypoints": ["cyan", "purple"],
            "end": "red",
            "is_complete": False,
            "missing_slots": ["start"],
        },
        {
            "text": "先经过橙色点，再经过黄色点，最后到紫色点",
            "intent": "navigation",
            "start": None,
            "waypoints": ["orange", "yellow"],
            "end": "purple",
            "is_complete": False,
            "missing_slots": ["start"],
        },
        {
            "text": "今天天气怎么样",
            "intent": "unknown",
            "start": None,
            "waypoints": [],
            "end": None,
            "is_complete": False,
            "missing_slots": [],
        },
        {
            "text": "讲个笑话",
            "intent": "unknown",
            "start": None,
            "waypoints": [],
            "end": None,
            "is_complete": False,
            "missing_slots": [],
        },
        {
            "text": "北京现在几点",
            "intent": "unknown",
            "start": None,
            "waypoints": [],
            "end": None,
            "is_complete": False,
            "missing_slots": [],
        },
        {
            "text": "打开音乐",
            "intent": "unknown",
            "start": None,
            "waypoints": [],
            "end": None,
            "is_complete": False,
            "missing_slots": [],
        },
    ]
]


def evaluate(parser: HybridPathNLP, samples: List[Dict[str, object]]) -> Dict[str, object]:
    counters = {
        "intent": 0,
        "start": 0,
        "end": 0,
        "waypoints": 0,
        "is_complete": 0,
        "missing_slots": 0,
        "semantic_frame": 0,
        "fallback": 0,
    }
    errors = []

    for sample in samples:
        debug = _parse_with_debug(parser, sample["text"])
        predicted = debug["result"]
        if debug["used_fallback"]:
            counters["fallback"] += 1

        error_fields = []
        for field, counter_name in [
            ("intent", "intent"),
            ("start", "start"),
            ("end", "end"),
            ("waypoints", "waypoints"),
            ("is_complete", "is_complete"),
            ("missing_slots", "missing_slots"),
        ]:
            if predicted[field] == sample[field]:
                counters[counter_name] += 1
            else:
                error_fields.append(field)

        if (
            predicted["start"] == sample["start"]
            and predicted["waypoints"] == sample["waypoints"]
            and predicted["end"] == sample["end"]
        ):
            counters["semantic_frame"] += 1

        if error_fields:
            errors.append(
                {
                    "text": sample["text"],
                    "expected": _expected_result(sample),
                    "predicted": predicted,
                    "error_fields": error_fields,
                    "fallback_reason": debug.get("fallback_reason"),
                }
            )

    total = len(samples)
    return {
        "total_samples": total,
        "intent_accuracy": counters["intent"] / total,
        "start_accuracy": counters["start"] / total,
        "end_accuracy": counters["end"] / total,
        "waypoint_exact_match_accuracy": counters["waypoints"] / total,
        "is_complete_accuracy": counters["is_complete"] / total,
        "missing_slots_accuracy": counters["missing_slots"] / total,
        "semantic_frame_accuracy": counters["semantic_frame"] / total,
        "fallback_count": counters["fallback"],
        "fallback_rate": counters["fallback"] / total,
        "errors": errors,
    }


def print_report(metrics: Dict[str, object]) -> None:
    total = metrics["total_samples"]
    print(f"Total samples: {total}")
    print(f"Intent accuracy: {metrics['intent_accuracy']:.4f}")
    print(f"Start accuracy: {metrics['start_accuracy']:.4f}")
    print(f"End accuracy: {metrics['end_accuracy']:.4f}")
    print(f"Waypoint exact match accuracy: {metrics['waypoint_exact_match_accuracy']:.4f}")
    print(f"Is complete accuracy: {metrics['is_complete_accuracy']:.4f}")
    print(f"Missing slots accuracy: {metrics['missing_slots_accuracy']:.4f}")
    print(f"Semantic frame accuracy: {metrics['semantic_frame_accuracy']:.4f}")
    print(f"Fallback used: {metrics['fallback_count']} / {total}")
    print(f"Fallback rate: {metrics['fallback_rate']:.4f}")

    errors = metrics["errors"]
    if not errors:
        print("No prediction errors.")
        return

    print("\nErrors:")
    for index, error in enumerate(errors, start=1):
        print(f"\n[{index}] text: {error['text']}")
        print("expected:", json.dumps(error["expected"], ensure_ascii=False))
        print("predicted:", json.dumps(error["predicted"], ensure_ascii=False))
        print("error fields:", ", ".join(error["error_fields"]))
        if error["fallback_reason"]:
            print("fallback reason:", error["fallback_reason"])


def _parse_with_debug(parser: HybridPathNLP, text: str) -> Dict[str, object]:
    parse_with_debug = getattr(parser, "parse_with_debug", None)
    if callable(parse_with_debug):
        return parse_with_debug(text)
    return {"result": parser.parse(text), "used_fallback": False, "slot_source": "unknown"}


def _expected_result(sample: Dict[str, object]) -> Dict[str, object]:
    return {
        "intent": sample["intent"],
        "start": sample["start"],
        "waypoints": sample["waypoints"],
        "end": sample["end"],
        "is_complete": sample["is_complete"],
        "missing_slots": sample["missing_slots"],
    }


def main() -> None:
    parser = HybridPathNLP()
    print_report(evaluate(parser, EVALUATION_SAMPLES))


if __name__ == "__main__":
    main()
