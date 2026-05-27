from typing import Dict, List, Optional, Tuple

from color_normalizer import ColorNormalizer
from intent_classifier import IntentClassifier
from path_nlp_parser import PathNLPParser
from slot_tagger import BiLSTMCRFSlotTagger


class HybridPathNLP:
    """Model-first path parser with rule-based fallback."""

    def __init__(
        self,
        intent_classifier: Optional[IntentClassifier] = None,
        slot_tagger: Optional[BiLSTMCRFSlotTagger] = None,
        fallback_parser: Optional[PathNLPParser] = None,
        color_normalizer: Optional[ColorNormalizer] = None,
    ):
        self.intent_classifier = intent_classifier or IntentClassifier()
        self.slot_tagger = slot_tagger or BiLSTMCRFSlotTagger()
        self.fallback_parser = fallback_parser or PathNLPParser()
        self.color_normalizer = color_normalizer or ColorNormalizer()

    def parse(self, text: str) -> Dict[str, object]:
        return self.parse_with_debug(text)["result"]

    def parse_with_debug(self, text: str) -> Dict[str, object]:
        intent = self.intent_classifier.predict(text)
        if intent == "unknown":
            return {
                "result": self._unknown_result(),
                "used_fallback": False,
                "fallback_reason": None,
                "slot_source": "none",
            }

        if not self._is_navigation_like(intent):
            return self._fallback_result(text, "unsupported_intent")

        model_result, fallback_reason = self._parse_with_slot_model(text)
        if model_result is not None:
            return {
                "result": model_result,
                "used_fallback": False,
                "fallback_reason": None,
                "slot_source": "model",
            }

        return self._fallback_result(text, fallback_reason)

    def _parse_with_slot_model(self, text: str) -> Tuple[Optional[Dict[str, object]], str]:
        slots = self.slot_tagger.predict(text)
        if not isinstance(slots, dict):
            return None, "invalid_slot_output"

        start_text = slots.get("start_text")
        end_text = slots.get("end_text")
        waypoint_texts = slots.get("waypoint_texts", [])

        if waypoint_texts is None:
            waypoint_texts = []
        if not isinstance(waypoint_texts, list):
            return None, "invalid_slot_output"
        if not start_text or not end_text:
            return None, "missing_required_slots"

        start = self.color_normalizer.normalize(start_text)
        end = self.color_normalizer.normalize(end_text)
        waypoints, unmapped_waypoint = self._normalize_waypoints(waypoint_texts)

        if start is None or end is None:
            return None, "unmapped_required_color"
        if unmapped_waypoint:
            return None, "unmapped_waypoint_color"

        missing_slots = []
        if start is None:
            missing_slots.append("start")
        if end is None:
            missing_slots.append("end")

        return {
            "intent": "navigation",
            "start": start,
            "waypoints": waypoints,
            "end": end,
            "is_complete": not missing_slots,
            "missing_slots": missing_slots,
        }, ""

    def _normalize_waypoints(self, waypoint_texts: List[str]) -> Tuple[List[str], bool]:
        waypoints = []
        for waypoint_text in waypoint_texts:
            if not waypoint_text or not isinstance(waypoint_text, str):
                return waypoints, True
            waypoint = self.color_normalizer.normalize(waypoint_text)
            if waypoint is not None:
                waypoints.append(waypoint)
            else:
                return waypoints, True
        return waypoints, False

    def _is_navigation_like(self, intent: str) -> bool:
        is_navigation_like = getattr(self.intent_classifier, "is_navigation_like", None)
        if callable(is_navigation_like):
            return is_navigation_like(intent)
        return intent in {"navigation", "add_waypoint", "query_route"}

    def _fallback_result(self, text: str, reason: str) -> Dict[str, object]:
        return {
            "result": self.fallback_parser.parse(text),
            "used_fallback": True,
            "fallback_reason": reason,
            "slot_source": "fallback",
        }

    @staticmethod
    def _unknown_result() -> Dict[str, object]:
        return {
            "intent": "unknown",
            "start": None,
            "waypoints": [],
            "end": None,
            "is_complete": False,
            "missing_slots": [],
        }
