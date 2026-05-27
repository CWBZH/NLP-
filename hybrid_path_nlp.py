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
                "model_slots": None,
            }

        if not self._is_navigation_like(intent):
            return self._fallback_result(text, "unsupported_intent", None)

        try:
            model_result, fallback_reason, model_slots = self._parse_with_slot_model(text)
        except Exception as error:
            return self._fallback_result(text, f"slot_tagger_exception:{type(error).__name__}", None)

        if model_result is not None:
            return {
                "result": model_result,
                "used_fallback": False,
                "fallback_reason": None,
                "slot_source": "model",
                "model_slots": model_slots,
            }

        return self._fallback_result(text, fallback_reason, model_slots)

    def _parse_with_slot_model(
        self, text: str
    ) -> Tuple[Optional[Dict[str, object]], str, Optional[Dict[str, object]]]:
        slots = self.slot_tagger.predict(text)
        if not isinstance(slots, dict):
            return None, "invalid_slot_output", None

        start_text = slots.get("start_text")
        end_text = slots.get("end_text")
        waypoint_texts = slots.get("waypoint_texts", [])

        if waypoint_texts is None:
            waypoint_texts = []
        if not isinstance(waypoint_texts, list):
            return None, "invalid_slot_output", slots
        if not start_text and not end_text and not waypoint_texts:
            return None, "missing_required_slots", slots

        start = self.color_normalizer.normalize(start_text) if start_text else None
        end = self.color_normalizer.normalize(end_text) if end_text else None
        waypoints, unmapped_waypoint = self._normalize_waypoints(waypoint_texts)

        if start_text and start is None:
            return None, "normalization_failed:start", slots
        if end_text and end is None:
            return None, "normalization_failed:end", slots
        if unmapped_waypoint:
            return None, "normalization_failed:waypoint", slots
        if start is None and end is None and not waypoints:
            return None, "missing_required_slots", slots
        if start is None and self._text_has_start_expression(text):
            return None, "missing_start_from_start_expression", slots
        if end is None and self._text_has_end_expression(text):
            return None, "missing_end_from_end_expression", slots

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
        }, "", slots

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

    @staticmethod
    def _text_has_start_expression(text: str) -> bool:
        return "从" in text or "起点" in text or "现在在" in text

    @staticmethod
    def _text_has_end_expression(text: str) -> bool:
        return "到" in text or "去" in text or "终点" in text or "最后" in text

    def _fallback_result(
        self, text: str, reason: str, model_slots: Optional[Dict[str, object]]
    ) -> Dict[str, object]:
        return {
            "result": self.fallback_parser.parse(text),
            "used_fallback": True,
            "fallback_reason": reason,
            "slot_source": "fallback",
            "model_slots": model_slots,
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
