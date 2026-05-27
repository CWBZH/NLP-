import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class _ColorMention:
    color: str
    start: int
    end: int


class PathNLPParser:
    """Parse simple Chinese navigation requests for a seven-color map."""

    COLOR_ALIASES: Dict[str, List[str]] = {
        "red": ["赤", "赤色", "赤点", "赤色点", "红", "红色", "红点", "红色点"],
        "orange": ["橙", "橙色", "橙点", "橙色点"],
        "yellow": ["黄", "黄色", "黄点", "黄色点"],
        "green": ["绿", "绿色", "绿点", "绿色点"],
        "cyan": ["青", "青色", "青点", "青色点"],
        "blue": ["蓝", "蓝色", "蓝点", "蓝色点"],
        "purple": ["紫", "紫色", "紫点", "紫色点"],
    }

    _NAVIGATION_KEYWORDS = (
        "从",
        "到",
        "去",
        "出发",
        "经过",
        "途经",
        "途径",
        "路过",
        "规划",
        "路线",
        "导航",
        "怎么走",
    )

    def __init__(self):
        alias_to_color = {}
        for color, aliases in self.COLOR_ALIASES.items():
            for alias in aliases:
                alias_to_color[alias] = color

        self._alias_to_color = alias_to_color
        alias_pattern = "|".join(
            re.escape(alias) for alias in sorted(alias_to_color, key=len, reverse=True)
        )
        self._color_re = re.compile(alias_pattern)
        self._from_re = re.compile(rf"从\s*(?P<color>{alias_pattern})")
        self._depart_re = re.compile(rf"(?P<color>{alias_pattern})\s*出发")
        self._end_re = re.compile(rf"(?:最后\s*)?(?:到|去)\s*(?P<color>{alias_pattern})")
        self._waypoint_re = re.compile(
            r"(?:先\s*)?(?:再\s*)?(?:经过|途经|途径|路过)\s*"
            r"(?P<body>.*?)(?=(?:最后\s*)?(?:到|去)|[，,。；;]|$)"
        )

    def parse(self, text):
        if not isinstance(text, str):
            text = "" if text is None else str(text)

        mentions = self._find_colors(text)
        if not self._is_navigation_intent(text, mentions):
            return self._result("unknown", None, [], None)

        start = self._extract_start(text, mentions)
        end = self._extract_end(text)
        waypoints = self._extract_waypoints(text)

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
        }

    def _result(self, intent, start, waypoints, end):
        return {
            "intent": intent,
            "start": start,
            "waypoints": waypoints,
            "end": end,
            "is_complete": intent == "navigation" and start is not None and end is not None,
            "missing_slots": [],
        }

    def _is_navigation_intent(self, text: str, mentions: List[_ColorMention]) -> bool:
        if not mentions:
            return False
        return any(keyword in text for keyword in self._NAVIGATION_KEYWORDS)

    def _find_colors(self, text: str, offset: int = 0) -> List[_ColorMention]:
        return [
            _ColorMention(
                self._alias_to_color[match.group(0)],
                match.start() + offset,
                match.end() + offset,
            )
            for match in self._color_re.finditer(text)
        ]

    def _extract_start(
        self, text: str, mentions: List[_ColorMention]
    ) -> Optional[str]:
        match = self._from_re.search(text)
        if match:
            return self._alias_to_color[match.group("color")]

        match = self._depart_re.search(text)
        if match:
            return self._alias_to_color[match.group("color")]

        endpoint_match = self._last_end_match(text)
        if endpoint_match:
            waypoint_ranges = self._waypoint_ranges(text)
            candidates = [
                mention
                for mention in mentions
                if mention.end <= endpoint_match.start()
                and not self._in_ranges(mention, waypoint_ranges)
            ]
            if candidates:
                return candidates[-1].color

        return None

    def _extract_end(self, text: str) -> Optional[str]:
        match = self._last_end_match(text)
        if match:
            return self._alias_to_color[match.group("color")]
        return None

    def _extract_waypoints(self, text: str) -> List[str]:
        waypoints = []
        for match in self._waypoint_re.finditer(text):
            body_start = match.start("body")
            body = match.group("body")
            waypoints.extend(mention.color for mention in self._find_colors(body, body_start))
        waypoints.extend(self._destination_sequence_waypoints(text))
        return waypoints

    def _last_end_match(self, text: str):
        matches = list(self._end_re.finditer(text))
        if not matches:
            return None
        return matches[-1]

    def _destination_sequence_waypoints(self, text: str) -> List[str]:
        matches = list(self._end_re.finditer(text))
        if len(matches) < 2:
            return []
        return [self._alias_to_color[match.group("color")] for match in matches[:-1]]

    def _waypoint_ranges(self, text: str) -> List[tuple]:
        ranges = []
        for match in self._waypoint_re.finditer(text):
            body_start = match.start("body")
            body = match.group("body")
            ranges.extend((mention.start, mention.end) for mention in self._find_colors(body, body_start))

        end_matches = list(self._end_re.finditer(text))
        if len(end_matches) >= 2:
            ranges.extend(
                (match.start("color"), match.end("color")) for match in end_matches[:-1]
            )
        return ranges

    @staticmethod
    def _in_ranges(mention: _ColorMention, ranges: List[tuple]) -> bool:
        return any(mention.start >= start and mention.end <= end for start, end in ranges)
