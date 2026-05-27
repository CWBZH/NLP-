import re
from typing import Dict, List, Optional

from path_nlp_parser import PathNLPParser


class ColorNormalizer:
    """Normalize Chinese color mentions to canonical map color names."""

    def __init__(self, aliases: Optional[Dict[str, List[str]]] = None):
        self.aliases = aliases or PathNLPParser.COLOR_ALIASES
        self._alias_to_color = {}
        for color, color_aliases in self.aliases.items():
            for alias in color_aliases:
                self._alias_to_color[alias] = color

        alias_pattern = "|".join(
            re.escape(alias) for alias in sorted(self._alias_to_color, key=len, reverse=True)
        )
        self._alias_re = re.compile(alias_pattern)

    def normalize(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None

        compact_text = re.sub(r"\s+", "", text)
        if compact_text in self._alias_to_color:
            return self._alias_to_color[compact_text]

        match = self._alias_re.search(compact_text)
        if match:
            return self._alias_to_color[match.group(0)]

        return None
