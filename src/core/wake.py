# src/core/wake.py
from typing import List, Optional, Tuple, Dict, Any
from rapidfuzz import fuzz
from src.utils.textnorm import normalize_text

class _FuzzyWake:
    def __init__(self, phrases: List[str], threshold: int = 72):
        self.raw = list(phrases or [])
        self.norm = [normalize_text(p) for p in (phrases or [])]
        self.threshold = int(threshold)

    def match(self, user_text: str) -> Optional[str]:
        t = normalize_text(user_text)
        if not t:
            return None
        best: Tuple[str, int] = ("", -1)
        for raw, n in zip(self.raw, self.norm):
            score = fuzz.partial_ratio(n, t)
            if score > best[1]:
                best = (raw, score)
        return best[0] if best[1] >= self.threshold else None

    def debug_scores(self, user_text: str):
        t = normalize_text(user_text or "")
        return {raw: fuzz.partial_ratio(n, t) for raw, n in zip(self.raw, self.norm)}

class WakeDetector:
    """
    Fuzzy text matching pentru wake word detection.
    Folosește ASR + potrivire text pentru a detecta fraze de trezire.
    """
    def __init__(self, cfg: Dict[str, Any], logger=None):
        self.cfg = cfg or {}
        self.log = logger
        self.fuzzy = _FuzzyWake(self.cfg.get("wake_phrases") or [], threshold=int(self.cfg.get("threshold", 72)))

    def match(self, user_text: str) -> Optional[str]:
        return self.fuzzy.match(user_text)

    def debug_scores(self, user_text: str):
        return self.fuzzy.debug_scores(user_text)

    def close(self):
        pass
