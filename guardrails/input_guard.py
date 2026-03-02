from pydantic import BaseModel


class GuardResult(BaseModel):
    passed: bool
    content: str = ""           # populated by OutputGuard; empty for InputGuard
    reason: str | None = None   # set when passed=False
    warning: str | None = None  # set for non-fatal issues (e.g. truncation)


_INJECTION_PATTERNS: list[str] = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard previous",
    "you are now",
    "forget your instructions",
]


class InputGuard:
    """Rule-based input validator.

    Checks (in order):
    1. Minimum length
    2. Maximum length
    3. Prompt-injection pattern detection (case-insensitive substring match)
    """

    def __init__(self, max_length: int = 2000, min_length: int = 3) -> None:
        self._min_length = min_length
        self._max_length = max_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, query: str) -> GuardResult:
        query = query.strip()

        if len(query) < self._min_length:
            return GuardResult(passed=False, reason="Query too short.")

        if len(query) > self._max_length:
            return GuardResult(passed=False, reason="Query too long.")

        lowered = query.lower()
        for pattern in _INJECTION_PATTERNS:
            if pattern in lowered:
                return GuardResult(
                    passed=False,
                    reason="Potential prompt injection detected.",
                )

        return GuardResult(passed=True)
