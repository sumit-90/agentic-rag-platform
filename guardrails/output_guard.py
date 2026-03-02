from guardrails.input_guard import GuardResult


class OutputGuard:
    """Rule-based output validator.

    Checks (in order):
    1. Empty response guard
    2. Maximum length guard (truncates rather than blocks)
    """

    def __init__(self, max_length: int = 8000) -> None:
        self._max_length = max_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, answer: str) -> GuardResult:
        answer = answer.strip()

        if not answer:
            return GuardResult(passed=False, reason="Empty response from model.")

        if len(answer) > self._max_length:
            truncated = answer[: self._max_length] + " [truncated]"
            return GuardResult(
                passed=True,
                content=truncated,
                warning="Response truncated.",
            )

        return GuardResult(passed=True, content=answer)
