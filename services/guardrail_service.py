import structlog

from guardrails.input_guard import GuardResult, InputGuard
from guardrails.output_guard import OutputGuard

logger = structlog.get_logger(__name__)


class GuardrailService:
    """Thin orchestrator for input and output guards.

    Both guards are injected so they can be overridden in tests or
    extended with custom rule sets without changing this class.
    """

    def __init__(
        self,
        input_guard: InputGuard | None = None,
        output_guard: OutputGuard | None = None,
    ) -> None:
        self._input = input_guard or InputGuard()
        self._output = output_guard or OutputGuard()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_input(self, query: str) -> GuardResult:
        result = self._input.check(query)
        logger.info(
            "input_guard_check",
            query=query[:100],
            passed=result.passed,
            reason=result.reason,
        )
        return result

    def check_output(self, answer: str) -> GuardResult:
        result = self._output.check(answer)
        logger.info(
            "output_guard_check",
            passed=result.passed,
            warning=result.warning,
        )
        return result
