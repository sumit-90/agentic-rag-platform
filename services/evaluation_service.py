import structlog

from evaluation.dataset_builder import DatasetBuilder, EvalSample
from evaluation.ragas_evaluator import RagasEvaluator

logger = structlog.get_logger(__name__)


class EvaluationService:
    """Orchestrates dataset construction and RAGAS evaluation.

    Usage::

        service = EvaluationService()
        scores = service.run(samples)
        # {"faithfulness": 0.91, "response_relevancy": 0.87, ...}
    """

    def __init__(self) -> None:
        self._builder = DatasetBuilder()
        self._evaluator = RagasEvaluator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, samples: list[EvalSample]) -> dict[str, float]:
        logger.info("evaluation_service_start", sample_count=len(samples))
        dataset = self._builder.build(samples)
        return self._evaluator.evaluate(dataset)
