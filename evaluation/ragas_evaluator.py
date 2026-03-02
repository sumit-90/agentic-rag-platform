import structlog
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset

from evaluation.metrics import DEFAULT_METRICS

logger = structlog.get_logger(__name__)


class RagasEvaluator:
    """Runs RAGAS evaluation on a pre-built EvaluationDataset.

    Metrics default to the four core RAGAS metrics:
    faithfulness, response_relevancy, context_precision, context_recall.
    """

    def __init__(self, metrics: list | None = None) -> None:
        self._metrics = metrics or DEFAULT_METRICS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, dataset: EvaluationDataset) -> dict[str, float]:
        log = logger.bind(
            metrics=[m.__class__.__name__ for m in self._metrics],
            samples=len(dataset),
        )
        log.info("ragas_evaluation_start")

        result = evaluate(dataset=dataset, metrics=self._metrics)

        # result supports dict-like access in RAGAS 0.2+
        scores = {k: round(float(v), 4) for k, v in dict(result).items()}
        log.info("ragas_evaluation_complete", scores=scores)
        return scores
