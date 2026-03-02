from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

# Default metric suite used by RagasEvaluator when no custom metrics are supplied.
# Each entry is an instantiated metric object (RAGAS 0.2+ API).
DEFAULT_METRICS = [
    Faithfulness(),
    ResponseRelevancy(),
    LLMContextPrecisionWithReference(),
    LLMContextRecall(),
]
