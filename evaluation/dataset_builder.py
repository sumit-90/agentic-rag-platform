from pydantic import BaseModel
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample


class EvalSample(BaseModel):
    """A single question-answer pair with supporting context for RAGAS evaluation."""

    question: str
    answer: str
    contexts: list[str]    # retrieved context chunks used to generate the answer
    ground_truth: str      # reference answer for recall-based metrics


class DatasetBuilder:
    """Converts a list of EvalSample objects into a RAGAS EvaluationDataset."""

    def build(self, samples: list[EvalSample]) -> EvaluationDataset:
        ragas_samples = [
            SingleTurnSample(
                user_input=s.question,
                response=s.answer,
                retrieved_contexts=s.contexts,
                reference=s.ground_truth,
            )
            for s in samples
        ]
        return EvaluationDataset(samples=ragas_samples)
