from functools import partial
import evaluate

from lm_eval.api.task import ConfigurableTask
from lm_eval.api.instance import Instance


def squad_metric(predictions, references):
    metric = evaluate.load("squad")
    return metric.compute(predictions=predictions, references=references)


def _agg(key, items):
    predictions, references = zip(*items)
    return squad_metric(predictions, references)[key]


class SQuAD1(ConfigurableTask):
    VERSION = 1
    DATASET_PATH = "squad"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return (
            "Context: " + doc["context"] + "\n\n"
            "Question: " + doc["question"] + "\nAnswer:"
        )

    def doc_to_target(self, doc):
        return " " + doc["answers"]["text"][0]

    def construct_requests(self, doc, ctx, **kwargs):
        return [
            Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(ctx, {"until": ["\n"]}),
                idx=0,
                **kwargs,
            ),
        ]

    def process_results(self, doc, results):
        continuation = results[0]

        predictions = {
            "id": doc["id"],
            "prediction_text": continuation,
        }
        references = {
            "id": doc["id"],
            "answers": doc["answers"],
        }

        return {
            "exact": (predictions, references),
            "f1": (predictions, references),
        }

    def aggregation(self):
        return {
            "exact": partial(_agg, "exact"),
            "f1": partial(_agg, "f1"),
        }

    def higher_is_better(self):
        return {
            "exact": True,
            "f1": True,
        }