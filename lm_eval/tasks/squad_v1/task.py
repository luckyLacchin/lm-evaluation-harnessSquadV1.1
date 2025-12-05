from functools import partial
import evaluate

from lm_eval.api.task import ConfigurableTask
from lm_eval.api.instance import Instance


def _squad_metric(predictions, references):
    metric = evaluate.load("squad")  # SQuAD 1.1
    return metric.compute(predictions=predictions, references=references)


def _squad_agg(key, items):
    predictions, references = zip(*items)
    return _squad_metric(predictions=predictions, references=references).get(key, 0.0)


class SQuAD1(ConfigurableTask):
    """
    SQuAD 1.1 task (answerable questions only).
    """

    VERSION = 1
    DATASET_PATH = "squad"
    DATASET_NAME = None

    def __init__(self, config=None):
        # config is what comes from the YAML (includes 'task', 'class', maybe 'metadata', ...)
        if config is None:
            config = {}

        # Remove the 'class' key coming from YAML, TaskConfig doesn't know this field
        config.pop("class", None)

        # Ensure we have metadata with a version
        metadata = config.get("metadata", {})
        metadata["version"] = self.VERSION
        config["metadata"] = metadata

        # Now delegate to the base class, safe to call TaskConfig(**config)
        super().__init__(config=config)

    # ------------ splits ------------

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    # ------------ formatting ------------

    def doc_to_text(self, doc):
        return (
            "Context: "
            + doc["context"]
            + "\n\nQuestion: "
            + doc["question"]
            + "\nAnswer:"
        )

    def doc_to_target(self, doc):
        # SQuAD1.1 always has at least one answer
        answer = doc["answers"]["text"][0]
        return " " + answer

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["context"]

    # ------------ LM requests ------------

    def construct_requests(self, doc, ctx, **kwargs):
        """
        Single generate_until request: model generates the answer.
        We ignore extra kwargs (apply_chat_template, chat_template, etc.)
        and explicitly set repeats=1.
        """
        inst = Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(ctx, {"until": ["\n"]}),
            idx=0,
        )

        # Make 100% sure repeats is an int, not None
        inst.repeats = 1

        return [inst]


    # ------------ metrics ------------

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
            "exact": partial(_squad_agg, "exact"),
            "f1": partial(_squad_agg, "f1"),
        }

    def higher_is_better(self):
        return {
            "exact": True,
            "f1": True,
        }
