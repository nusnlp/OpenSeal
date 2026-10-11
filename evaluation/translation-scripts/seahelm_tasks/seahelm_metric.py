import logging
import re
import string

import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from base_logger import get_logger

logger = get_logger(__name__)


class SeaHelmMetric:
    def __init__(
        self,
        inference_df: pd.DataFrame,
        task_config: dict,
        task: str,
        lang: str,
        response_column: str = "responses",
        postprocessed_response_column: str = "cleaned_response",
        label_column: str = "label",
    ):
        self.task_config = task_config
        self.task = task
        self.lang = lang
        self.inference_df = inference_df
        self.response_column = response_column
        self.postprocessed_response_column = postprocessed_response_column
        self.label_column = label_column

    def get_response(self, row: pd.Series, turn: int = 0) -> str:
        return row[self.response_column][turn]

    def get_response_counts(self) -> dict:
        logger.debug("Response Value Counts:")
        counts = self.inference_df.apply(self.get_response, axis=1).value_counts()
        logger.debug(counts)
        return counts.to_json()

    def drop_error_responses(self) -> None:
        should_drop = []
        for response_list in self.inference_df[self.response_column]:
            drop = False
            for response in response_list:
                if response is None:
                    drop = True
                    break
            should_drop.append(drop)

        self.inference_df = self.inference_df[~pd.Series(should_drop)].copy()

    def replace_error_responses(self, replacement: str = "") -> None:
        self.inference_df[self.response_column] = self.inference_df[
            self.response_column
        ].map(
            lambda response: [
                replacement if item is None else item for item in response
            ]
        )

    def normalize_answer(self, s: str) -> str:
        """Lower text and remove punctuation and extra whitespace."""

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            # Remove punctuations + trailing whitespace/newline
            return text.strip(string.punctuation + " " + "\n")

        def lower(text):
            return text.lower()

        return white_space_fix(remove_punc(lower(s)))

    def calculate_metrics(self):
        pass

    def extract_response(self, response: list, use_lowercase: bool = False) -> str:
        # try to extract the answer from the response using regex else return the response as it is
        if use_lowercase:
            _response = response[0].lower()
        else:
            _response = response[0]

        try:
            output = re.search(self.regex_string, _response).group(0)
            output = output.strip("$")
        except:
            output = _response

        return output.strip()

    def evaluate_responses(
        self, drop_error_response: bool = False
    ) -> tuple[dict, pd.DataFrame]:
        if drop_error_response:
            logger.info("Dropping error responses")
            self.drop_error_responses()
        else:
            logger.info('Replacing error responses with ""')
            self.replace_error_responses()

        logger.info("Post processing responses...")
        self.postprocess_responses()
        logger.info("Post processing of responses completed!")

        if logger.isEnabledFor(logging.DEBUG):
            self.get_response_counts()

        logger.info("Calculating metrics...")
        output_json, inference_df = self.calculate_metrics()
        logger.info("Metrics calculation completed!")
        metric = {self.task: output_json}
        return metric, inference_df

    def postprocess_responses(self) -> None:
        self.inference_df[self.postprocessed_response_column] = self.inference_df[
            self.response_column
        ].map(self.extract_response)

    def normalize_score(
        self, score: float, min_score: float, max_score: float
    ) -> float:
        normalized_score = max((score - min_score) / (max_score - min_score), 0)
        return normalized_score


class F1AccMetric(SeaHelmMetric):
    def __init__(
        self, inference_df: pd.DataFrame, task_config: dict, task: str, lang: str
    ):
        super().__init__(
            inference_df=inference_df, task_config=task_config, task=task, lang=lang
        )
        self.null_label = task_config["null_label"]

    def extract_response(self, response: list) -> str | int:
        output = super().extract_response(response, use_lowercase=True)
        output = self.normalize_answer(output)
        output = self.label_map.get(output, self.null_label)
        return output

    def calculate_metrics(self) -> tuple[dict, pd.DataFrame]:
        predictions = self.inference_df[self.postprocessed_response_column]
        references = self.inference_df[self.label_column]

        labels = list(set(references))
        null_count = sum(predictions == self.null_label)

        accuracy = balanced_accuracy_score(
            y_true=references,
            y_pred=predictions,
        )
        individual_scores = predictions.eq(references, axis=0).astype(int)
        self.inference_df["individual_scores"] = [
            {"normalized_accuracy": x} for x in individual_scores
        ]

        avg_f1 = f1_score(
            y_true=references,
            y_pred=predictions,
            labels=labels,
            average="macro",
        )
        null_weighted_f1 = avg_f1 * (1 - null_count / len(predictions))

        macro_f1 = f1_score(
            y_true=references,
            y_pred=predictions,
            average="macro",
        )
        conf_matrix = confusion_matrix(y_true=references, y_pred=predictions)
        class_report = classification_report(y_true=references, y_pred=predictions)
        logger.info(
            f"Balanced Acc = {accuracy*100:.2f} | Macro-F1 = {macro_f1*100:.2f} | Null-Weighted-F1 = {null_weighted_f1*100:.2f}"
        )
        logger.info("Confusion matrix:\n%s", conf_matrix)
        logger.info("Classification report:\n%s", class_report)

        metric_dict = {
            "accuracy": 100 * accuracy,
            "macro_f1": 100 * macro_f1,
            "null_weighted_f1": 100 * null_weighted_f1,
            "normalized_accuracy": 100
            * self.normalize_score(accuracy, 1 / len(self.label_map), 1),
            "null_count": null_count,
        }
        return metric_dict, self.inference_df
