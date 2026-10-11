import pandas as pd
import torch
from sacrebleu.metrics import CHRF
from transformers import AutoTokenizer

from base_logger import get_logger
from seahelm_tasks.nlg.translation.models import MT5ForRegression
from seahelm_tasks.seahelm_metric import SeaHelmMetric

logger = get_logger(__name__)

try:
    from seahelm_tasks.nlg.translation.evaluation_comet import (
        CometKiwiMetric,
        CometMetric,
    )
except ImportError:
    logger.warning(
        "COMET not installed. Please install COMET to use the COMET metrics."
    )

METRICX_MIN_SCORE, METRICX_MAX_SCORE = 25, 0


class TranslationMetric(SeaHelmMetric):
    def __init__(
        self, inference_df: pd.DataFrame, task_config: dict, task: str, lang: str
    ):
        super().__init__(
            inference_df=inference_df, task_config=task_config, task=task, lang=lang
        )
        self.use_comet_metric = task_config["use_comet_metric"]
        self.use_metricx_metric = task_config["use_metricx_metric"]
        self.use_chrf_metric = task_config["use_chrf_metric"]
        self.regex_string = {
            "id": r"(?<=[T|t]erjemahan:)[\s\r\n]*.*",
            "th": r"(?<=คำแปล:)[\s\r\n]*.*",
            "vi": r"(?<=[B|b]ản dịch:)[\s\r\n]*.*",
            "ta": r"(?<=மொழிபெயர்ப்பு:)[\s\r\n]*.*",
            "jv": r"(?<=[T|t]erjemahan:)[\s\r\n]*.*",
            "su": r"(?<=[T|t]arjamahan:)[\s\r\n]*.*",
            "tl": r"(?<=[S|s]alin:)[\s\r\n]*.*",
            "tl_full": r"(?<=[S|s]alin:)[\s\r\n]*.*",
            "km": r"(?<=ការប្រែសម្រួល:)[\s\r\n]*.*",
            "lo": r"(?<=ການແປ:)[\s\r\n]*.*",
            "my": r"(?<=ဘာသာပြန်ချက် -)[\s\r\n]*.*",
            "ms": r"(?<=[T|t]erjemahan:)[\s\r\n]*.*",
            "zh": r"(?<=翻译:)[\s\r\n]*.*"
        }[lang]

    def evaluate_with_comet(self, references, sources, predictions):
        logger.info("Loading COMET scorers...")
        comet_scorer = CometMetric()
        comet_kiwi_scorer = CometKiwiMetric()

        logger.info("Running COMET scorers...")
        comet_scores = comet_scorer.compute_scores(
            sources=sources, predictions=predictions, references=references
        )
        comet_kiwi_scores = comet_kiwi_scorer.compute_scores(
            sources=sources, predictions=predictions
        )

        logger.info(f'COMET22 System Score: {comet_scores["system_score"]}')
        logger.info(f'COMET-Kiwi System Score: {comet_kiwi_scores["system_score"]}')

        return {
            "comet22": comet_scores["system_score"],
            "comet-kiwi": comet_kiwi_scores["system_score"],
        }

    def setup_metricx_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-xl", use_fast=False)
        self.model = MT5ForRegression.from_pretrained(
            "google/metricx-24-hybrid-xxl-v2p6-bfloat16", torch_dtype="auto"
        )

        self.model = self.model.to("cuda")
        self.model.eval()

    def calculate_metricx_scores(
        self, sources, predictions, references=None, batch_size=64
    ):
        prompts = []
        if references is not None:
            for source, pred, ref in zip(sources, predictions, references):
                prompts.append(f"source: {source} candidate: {pred} reference: {ref}")
        else:
            for source, pred in zip(sources, predictions):
                prompts.append(f"source: {source} candidate: {pred}")

        scores = []
        for i in range(0, len(predictions), batch_size):
            _prompts = prompts[i : i + batch_size]

            tokens = self.tokenizer(
                _prompts,
                truncation=True,
                padding=True,
                max_length=1536,
                return_tensors="pt",
            )

            # remove eos token
            tokens["input_ids"] = tokens["input_ids"][:, :-1]
            tokens["attention_mask"] = tokens["attention_mask"][:, :-1]

            # move tokens to cuda device
            tokens["input_ids"] = tokens["input_ids"].to("cuda")
            tokens["attention_mask"] = tokens["attention_mask"].to("cuda")

            with torch.no_grad():
                outputs = self.model(**tokens)

            _scores = outputs.predictions.cpu().tolist()
            scores.extend(_scores)

        return scores

    def evaluate_with_metricx(
        self, sources, predictions, references=None, batch_size=64
    ):
        # reference scores
        scores = self.calculate_metricx_scores(
            sources, predictions, references, batch_size=batch_size
        )

        metricx_wmt24_scores = sum(scores) / len(scores)
        normalized_scores = [
            self.normalize_score(x, METRICX_MIN_SCORE, METRICX_MAX_SCORE) * 100
            for x in scores
        ]
        metricx_wmt24_norm_scores = sum(normalized_scores) / len(normalized_scores)
        metrics = {
            f"metricx_wmt24_{'wo_ref_' if references is None else ''}scores": metricx_wmt24_scores,
            f"normalized_metricx_wmt24_{'wo_ref_' if references is None else ''}scores": metricx_wmt24_norm_scores,
        }
        logger.info(
            "MetricX WMT24%s score: %f",
            " with references" if references is None else "",
            metricx_wmt24_scores,
        )
        return metrics, normalized_scores

    def evaluate_with_chrf(self, references, predictions):
        chrf = CHRF(word_order=2)

        if len(predictions) > 0:
            scores = chrf.corpus_score(predictions, [references])
            logger.info(f"ChrF++ Score: {scores.score}")
            score = scores.score
        else:
            score = None

        return {
            "chrf_score": score,
            "normalized_chrf_score": self.normalize_score(score, 0, 1),
        }

    def calculate_metrics(self):
        predictions = self.inference_df[self.postprocessed_response_column].to_list()
        sources = (
            self.inference_df["prompts"]
            .map(lambda x: x[0]["text"])
            .astype(str)
            .to_list()
        )
        references = self.inference_df[self.label_column].to_list()

        metric_dict = {}

        # run comet metrics
        if self.use_comet_metric:
            # TODO implement individual scores
            comet_metrics = self.evaluate_with_comet(references, sources, predictions)
            metric_dict.update(comet_metrics)

        # run metricx metrics
        if self.use_metricx_metric:
            batch_size = self.task_config["metricx_batch_size"]
            retry_count = 3
            while batch_size > 0 and retry_count >= 1:
                try:
                    self.setup_metricx_model()
                    metricx_metricx, scores = self.evaluate_with_metricx(
                        sources,
                        predictions,
                        references,
                        batch_size=batch_size,
                    )
                    metric_dict.update(metricx_metricx)

                    metricx_metricx, scores_wo_ref = self.evaluate_with_metricx(
                        sources,
                        predictions,
                        references=None,
                        batch_size=batch_size,
                    )
                    metric_dict.update(metricx_metricx)
                    self.inference_df["individual_scores"] = [
                        {"normalized_metricx_wmt24_scores": x} for x in scores
                    ]
                    break
                except Exception as e:
                    logger.exception("Failed to load MetricX model.")
                    logger.exception(e)
                    batch_size = batch_size // 2
                    retry_count -= 1
                    logger.warning(
                        f"Halving batch size to {batch_size} and trying again..."
                    )
                finally:
                    # delete model and free up vram
                    del self.model
                    del self.tokenizer
                    torch.cuda.empty_cache()

        if self.use_chrf_metric:
            # run CHRF metrics
            chrf_metrics = self.evaluate_with_chrf(references, predictions)
            metric_dict.update(chrf_metrics)

        null_count = sum([1 for pred in predictions if pred == ""])
        metric_dict["null_count"] = null_count

        return metric_dict, self.inference_df
