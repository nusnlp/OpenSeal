import copy

from base_logger import get_logger
from constants import PRAGMATICS_MIN_SCORE

logger = get_logger(__name__)


def aggregate_pragmatics_metrics(metrics: dict, lang="id") -> None:
    logger.info("---------- Task: PRAGMATICS (%s) ----------", lang.upper())
    PRAGMATICS_PHENOMENA = ["scalar_implicatures", "presuppositions"]
    PRAGMATICS_TASKS = [
        "pragmatic-single",
        "pragmatic-pair",
    ]

    metrics[lang]["linguistic-diagnostics"]["pragmatics"] = {
        "accuracy": None,
        "subcategories": {},
    }
    for phenomenon in PRAGMATICS_PHENOMENA:
        correct_count, total_count = 0, 0

        for task in PRAGMATICS_TASKS:
            if (
                phenomenon
                in metrics[lang]["linguistic-diagnostics"][task]["subcategories"]
            ):
                correct_count += metrics[lang]["linguistic-diagnostics"][task][
                    "subcategories"
                ][phenomenon][0]
                total_count += metrics[lang]["linguistic-diagnostics"][task][
                    "subcategories"
                ][phenomenon][1]
        subset_accuracy = correct_count / total_count

        metrics[lang]["linguistic-diagnostics"]["pragmatics"]["subcategories"][
            phenomenon
        ] = {
            "accuracy": subset_accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
        }

        logger.info(
            "Accuracy for phenomenon <%s_%s>: %d / %d : %f",
            lang.upper(),
            phenomenon,
            correct_count,
            total_count,
            subset_accuracy,
        )

    pragmatics_subset_scores = [
        metrics[lang]["linguistic-diagnostics"]["pragmatics"]["subcategories"][
            phenomenon
        ]["accuracy"]
        for phenomenon in PRAGMATICS_PHENOMENA
    ]
    overall_accuracy = sum(pragmatics_subset_scores) / len(pragmatics_subset_scores)
    metrics[lang]["linguistic-diagnostics"]["pragmatics"]["accuracy"] = (
        overall_accuracy * 100
    )
    logger.info(
        "Overall accuracy for <%s_linguistic_diagnostics_pragmatics>: %f",
        lang,
        overall_accuracy,
    )

    # score normalization for pragmatic reasoning
    min_score = PRAGMATICS_MIN_SCORE
    max_score = 1
    normalized_accuracy = max(
        (overall_accuracy - min_score) / (max_score - min_score), 0
    )
    metrics[lang]["linguistic-diagnostics"]["pragmatics"]["normalized_accuracy"] = (
        normalized_accuracy * 100
    )
    logger.info(
        "Normalized accuracy for <%s_linguistic_diagnostics_pragmatics>: %f",
        lang,
        normalized_accuracy,
    )
    return metrics


def aggregate_lindsea_metrics(metrics: dict, lang="id") -> None:
    LINDSEA_CATEGORIES = ["mp-r", "pragmatics"]

    lindsea_scores = []
    metrics[lang]["linguistic-diagnostics"]["lindsea"] = {"subcategories": {}}

    for category in LINDSEA_CATEGORIES:
        category_metrics = metrics[lang]["linguistic-diagnostics"][category]
        metrics[lang]["linguistic-diagnostics"]["lindsea"]["subcategories"][
            category
        ] = category_metrics

        lindsea_scores.append(category_metrics["normalized_accuracy"])

    overall_accuracy = sum(lindsea_scores) / len(lindsea_scores)
    metrics[lang]["linguistic-diagnostics"]["lindsea"] = {
        "normalized_accuracy": overall_accuracy,
        "subcategories": {},
    }
    logger.info(
        "Overall normalized accuracy for <%s_lindsea>: %f\n", lang, overall_accuracy
    )

    return metrics


def aggregate_metrics(metrics: dict, config) -> None:
    logger.info("---------- Aggregation of metrics ----------")
    total_all_langs = {}
    for lang, competencies in metrics.items():
        logger.info("---------- Aggregation | Lang: %s ----------", lang.upper())
        total_lang = {}
        for competency, tasks in competencies.items():
            logger.info(
                "### Competency: %s",
                competency.upper(),
            )
            # handle special case for linguistic-diagnostics
            if competency == "linguistic-diagnostics":
                metrics = aggregate_pragmatics_metrics(metrics, lang)
                metrics = aggregate_lindsea_metrics(metrics, lang)

                total_lang[competency] = metrics[lang]["linguistic-diagnostics"][
                    "lindsea"
                ]["normalized_accuracy"]
                metrics[lang][competency]["total"] = metrics[lang][
                    "linguistic-diagnostics"
                ]["lindsea"]["normalized_accuracy"]
                continue

            scores = {}
            aggregations = {}

            _tasks = copy.deepcopy(tasks)
            for task, results in _tasks.items():
                metric = config["tasks"][task]["metric"]
                aggregation_group = config["tasks"][task].get("aggregation_group", None)

                if aggregation_group:
                    if aggregation_group not in aggregations:
                        aggregations[aggregation_group] = [results[metric]]
                    else:
                        aggregations[aggregation_group].append(results[metric])

                    _aggregation_total = sum(aggregations[aggregation_group]) / len(
                        aggregations[aggregation_group]
                    )
                    scores[aggregation_group] = _aggregation_total
                    metrics[lang][competency][aggregation_group] = _aggregation_total
                else:
                    scores[task] = results[metric]

            competency_total = sum(scores.values()) / len(scores)
            total_lang[competency] = competency_total
            metrics[lang][competency]["total"] = competency_total
            logger.info(
                "Overall normalized accuracy for <%s_%s>: %f\n",
                lang,
                competency,
                competency_total,
            )

        language_total = sum(total_lang.values()) / len(total_lang)
        total_all_langs[lang] = language_total
        metrics[lang]["total"] = language_total

        logger.info("Overall normalized accuracy for <%s>: %f\n", lang, language_total)

    overall_total = sum(total_all_langs.values()) / len(total_all_langs)
    metrics["total"] = overall_total
    logger.info("Overall normalized accuracy: %f\n", overall_total)
    logger.info("Aggregation of all metrics completed\n")

    return metrics
