import comet


REFERENCE_MODEL_NAME = "Unbabel/wmt22-comet-da"
REFERENCELESS_MODEL_NAME = "Unbabel/wmt22-cometkiwi-da"


class CometMetric:
    def __init__(self):
        model_path = comet.download_model(REFERENCE_MODEL_NAME)
        self.scorer = comet.load_from_checkpoint(model_path)

    def compute_scores(self, sources: list, predictions: list, references: list):
        data = {"src": sources, "mt": predictions, "ref": references}
        data = [dict(zip(data, t)) for t in zip(*data.values())]
        results = self.scorer.predict(
            data,
            batch_size=8,
            gpus=1,
            mc_dropout=False,
            progress_bar=True,
            accelerator="cuda",
            num_workers=None,
            length_batching=True,
        )
        return {"scores": results["scores"], "system_score": results["system_score"]}


class CometKiwiMetric:
    def __init__(self):
        model_path = comet.download_model(REFERENCELESS_MODEL_NAME)
        self.scorer = comet.load_from_checkpoint(model_path)

    def compute_scores(self, sources: list, predictions: list):
        data = {"src": sources, "mt": predictions}
        data = [dict(zip(data, t)) for t in zip(*data.values())]
        results = self.scorer.predict(
            data,
            batch_size=8,
            gpus=1,
            mc_dropout=False,
            progress_bar=True,
            accelerator="cuda",
            num_workers=None,
            length_batching=True,
        )
        return {"scores": results["scores"], "system_score": results["system_score"]}
