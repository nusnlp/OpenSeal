# OpenSeal

This repository provides the code to perform controlled experiments on continual pretraining strategies and built the first fully open-source Southeast Asian LLM, OpenSeal, as reported in the following publication:

> OpenSeal: Good, Fast, and Cheap Construction of an Open-Source Southeast Asian LLM via Parallel Data  <br>
> [Tan Sang Nguyen](https://scholar.google.com/citations?user=1RyWgLgAAAAJ&hl=en), [Muhammad Reza Qorib](https://mrqorib.github.io/), and [Hwee Tou Ng](https://www.comp.nus.edu.sg/~nght/)  <br> 
> Findings of the 2026 Conference on Empirical Methods in Natural Language Processing ([PDF](https://arxiv.org/abs/2602.02266))

We use the [OLMo](https://github.com/allenai/olmo) code base for the [training](./training) code and [LM-Eval Harness](https://github.com/eleutherAI/lm-evaluation-harness/) for the evaluation.

## Final Models
* OpenSeal 1B: [nusnlp/OpenSeal-1B](https://huggingface.co/nusnlp/OpenSeal-1B)
* OpenSeal 7B: [nusnlp/OpenSeal-7B](https://huggingface.co/nusnlp/OpenSeal-7B)

# Experiment Models
* 1B
  * Multilingual: [nusnlp/OpenSeal-1B-Multilingual](https://huggingface.co/nusnlp/OpenSeal-1B-Multilingual)
  * Mixed: [nusnlp/OpenSeal-1B-Mixed](https://huggingface.co/nusnlp/OpenSeal-1B-Mixed)
  * Parallel First [nusnlp/OpenSeal-1B-Parallel-First](https://huggingface.co/nusnlp/OpenSeal-1B-Parallel-First)
  * Parallel Last [nusnlp/OpenSeal-1B-Parallel-Last](https://huggingface.co/nusnlp/OpenSeal-1B-Parallel-Last)
* 7B
  * Multilingual: [nusnlp/OpenSeal-7B-Multilingual](https://huggingface.co/nusnlp/OpenSeal-7B-Multilingual)
  * Multilingual-Replacement: [nusnlp/OpenSeal-7B-Multilingual-Replacement](https://huggingface.co/nusnlp/OpenSeal-7B-Multilingual-Replacement)

## Training Data
| Experiment | Datasets |
| ---------- | --- |
| OpenSeal 1B, OpenSeal 7B | [nusnlp/OpenSeal-Parallel](https://huggingface.co/datasets/nusnlp/OpenSeal-Parallel) + [nusnlp/OpenSeal-OLMo2-Replay](https://huggingface.co/datasets/nusnlp/OpenSeal-OLMo2-Replay) |
| Multilingual | [nusnlp/OpenSeal-SEA-PILE-v2](https://huggingface.co/datasets/nusnlp/OpenSeal-SEA-PILE-v2) + [nusnlp/OpenSeal-MapCC](https://huggingface.co/datasets/nusnlp/OpenSeal-MapCC) + [nusnlp/OpenSeal-OLMo2-Replay](https://huggingface.co/datasets/nusnlp/OpenSeal-OLMo2-Replay) |
| Mixed, Parallel First, Parallel Last | [nusnlp/OpenSeal-SEA-PILE-v2](https://huggingface.co/datasets/nusnlp/OpenSeal-SEA-PILE-v2) + [nusnlp/OpenSeal-MapCC](https://huggingface.co/datasets/nusnlp/OpenSeal-MapCC) + [nusnlp/OpenSeal-Parallel](https://huggingface.co/datasets/nusnlp/OpenSeal-Parallel) + [nusnlp/OpenSeal-OLMo2-Replay](https://huggingface.co/datasets/nusnlp/OpenSeal-OLMo2-Replay) |
| Multilingual-Replacement | [nusnlp/OpenSeal-Multilingual-Replacement](https://huggingface.co/datasets/nusnlp/OpenSeal-Multilingual-Replacement) + [nusnlp/OpenSeal-OLMo2-Replay](https://huggingface.co/datasets/nusnlp/OpenSeal-OLMo2-Replay) |


## Training
Please refer to [training/train.md](./training/train.md) for instructions on reproducing the pretraining of our models.

## Evaluation
Please refer to [evaluation/evaluation.md](./evaluation/evaluation.md) for instructions on evaluating the models.

## License
This repository is licensed under the Apache-2.0 license.

## Citation
If you find our work useful, we kindly ask that you cite our paper.
```
@inproceedings{nguyen-etal-2025-openseal,
    title = "OpenSeal: Good, Fast, and Cheap Construction of an Open-Source Southeast Asian LLM via Parallel Data",
    author = "Nguyen, Tan Sang and
      Qorib, Muhammad Reza  and
      Ng, Hwee Tou",
    booktitle = "Findings of the 2026 Conference on Empirical Methods in Natural Language Processing",
}
```

