# lm-evaluation-harness

## Setup

Please use lm-evaluation-harness version v0.4.9.1.

```bash
git clone --branch v0.4.9.1 --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness

# Optional: verify the checked out tag
git describe --tags

pip install -e .
```

## Required fixes

1. PAWS-X: no change needed.
2. XCOPA (English): copy from `/code/evaluation/lm-evaluation-harness-addons/xcopa` into `lm_eval/tasks/xcopa`:
	 - `default_en.yaml`
	 - `utils.py`
	 - In `lm_eval/tasks/xcopa/_xcopa.yaml`, add:
	 	```yaml
		task:
			- xcopa_en
		```
3. XNLI: in `lm_eval/tasks/xnli/xnli_common_yaml`, add:

```yaml
test_split: test
```

## Run example

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="sail/Sailor2-8B"
TASKS="pawsx,xcopa,xnli,xquad"
OUTPUT_PATH="outputs/lm-eval/Sailor2-8B"

lm_eval --model hf \
	--model_args pretrained=${MODEL},parallelize=True \
	--tasks ${TASKS} \
	--log_samples \
	--output_path ${OUTPUT_PATH} \
	--batch_size 32
```

## Notes

### XQuAD scores
To reproduce XQuAD scores for Thai (`th`) and Chinese (`zh`), run the additional scripts in `/code/evaluation/lm-evaluation-harness-addons/xquad`.

You may need to install the following extra Python modules:
- `jieba==0.42.1`
- `pythainlp==5.0.5`

How to run:
1. Go to `/code/evaluation/lm-evaluation-harness-addons/xquad`.
2. In `process_f1_th_zh_scores.sh`, set `RESULT_DIR` to the same path used for `OUTPUT_PATH` in your `lm_eval` run.
3. Run `process_f1_th_zh_scores.sh`.

After it finishes, check `f1_th_zh_scores.log` for the recalculated `th` and `zh` F1 scores.

# Translation

This section runs translation evaluation and computes BLEU scores from the generated outputs.

## Installation
To evaluate BLEU scores correctly, install the following Python libraries in your environment:

- `khmer-nltk`
- `laonlp`
- `indic-nlp-library`
- `pythainlp`

You also need to install `myTokenize` for Burmese tokenization.

How to install `myTokenize`:

```bash
git clone https://github.com/ThuraAung1601/myTokenize.git
cd myTokenize
pip install --no-deps -r requirements.txt
pip install --no-deps -e .
```

## Run
Go to `openseal-project/code/evaluation`.
```bash
cd openseal-project/code/evaluation
```

1. Run translation generation/evaluation:

```bash
cd translation-scripts
bash run_evaluation.sh
```

This step creates the `inference` folder.

2. Calculate BLEU scores:

```bash
bash process_translation.sh
```

If needed, edit `PARENT` in `process_translation.sh` to point to your desired directory.

