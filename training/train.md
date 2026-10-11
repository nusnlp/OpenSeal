# Training

This document summarizes the training-related materials released for OpenSeal.

## Location

- Main training scripts and configuration are under this folder.
- OLMo-related training implementation is in OLMo/.

## What Is Included

- Job scripts for cluster execution (for example, PBS scripts).
- Training code, configs, and utilities used for OpenSeal experiments.

## Training Entry

- Main launch script: olmo-train.pbs
- Training program: OLMo/scripts/train.py

## OpenSeal Configs

### 1B

- Mixed:
	- OLMo/configs/CPT/n10B_Mix_mono-3.75_para-3.75_replay-2.5.yaml
	- OLMo/configs/CPT/n10B_Mix_mono-3.75_para-3.75_replay-2.5_Decay.yaml

- Para-first:
	- OLMo/configs/CPT/n10B_Par-First_mono-3.75_para-3.75_replay-2.5.yaml
	- OLMo/configs/CPT/n10B_Par-First_mono-3.75_para-3.75_replay-2.5_Decay.yaml

- Para-last:
	- OLMo/configs/CPT/n10B_Par-Last_mono-3.75_para-3.75_replay-2.5.yaml
	- OLMo/configs/CPT/n10B_Par-Last_mono-3.75_para-3.75_replay-2.5_Decay.yaml

- Para-only:
	- OLMo/configs/CPT/n10B_para-7.5_replay-2.5.yaml
	- OLMo/configs/CPT/n10B_para-7.5_replay-2.5_Decay.yaml

- Multilingual:
	- OLMo/configs/CPT/n10B_multi-uniform-sea-7.5_replay-2.5.yaml
	- OLMo/configs/CPT/n10B_multi-uniform-sea-7.5_replay-2.5_Decay.yaml

### 7B

- Parallel-only (current OpenSeal 7B):
	- OLMo/configs/CPT/7B_n34.8_para-26_replay-8.7.yaml
	- OLMo/configs/CPT/7B_n34.8_para-26_replay-8.7_Decay.yaml

- Multilingual:
	- OLMo/configs/CPT/7B_n34.8_multilingual-uniform-26_replay-8.7.yaml
	- OLMo/configs/CPT/7B_n34.8_multilingual-uniform-26_replay-8.7_Decay.yaml

- Multilingual replacement:
	- OLMo/configs/CPT/7B_n34.8_multilingual-26_replay-8.7.yaml
	- OLMo/configs/CPT/7B_n34.8_multilingual-26_replay-8.7_Decay.yaml

## OpenSeal 7B Training Commands

OpenSeal 7B training is run from `olmo-train.pbs` using these commands:

```bash
torchrun --nproc_per_node=8 \
OLMo/scripts/train.py \
OLMo/configs/CPT/7B_n34.8_para-26_replay-8.7.yaml 2>&1 | tee logs/olmo-train_7B_n34.8_para-26_replay-8.7_1.log

torchrun --nproc_per_node=8 \
OLMo/scripts/train.py \
OLMo/configs/CPT/7B_n34.8_para-26_replay-8.7_Decay.yaml 2>&1 | tee logs/olmo-train_7B_n34.8_para-26_replay-8.7_Decay.log
```
