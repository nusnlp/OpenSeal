import sys

import torch

from olmo.data import build_train_dataloader
from olmo.torch_util import (
    SingleAccelerator,
    barrier,
    get_default_device,
    get_global_rank,
    get_local_rank,
    get_local_world_size,
    get_world_size,
    # peak_gpu_memory,
    seed_all,
)
from olmo.config import (
    # CheckpointType,
    # DDPGradSyncMode,
    # DistributedStrategy,
    TrainConfig,
)
from olmo.util import (
    # add_cached_path_clients,
    clean_opt,
)
from transformers import AutoTokenizer


def main(cfg: TrainConfig) -> None:
    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None  # for mypy
    
    # Set seed.
    seed_all(cfg.seed)

    # Construct data loader.
    train_loader = build_train_dataloader(cfg)
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")

    # print('Data len:', len(train_loader))
    for batch in train_loader:
        data = batch["input_ids"]
        print('|#|'.join(tokenizer.batch_decode(data, skip_special_tokens=False)))
        print('\n')

if __name__ == "__main__":
    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    if torch.backends.mps.is_available():
        log.info("Device is MPS. Updating config...")
        cfg.model.init_device = "mps"
        cfg.distributed_strategy = "single"  # type: ignore

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        log.info("Device is CPU. Updating config...")
        cfg.model.init_device = "cpu"
        cfg.distributed_strategy = "single"  # type: ignore
    main(cfg)