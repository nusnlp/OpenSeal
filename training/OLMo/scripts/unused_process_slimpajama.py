# Very loosely inspired by indexed_dataset in Fairseq, Megatron
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py

import math
import os
import random
import struct
import glob
import sys
import time

from pathlib import Path
from typing import List

from multiprocessing import Process, cpu_count

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


from transformers import AutoTokenizer
from tokenizers import Tokenizer as HFTokenizer

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from lit_gpt.model import GPT, Block, Config
import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Tokenizer

dtypes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float32, 7: np.float64, 8: np.uint16}


def code(dtype):
    for k in dtypes:
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes

def _read_header(path):
    with open(path, "rb") as f:
        magic = f.read(len(HDR_MAGIC))
        assert magic == HDR_MAGIC, "File doesn't match expected format."
        version = struct.unpack("<Q", f.read(8))
        assert version == (1,)
        (dtype_code,) = struct.unpack("<B", f.read(1))
        dtype = dtypes[dtype_code]
        (chunk_size,) = struct.unpack("<Q", f.read(8))
    return dtype, chunk_size

def _load_n_chunks(self):
    self._close_mmaps()
    self._mmaps = []
    self._buffers = []

    # if self._n_chunks > len(self._filenames[self._file_idx :]):
    #     # if not self._wrap:
    #     #     raise StopIteration
    #     self._file_idx = 0

    last_chunk_size = -1
    n_all_blocks = 0
    for i in range(self._n_chunks):
        # filename = self._filenames[self._file_idx + i]
        filename = self._filenames[self._file_idx]
        # if self._dtype is None:
        mmap = _load_one_chunk(filename)
        self._mmaps.append(mmap)
        self._buffers.append(memoryview(mmap))

def _load_one_chunk(filename):
        _dtype, _chunk_size = _read_header(filename)
        # TODO: check header matches with previous files
        return np.memmap(filename, mode="r", order="C", offset=HDR_SIZE), _dtype, _chunk_size


def prepare_full(
    data: List[str],
    tokenizer,
    destination_path: Path,
    chunk_size: int,
    out_filename: str,
) -> None:
    destination_path.mkdir(parents=True, exist_ok=True)

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=out_filename,
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )
    total_tokens = 0
    for idx, text in enumerate(data):
        if idx % 1000 == 0:
            print('> [{}] generated text:\n{}'.format(idx, text), flush=True)
        text_ids = tokenizer.encode(text)
        total_tokens += len(text_ids)
        builder.add_array(np.array(text_ids, dtype=builder.dtype))
    print('>> Finished writing {} tokens to {}. Probably wasted {} tokens.'.format(total_tokens, destination_path, total_tokens % 2049), flush=True)
    # we throw away the final corpus to avoid meaningless corpus filled with bos_ids, see https://github.com/jzhang38/TinyLlama/issues/83 for more details
    # builder.write_reminder()
    return total_tokens



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main(filepath: Path,
         include_english: bool,
         destination_path: Path,
         chunk_size: int = 2049 * 8,
         vocabulary_path: Path = Path("data/TinyLlama_v1.1_tokenizer/"),
        ) -> None:
    model_name = "tiny_LLaMA_1b"
    tokenizer = AutoTokenizer.from_pretrained(vocabulary_path)
    writer_tokenizer = Tokenizer(vocabulary_path)

    config = Config.from_name(model_name)
    parallel_files = sorted(glob.glob(str(filepath / f"parallel*")))
    file_tree = {}

    for f in parallel_files:
        prefix, number_ext = f.rsplit('_', 1)
        prefix = os.path.basename(prefix)
        if prefix not in file_tree:
            file_tree[prefix] = []
        number, _ext = number_ext.split('.', 1)
        file_tree[prefix].append((int(number), f))
    for prefix in file_tree.keys():
        file_tree[prefix] = sorted(file_tree[prefix])

    processes = []
    start_time = time.time()

    num_processes = cpu_count() 
    chunked_filenames = np.array_split(list(file_tree.items()), num_processes)
    # for prefix, file_infos in :
    #     process(prefix, file_infos)
    for i, subset in enumerate(chunked_filenames):
        p = Process(target=process_file, args=(subset, include_english, config, tokenizer, writer_tokenizer, destination_path, chunk_size))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

def process_file(orig_data, include_english, config, tokenizer, writer_tokenizer, destination_path, chunk_size):
    for prefix, file_infos in orig_data:
        data = {}
        total_lengths = {}
        total_tokens = 0
        last_lang = None
        text = ""
        _block_size=config.block_size
        num_chunk = len(file_infos)
        for file_idx, (_num, p) in enumerate(file_infos):
            _buffers = []
            mmap, _dtype, _chunk_size = _load_one_chunk(p)
            # arr = memoryview(mmap)
            _buffers.append(memoryview(mmap))
            _n_blocks = _chunk_size // _block_size
            n_all_blocks = _n_blocks
            offset = 0
            # offset = np.dtype(_dtype).itemsize * elem_id
            # arr = np.frombuffer(arr, dtype=_dtype, count=_block_size, offset=offset)
            # self._curr_idx += 1

            _block_idxs = range(n_all_blocks)
            # for _curr_idx in range(len(_block_idxs)):
            #     block_idx = _block_idxs[_curr_idx]
            for block_idx in _block_idxs:
                elem_id = (block_idx % _n_blocks) * _block_size
                offset = np.dtype(_dtype).itemsize * elem_id
                arr = np.frombuffer(_buffers[0], dtype=_dtype, count=_block_size, offset=offset)
            
                tensor = torch.from_numpy(arr.astype(np.int64))
                cur_length = tensor.numel()
                total_tokens += cur_length
                # targets = tensor[:, 0 : config.block_size + 1].contiguous()
                # text = tokenizer.batch_decode(tensor.contiguous())
                cur_text = tokenizer.decode(tensor.contiguous())
                # print('cur_text start:{}| non-whitespace start? {}'.format(cur_text[:5], cur_text[:5] == cur_text[:5].lstrip()))
                # print('cur_text ends:{}| non-whitespace end?{}'.format(cur_text[-5:], cur_text[-5:] == cur_text[-5:].rstrip()))
                text += cur_text
        
        
        
        num_all_tokens = prepare_full(all_texts, writer_tokenizer, destination_path, chunk_size, prefix)
        # return num_all_tokens
        # print('Final token count:', num_all_tokens)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)