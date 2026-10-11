import argparse
import gzip
import json
import os
import time

from multiprocessing import Process, cpu_count

import numpy as np


def main(args):
    print(args)
    filepaths = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) \
                    if os.path.isfile(os.path.join(args.data_dir, f))]
    
    if args.metadata is not None:
        try:
            metadata = json.loads(args.metadata)
            result['metadata'] = metadata
        except Exception as e:
            print("[WARNING] Failed to add metadata due to: {}".format(e))
    else:
        metadata = None
    
    if args.num_worker == -1:
        num_worker = cpu_count()
    else:
        num_worker = args.num_worker
    chunked_filepaths = np.array_split(filepaths, num_worker)

    processes = []
    start_time = time.time()
    for i, subset in enumerate(chunked_filepaths):
        p = Process(target=process,
                    args=(subset, args.source_name, args.out_dir, args.eos_replacement, metadata))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
    time.sleep(60)


def process(filepaths, source_name, out_dir, eos_replacement, metadata=None):
    for filepath in filepaths:
        filename, _ext = os.path.splitext(os.path.basename(filepath))
        try:
            with open(filepath, encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print('[ERROR] failed to open {} due to {}'.format(filepath, e), flush=True)
            continue
        
        if eos_replacement is not None:
            text = text.replace("</s>", eos_replacement)
        result = {
            'id': filename,
            'text': text,
            'source': source_name,
        }
        if metadata is not None:
            result['metadata'] = metadata
        
        out_path = os.path.join(out_dir, filename + '.jsonl.gz')
        with gzip.open(out_path, 'wb') as out:
            out.write(json.dumps(result).encode())
        

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to the data directory.')
    parser.add_argument('--out_dir', required=True, help='Path to the output directory.')
    parser.add_argument('--source_name', type=str, required=True,
        help='Name of the data source for the meta data.')
    parser.add_argument('--eos_replacement', type=str, default=None,
        help='Specify the new EOS string to replace </s> (if any).')
    parser.add_argument('--metadata', type=str, default=None,
        help='Optional meta-data in JSON string format.')
    parser.add_argument('--num_worker', type=int, default=-1,
        help="Number of parallel worker.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
