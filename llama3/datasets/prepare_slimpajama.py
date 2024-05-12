import sys
import os

# deal problem with tokenizer import
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(script_directory)
if parent_directory not in sys.path:
    sys.path.append(parent_directory)

import zstandard as zstd
from glob import glob
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import torch

from llama.tokenizer import Tokenizer


def process(params) -> None:
    # unpack all parameters
    i, file_path, tokenizer_model, output_path, dtype = params
    tokenizer = Tokenizer(tokenizer_model)
    with zstd.open(open(file_path, 'rb'), 'rt', encoding='utf-8') as f:
        result = []
        for row in f:
            text = json.loads(row)["text"]
            text_ids = torch.tensor(tokenizer.encode(text, eos=True, bos=True), dtype=dtype)
            result.append(text_ids)
    # save results
    output_filename = os.path.join(output_path, f"data{i}.pt")
    torch.save(result, output_filename)


def prepare(dataset_path: str = 'SlimPajama-627B',
            output_path: str = 'tokenized/SlimPajama-627B',
            tokenizer_model: str = '../models/tokenizer.model',
            dtype: torch.dtype = torch.int32) -> None:
    # check if output path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # serch for all .zst data file in dataset path
    filenames = glob(f'{dataset_path}/**/*.zst', recursive=True)
    # start multi-processing
    pool = Pool(processes=cpu_count())
    params = []
    # append data file after existing files
    files = glob(f'{output_path}/*.pt', recursive=True)
    for i, filename in enumerate(filenames):
        idx = i + len(files)
        params.append((idx, filename, tokenizer_model, output_path, dtype))
    # Use tqdm with imap for progress bar
    for _ in tqdm(pool.imap(process, params), total=len(filenames), desc="Processing Files"):
        pass
    pool.close()
    pool.join()


if __name__ == '__main__':
    prepare()
