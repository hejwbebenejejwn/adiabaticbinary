import math
import os
import random
import sys
from pathlib import Path

import fire

# deal problem with tokenizer import
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(script_directory)
if parent_directory not in sys.path:
    sys.path.append(parent_directory)

from glob import glob
import json
from tqdm import tqdm
from torch.multiprocessing import Pool, cpu_count
import torch

from config import Config
from utils import Tokenizer


def process(params) -> None:
    tokenizer, rows, dtype, output_path, chunk_id, file_idx = params
    out = [tokenizer.encode(row["text"], eos=True, bos=True).to(dtype=dtype) for row in rows]

    # Save the result with a filename that includes the process ID and chunk ID
    output_filename = os.path.join(output_path, f"data_{file_idx}_chunk_{chunk_id}.pt")
    torch.save(out, output_filename)


def prepare(dataset_path: str = Config.dataset_dir,
            output_path: str = Config.tokenized_dataset_dir,
            tokenizer_model: str = Path(Config.full_ckpt_dir),
            dtype: torch.dtype = torch.int32) -> None:
    # check if output path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # serch for all .json data file in dataset path
    filenames = glob(f'{dataset_path}/*.json', recursive=True)
    files = glob(f'{output_path}/*.pt', recursive=True)
    tokenizer = Tokenizer(tokenizer_model)

    # start multi-processing
    pool = Pool(processes=cpu_count())
    for i, filename in enumerate(filenames):
        idx = i + len(files)
        with open(filename) as fp:
            text = json.load(fp)

        random.shuffle(text)

        # Chunk the text data
        chunk_size = math.ceil(len(text) / cpu_count())
        params = [(tokenizer, text[j:j + chunk_size], dtype, output_path, chunk_id, idx)
                  for chunk_id, j in enumerate(range(0, len(text), chunk_size))]

        # process the text in chunks
        for _ in tqdm(pool.imap(process, params), total=len(params), desc=f"Processing File {i}"):
            pass

    pool.close()
    pool.join()

    print("Processed data saved in:", output_path)


if __name__ == '__main__':
    fire.Fire(prepare)
