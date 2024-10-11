import os
from glob import glob
from multiprocessing import Pool, cpu_count

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from tqdm import tqdm


def load_data(file):
    """Load data from a file and return the length of entries."""
    entries = torch.load(file)
    return len(entries)


def calculate_index_map(file_length_pair):
    file_index, length = file_length_pair
    return [(file_index, entry_index) for entry_index in range(length)]


class TokenizedDataset(Dataset):
    def __init__(
            self,
            tokenized_dataset_path='tokenized/SlimPajama-627B',
            validation_dataset_items: int = 8000,
            test_dataset_items: int = 80000,
            mode='training',
            shuffle=True,
            max_seq_len: int = 512,
            pad_id: int = -1
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.files = glob(f'{tokenized_dataset_path}/*.pt', recursive=True)

        # make/load index map
        os.makedirs(f'{tokenized_dataset_path}/index_map', exist_ok=True)
        index_map_path = f'{tokenized_dataset_path}/index_map/index_map.pt'
        if os.path.exists(index_map_path):
            self.index_map = torch.load(index_map_path)
        else:
            local_rank = int(os.environ["LOCAL_RANK"])
            if local_rank == 0:
                # Using multiprocessing to load data and measure lengths
                with Pool(processes=cpu_count()) as pool:
                    lengths = list(tqdm(pool.imap(load_data, self.files), total=len(self.files), desc=f"Indexing Data"))
                # Calculate index map
                file_length_pairs = [(file_index, length) for file_index, length in enumerate(lengths)]
                with Pool(processes=cpu_count()) as pool:
                    index_map_parts = list(tqdm(pool.imap(calculate_index_map, file_length_pairs),
                                                total=len(file_length_pairs), desc="Calculating Index Map"))
                self.index_map = [item for sublist in index_map_parts for item in sublist]
                if shuffle:
                    self.index_map = torch.tensor(self.index_map)[torch.randperm(len(self.index_map))]
                torch.save(self.index_map, index_map_path)
            dist.barrier()
            if local_rank != 0:
                self.index_map = torch.load(index_map_path)
        dist.barrier()

        # split dataset
        index_map_len = len(self.index_map)

        # check if there are enough items for split
        if validation_dataset_items + test_dataset_items > index_map_len:
            raise ValueError("Not enough items in the dataset for the specified validation and test sizes")

        if shuffle:
            entry_ids = torch.randperm(index_map_len)
        else:
            entry_ids = torch.arange(index_map_len)

        valid_ids = entry_ids[:validation_dataset_items]
        test_ids = entry_ids[validation_dataset_items:validation_dataset_items + test_dataset_items]
        train_ids = entry_ids[validation_dataset_items + test_dataset_items:]

        if mode == 'training':
            self.index_map = self.index_map[train_ids]
        elif mode == 'validation':
            self.index_map = self.index_map[valid_ids]
        elif mode == 'test':
            self.index_map = self.index_map[test_ids]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_index, entry_index = self.index_map[idx]
        file_path = self.files[file_index]
        data = torch.load(file_path)[entry_index]
        data_padded = torch.full([self.max_seq_len], self.pad_id, device=data.device, dtype=data.dtype)
        data_padded[:data.size()[0]] = data[:self.max_seq_len]
        return data_padded
