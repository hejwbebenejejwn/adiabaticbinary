import os
from glob import glob
from multiprocessing import Pool, cpu_count

import torch
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
            dataset_ratio=None,
            mode='training',
            shuffle=True,
            max_seq_len: int = 512,
            pad_id: int = -1
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.files = glob(f'{tokenized_dataset_path}/*.pt', recursive=True)

        # check dataset ratio
        if dataset_ratio is None:
            dataset_ratio = [0.9, 0.05, 0.05]
        if sum(dataset_ratio) > 1:
            dataset_ratio = torch.tensor(dataset_ratio) / sum(dataset_ratio)

        # make/load index map
        os.makedirs(f'{tokenized_dataset_path}/index_map', exist_ok=True)
        index_map_path = f'{tokenized_dataset_path}/index_map/index_map.pt'
        if os.path.exists(index_map_path):
            self.index_map = torch.load(index_map_path)
        else:
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

        # split dataset
        entry_num = len(self.index_map)
        train_end = int(entry_num * dataset_ratio[0])
        valid_end = train_end + int(entry_num * dataset_ratio[1])
        if mode == 'training':
            self.index_map = self.index_map[:train_end]
        elif mode == 'validation':
            self.index_map = self.index_map[train_end:valid_end]
        elif mode == 'test':
            self.index_map = self.index_map[valid_end:]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_index, entry_index = self.index_map[idx]
        file_path = self.files[file_index]
        data = torch.load(file_path)[entry_index]
        data_padded = torch.full([self.max_seq_len], self.pad_id, device=data.device, dtype=data.dtype)
        data_padded[:data.size()[0]] = data[:self.max_seq_len]
        return data_padded
