import os
import json
from pathlib import Path
from typing import Callable, Tuple

import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image
from torchtyping import TensorType


class ScicapDataset(Dataset):
    def __init__(self, experiment: str, split: str, transform: Callable = None):
        self.transform = transform

        root = Path("./scicap_data")
        self.metadata_dir = root / "SciCap-Caption-All" / split
        self.image_dir = root / "SciCap-No-Subfig-Img" / split

        file_idx = root / "List-of-Files-for-Each-Experiments" / \
            experiment / "No-Subfig" / split / "file_idx.json"

        with open(file_idx) as f:
            self.metadata_files = json.load(f)
        # We want metadata files, not images.
        self.metadata_files = [name.replace(
            ".png", ".json") for name in self.metadata_files]

    def __len__(self):
        return len(self.metadata_files)

    def __getitem__(self, idx) -> Tuple[TensorType[3, "height", "width"], dict]:
        with open(self.metadata_dir / self.metadata_files[idx]) as f:
            metadata = json.load(f)
        figure = read_image(str(self.image_dir / metadata["figure-ID"]))

        if self.transform:
            figure = self.transform(figure)
        return figure, metadata["2-normalized"]["2-1-basic-num"]["caption"]


class ScicapDataModule(pl.LightningDataModule):
    def __init__(self, experiment: str, transform, batch_size: int = 32):
        super().__init__()
        self.train_dset = ScicapDataset(experiment, "train", transform)
        self.test_dset = ScicapDataset(experiment, "test", transform)
        self.val_dset = ScicapDataset(experiment, "val", transform)
        self.batch_size = batch_size
        assert len(self.train_dset) > len(self.test_dset) and \
            len(self.train_dset) > len(self.test_dset)

    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dset, batch_size=self.batch_size)
