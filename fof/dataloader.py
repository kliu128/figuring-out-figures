import os
import json
from pathlib import Path
from typing import Callable, Tuple

import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image
from torchtyping import TensorType
from torchvision import transforms
import transformers as tr
import torch


class ScicapDataset(Dataset):
    def __init__(self,
                 experiment: str,
                 split: str,
                 transform: Callable,
                 limit: int = None,
                 tokenizer=None,
                 caption_type="orig"):
        self.transform = transform
        self.limit = limit
        self.tokenizer = tokenizer
        self.caption_type = caption_type

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
        if self.limit is None:
            return len(self.metadata_files)
        else:
            return min(self.limit, len(self.metadata_files))

    def __getitem__(self, idx) -> Tuple[TensorType[3, "height", "width"], dict]:
        with open(self.metadata_dir / self.metadata_files[idx]) as f:
            metadata = json.load(f)
        figure = read_image(
            str(self.image_dir / metadata["figure-ID"])).to(dtype=torch.float)

        if self.transform:
            figure = self.transform(figure)

        if self.caption_type == "orig":
            caption = metadata["0-originally-extracted"]
        elif self.caption_type == "normalized":
            caption = metadata["2-normalized"]["2-2-advanced-euqation-bracket"]["caption"]

        return {
            "figure": figure,
            "labels": caption,
        }


class ScicapDataModule(pl.LightningDataModule):
    def __init__(
            self,
            experiment: str,
            tokenizer,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]),
            batch_size: int = 32,
            limit: int = None, **kwargs):
        super().__init__()
        self.train_dset = ScicapDataset(
            experiment, "train", transform, limit, tokenizer, **kwargs)
        self.test_dset = ScicapDataset(
            experiment, "test", transform, limit, tokenizer, **kwargs)
        self.val_dset = ScicapDataset(
            experiment, "val", transform, limit, tokenizer, **kwargs)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.batch_size, num_workers=32, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dset, batch_size=self.batch_size, num_workers=32, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dset, batch_size=self.batch_size, num_workers=32, pin_memory=True)
