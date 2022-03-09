import torch
from torchvision import transforms
from torchtyping import TensorType
from torchvision.io import read_image
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from typing import Callable, Tuple, Dict
from pathlib import Path
import json
import time
from tqdm import tqdm

import os


class ScicapDataset(Dataset):
    def __init__(self,
                 experiment: str,
                 split: str,
                 transform: Callable,
                 limit: int = None,
                 tokenizer=None,
                 caption_type="orig",
                 root=Path("./scicap_data")):
        self.transform = transform
        self.limit = limit
        self.tokenizer = tokenizer
        self.caption_type = caption_type
        # split is 'train', 'test', or 'val'
        self.metadata_dir = root / "SciCap-Caption-All" / split  # every figure caption
        self.image_dir = root / "SciCap-No-Subfig-Img" / split

        # Contains all json with all file names of figures with no subfigures
        file_idx = root / "List-of-Files-for-Each-Experiments" / \
            experiment / "No-Subfig" / split / "file_idx.json"

        # Get all file names
        with open(file_idx) as f:
            self.metadata_files = json.load(f)

        # We want metadata files, not images.
        self.metadata_files = [name.replace(  # contains all names of figures with no subfigures
            ".png", ".json") for name in self.metadata_files]

        # Get actual metadata from the papers (i.e., abstracts and titles)
        self.paper_metadata_file = root / 'arxiv-metadata-oai-snapshot.json'
        # Directory to find jsons
        self.paper_metadata_json_dir = root / 'metadata'

        self.references_json_dir = root / 'references'

        # self.paper_metadata_id_to_json = None

        if not self.paper_metadata_json_dir.is_dir():
            os.mkdir(self.paper_metadata_json_dir)
            print('Creating JSON files...')
            with open(self.paper_metadata_file) as f:
                # self.paper_metadata_id_to_json = {}
                start = time.time()
                lines = tqdm(f.readlines(), unit='MB')
                end = time.time()
                for line in lines:
                    js = json.loads(line)  # load string
                    id = js['id']
                    if '/' in id:  # faulty ID, ignore any ids that would required subdirectories
                        continue
                    with open(self.paper_metadata_json_dir / (id + '.json'), 'w') as newJSON:
                        json.dump(js, newJSON)
                loop_end = time.time()
            print(
                f'Reading time = {end - start}; Loop time = {loop_end - end}')
        #         # self.paper_metadata_id_to_json[id] = js

        # if self.paper_metadata_json_file.is_file():
        #     print('JSON file for metadata detected! Loading...')
        #     with open(self.paper_metadata_json_file, 'r') as f:
        #         self.paper_metadata_id_to_json = json.load(f)
        # else:
        #     print('JSON file for metadata not detected. Gathering metadata...')
        #     with open(self.paper_metadata_file) as f:
        #         self.paper_metadata_id_to_json = {}
        #         for line in f:
        #             js = json.loads(line)  # load string
        #             id = js['id']
        #             self.paper_metadata_id_to_json[id] = js
        #     with open(self.paper_metadata_json_file, 'w') as f:
        #         json.dump(self.paper_metadata_id_to_json, f)
        # assert self.paper_metadata_id_to_json is not None

        # Example entry (key is the figure ID, value is the below dict)
        # 'abstract': LOTS OF TEXT
        # 'authors': 'P. Papadimitratos and A. Jovanovic',
        # 'authors_parsed': [['Papadimitratos', 'P.', ''], ['Jovanovic', 'A.', '']],
        # 'categories': 'cs.CR',
        # 'comments': None,
        # 'doi': None,
        # 'id': '1001.0025',
        # 'journal-ref': 'IEEE MILCOM, San Diego, CA, USA, November 2008',
        # 'license': 'http://arxiv.org/licenses/nonexclusive-distrib/1.0/',
        # 'report-no': None,
        # 'submitter': 'Panos Papadimitratos',
        # 'title': 'GNSS-based positioning: Attacks and Countermeasures',
        # 'update_date': '2010-01-05',
        # 'versions': [{'created': 'Wed, 30 Dec 2009 22:13:59 GMT', 'version': 'v1'}]}

    def __len__(self):
        if self.limit is None:
            return len(self.metadata_files)
        else:
            return min(self.limit, len(self.metadata_files))

    def __getitem__(self, idx) -> Tuple[TensorType[3, "height", "width"], Dict]:
        with open(self.metadata_dir / self.metadata_files[idx]) as f:
            metadata = json.load(f)
        figure = read_image(
            str(self.image_dir / metadata["figure-ID"])).to(dtype=torch.float)

        # shave off version number e.g., 'v1'
        figure_id = metadata['paper-ID'][:-2]
        # Check if shaved off the right thing
        assert 'v' not in figure_id and metadata['paper-ID'][-2] == 'v'

        with open(self.paper_metadata_json_dir / (figure_id + '.json')) as f:
            js = json.load(f)
            abstract = js['abstract']
            title = js['title']

        try:
            with (self.references_json_dir / (metadata['figure-ID'].replace(".png", "") + '.json')).open() as f:
                references = json.load(f)["references"]
        except json.JSONDecodeError:
            print("Failed to load references")
            references = []

        # Trim off the 200x200 window of each reference to 100x100
        references = [ref[100:-100] for ref in references]
        references = "[SEP]".join(references)

        if self.transform:
            figure = self.transform(figure)

        if self.caption_type == "orig":
            caption = metadata["0-originally-extracted"]
        elif self.caption_type == "normalized":
            caption = metadata["2-normalized"]["2-2-advanced-euqation-bracket"]["caption"]

        return {
            "id": metadata["figure-ID"],
            "figure": figure,
            'abstract': abstract,
            'title': title,
            "labels": caption,
            "references": references
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
            limit: int = None,
            num_workers: int = 32,
            **kwargs):
        super().__init__()

        start = time.time()
        print('Initializing SCICAP training dataset')
        self.train_dset = ScicapDataset(
            experiment, "train", transform, limit, tokenizer, **kwargs)

        print('Initializing SCICAP testing dataset')
        self.test_dset = ScicapDataset(
            experiment, "test", transform, limit, tokenizer, **kwargs)

        print('Initializing SCICAP validation dataset')
        self.val_dset = ScicapDataset(
            experiment, "val", transform, limit, tokenizer, **kwargs)

        print(f'Time taken: {time.time() - start}')
        self.dataloader_args = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
            "shuffle": True
        }

    def train_dataloader(self):
        return DataLoader(self.train_dset, **self.dataloader_args)

    def val_dataloader(self):
        return DataLoader(self.val_dset, **self.dataloader_args)

    def test_dataloader(self):
        return DataLoader(self.test_dset, **self.dataloader_args)
