import dataloader


def test_dataset():
    dset = dataloader.ScicapDataset("First-Sentence", "train")
    assert len(dset) == 106834
    dset[0]
