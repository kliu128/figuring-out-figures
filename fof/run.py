from dotenv import load_dotenv

load_dotenv()

import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from encdec import EncoderDecoderModel
from dataloader import ScicapDataModule


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("mode", choices=["train", "validate"])
    parser.add_argument("--exp", default="x")
    parser.add_argument("--model", type=str,
                        default="clip+gpt2", choices=["clip+gpt2", "encdec"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-5)
    # Extract model name from temp args
    temp_args, _ = parser.parse_known_args()

    # let the model add what it wants
    if temp_args.model == "encdec":
        parser = EncoderDecoderModel.add_model_specific_args(parser)

    args = parser.parse_args()
    val_callback = ModelCheckpoint(
        save_top_k=3, mode="min", monitor="val/loss")
    epoch_callback = ModelCheckpoint(
        every_n_epochs=10)

    # wandb_logger = WandbLogger(name=args.exp, project="figuring-out-figures")
    logger = TensorBoardLogger("tb_logs", name=args.exp)
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[val_callback, epoch_callback], logger=logger)

    dict_args = vars(args)
    if args.model == "encdec":
        model = EncoderDecoderModel(**dict_args)

    datamodule = ScicapDataModule(
        "First-Sentence",
        batch_size=args.batch_size,
        limit=args.limit,
        tokenizer=model.text_tokenizer)

    if args.mode == "train":
        trainer.tune(model, datamodule=datamodule)
        trainer.fit(model, datamodule=datamodule)
    elif args.mode == "validate":
        trainer.validate(model, datamodule=datamodule)
    elif args.mode == "test":
        trainer.test(model, datamodule=datamodule)
