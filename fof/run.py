#autopep8: off
import sys; sys.path.append("./")
#autopep8: on
from dotenv import load_dotenv
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from fof.encdec import EncoderDecoderModel
from fof.deit_gpt import DeitGPTModel
from fof.dataloader import ScicapDataModule
from typing import List


load_dotenv()


def get_parser(args: List[str] = None):
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("mode", choices=["train", "validate"])
    parser.add_argument("--exp", default="x")
    parser.add_argument("--model", type=str,
                        default="clip+gpt2", choices=["clip+gpt2", "encdec", "deit_gpt"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--caption_type", type=str, default="orig")
    parser.add_argument("--pl_logger", type=str,
                        choices=["wandb", "tb"], default="wandb")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    # Extract model name from temp args
    temp_args, _ = parser.parse_known_args(args)

    # let the model add what it wants
    if temp_args.model == "encdec":
        parser = EncoderDecoderModel.add_model_specific_args(parser)
    elif temp_args.model == "deit_gpt":
        parser = DeitGPTModel.add_model_specific_args(parser)

    return parser


def main(args):
    val_callback = ModelCheckpoint(
        save_top_k=3, mode="min", monitor="val/loss")
    epoch_callback = ModelCheckpoint(
        every_n_epochs=10)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    swa_callback = pl.callbacks.StochasticWeightAveraging()

    if args.pl_logger == "wandb":
        logger = WandbLogger(
            name=args.exp, project="figuring-out-figures", log_model=True)
    if args.pl_logger == "tb":
        logger = TensorBoardLogger("tb_logs", name=args.exp)
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[val_callback, epoch_callback, lr_monitor, swa_callback],
        logger=logger)

    dict_args = vars(args)
    if args.model == "encdec":
        model = EncoderDecoderModel(**dict_args)
    elif args.model == "deit_gpt":
        model = DeitGPTModel(**dict_args)

    datamodule = ScicapDataModule(
        "First-Sentence",
        batch_size=args.batch_size,
        limit=args.limit,
        tokenizer=model.text_tokenizer,
        caption_type=args.caption_type)

    if args.mode == "train":
        trainer.tune(model, datamodule=datamodule)
        trainer.fit(model, datamodule=datamodule)
    elif args.mode == "validate":
        trainer.validate(model, datamodule=datamodule,
                         ckpt_path=args.load_checkpoint)
    elif args.mode == "test":
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
