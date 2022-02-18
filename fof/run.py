import argparse

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchvision import transforms
from torchtyping import TensorType
from einops import rearrange
from dataloader import ScicapDataModule


class ClipGPT2Model(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        # encode_image: [B x 224 x 224] -> [B x 512]
        self.image_encoder, _ = clip.load("ViT-B/32", jit=False)
        self.image_encoder = self.image_encoder.float()
        model_name = "gpt2"
        # ?? -> []
        self.text_decoder = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # GPT-2 doesn't use padding tokens, so we should just assume EOS.
        # https://github.com/huggingface/transformers/issues/12594
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # map from CLIP 512 embed space to GPT-2 768 embed space
        self.bridge = nn.Linear(512, 768)

        self.lr = 1e-4

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ClipGPT2Model")
        return parent_parser

    def forward(self, figure, labels):
        B = len(labels)

        image_encoding = self.image_encoder.encode_image(figure)
        image_conditioned: TensorType["b", 768] = F.gelu(
            self.bridge(image_encoding))
        image_conditioned = rearrange(image_conditioned, "b e -> b 1 e")
        labels = self.tokenizer(
            list(labels), padding=True, truncation=True, return_tensors="pt")["input_ids"].to(self.device)

        dummy_token = torch.zeros(
            (B, 1), dtype=torch.int64, device=self.device)
        inputs_embeds = torch.cat([image_conditioned,
                                   self.text_decoder.transformer.wte(labels)], dim=1)
        labels = torch.cat([dummy_token, labels], dim=1)

        output = self.text_decoder(
            inputs_embeds=inputs_embeds,
            labels=labels)

        return output["logits"], output["loss"]

    def training_step(self, batch, batch_idx):
        figure, metadata = batch
        y_hat, loss = self(
            figure, metadata)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("mode", choices=["train", "validate"])
    parser.add_argument("--exp", default="x")
    parser.add_argument("--model", type=str,
                        default="clip+gpt2", choices=["clip+gpt2"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    # Extract model name from temp args
    temp_args, _ = parser.parse_known_args()

    # let the model add what it wants
    if temp_args.model == "clip+gpt2":
        parser = ClipGPT2Model.add_model_specific_args(parser)

    args = parser.parse_args()
    val_callback = ModelCheckpoint(
        save_top_k=3, mode="min", monitor="val_loss")
    epoch_callback = ModelCheckpoint(
        every_n_epochs=10)
    logger = TensorBoardLogger("tb_logs", name=args.exp)
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[val_callback, epoch_callback], logger=logger)

    _, image_preprocessor = clip.load("ViT-B/32")

    dict_args = vars(args)
    if args.model == "clip+gpt2":
        model = ClipGPT2Model(**dict_args)

    datamodule = ScicapDataModule(
        "First-Sentence", transform=transforms.Compose([
            # TODO This converts it into and out of PIL, inefficient
            transforms.ToPILImage(),
            image_preprocessor]),
        batch_size=args.batch_size,
        limit=args.limit)

    if args.mode == "train":
        trainer.tune(model, datamodule=datamodule)
        trainer.fit(model, datamodule=datamodule)
    elif args.mode == "validate":
        trainer.validate(model, datamodule=datamodule)
    elif args.mode == "test":
        trainer.test(model, datamodule=datamodule)
