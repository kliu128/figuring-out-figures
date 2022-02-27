from typing import List
import pytorch_lightning as pl
import transformers as tr
import torch
import torch.nn as nn
from datasets import load_metric


class ExtensibleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deit = tr.DeiTModel.from_pretrained(
            "facebook/deit-base-distilled-patch16-224")
        self.config = self.deit.config
        self.main_input_name = self.deit.main_input_name

    def forward(self, *args, **kwargs):
        return self.deit(*args, **kwargs)


class DeitGPTModel(pl.LightningModule):
    def __init__(self, lr: float = 5e-5, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        text_model = "gpt2"
        encoder = ExtensibleEncoder()
        decoder = tr.AutoModelForCausalLM.from_pretrained(
            text_model, add_cross_attention=True)

        model = tr.VisionEncoderDecoderModel(
            encoder=encoder.deit, decoder=decoder)
        # model.encoder = encoder
        # use GPT2's eos_token as the pad as well as eos token
        # TODO is this line correct?
        model.config.decoder_start_token_id = model.config.decoder.bos_token_id
        model.config.eos_token_id = model.config.decoder.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

        self.model = model
        self.image_processor = tr.CLIPFeatureExtractor(
            # Skip resize since the datamodule already resized it
            do_resize=False,
            do_center_crop=False,
        )
        self.text_tokenizer = tr.AutoTokenizer.from_pretrained(text_model)
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        self.lr = lr

        # Use sacrebleu as a standard BLEU computer.
        self.bleu_metric = load_metric('sacrebleu')
        self.rouge_metric = load_metric('rouge')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DeitGPTModel")
        # parser.add_argument("--text_model", type=str, default="distilgpt2")
        return parent_parser

    def preprocess_image(self, figure):
        return figure

    def forward(self, image, labels):
        output = self.model(
            pixel_values=image,
            # Decoder input ids and attention masks are automatically generated
            # by shifting the input ids to the right and adding a start token
            # for causal LM, e.g.
            # inputs: <start> A B C D
            # labels: A       B C D <end>
            labels=labels,
        )
        return output

    def training_step(self, batch, batch_idx: int):
        figure, metadata = batch["figure"], batch["labels"]
        image = self.preprocess_image(figure)
        output = self(image, metadata)
        self.log("train/loss", output.loss)
        self.log("train/perplexity", torch.exp(output.loss))

        # Print samples for debugging
        # generated = self.model.generate(
        #     image.to(self.device), return_dict_in_generate=True, do_sample=True,
        #     bos_token_id=self.text_tokenizer.bos_token_id, eos_token_id=self.text_tokenizer.eos_token_id)
        # decoded: List[str] = self.text_tokenizer.batch_decode(
        #     generated.sequences, skip_special_tokens=True)
        # print(metadata[0], "<->", decoded[0])

        return output.loss

    def validation_step(self, batch, batch_idx: int):
        figure, labels = batch["figure"], batch["labels"]
        image = self.preprocess_image(figure)
        output = self(image, labels)
        # Use sampling to generate sentences
        generated = self.model.generate(
            image, return_dict_in_generate=True, do_sample=True,
            bos_token_id=self.text_tokenizer.bos_token_id, eos_token_id=self.text_tokenizer.eos_token_id)
        decoded: List[str] = self.text_tokenizer.batch_decode(
            generated.sequences, skip_special_tokens=True)
        labels: List[str] = self.text_tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        # Compute metrics (queue batch to compute metrics later)
        self.bleu_metric.add_batch(predictions=decoded, references=[
                                   [label] for label in labels])
        self.rouge_metric.add_batch(predictions=decoded, references=labels)

        # Logs average val loss
        self.log("val/loss", output.loss)
        self.log("val/perplexity", torch.exp(output.loss))

        return output.loss

    def validation_epoch_end(self, outputs):
        # Compute over all batches
        self.log("val/bleu_score",
                 self.bleu_metric.compute(lowercase=True)['score'])
        self.log("val/rouge_score", self.rouge_metric.compute()
                 ['rouge1'].mid.fmeasure)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss"
        }
