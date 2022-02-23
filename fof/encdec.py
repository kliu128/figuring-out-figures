from typing import List
import pytorch_lightning as pl
import transformers as tr
import torch
import torch.nn as nn
from datasets import load_metric


class ExtensibleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = tr.CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32")
        self.config = self.clip.config
        self.main_input_name = self.clip.main_input_name

    def forward(self, *args, **kwargs):
        return self.clip(*args, **kwargs)


class EncoderDecoderModel(pl.LightningModule):
    def __init__(self, lr: int, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        encoder = ExtensibleEncoder()
        gpt2 = tr.AutoModelForCausalLM.from_pretrained(
            "distilgpt2", add_cross_attention=True)

        model = tr.VisionEncoderDecoderModel(
            encoder=encoder.clip, decoder=gpt2)
        model.encoder = encoder
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
        self.text_tokenizer = tr.AutoTokenizer.from_pretrained("distilgpt2")
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        # breakpoint()

        self.lr = lr

        self.bleu_metric = load_metric('bleu')
        self.rouge_metric = load_metric('rouge')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("EncDecModel")
        return parent_parser

    def preprocess_image(self, figure):
        # images = torch.split(figure, split_size_or_sections=1)
        # squeezed_tensors = list(map(torch.squeeze, images))
        # image = self.image_processor(images=squeezed_tensors)
        # image = torch.stack(image["pixel_values"])
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
        self.log("val/loss", output.loss)
        # Use sampling to generate sentences
        generated = self.model.generate(
            image, return_dict_in_generate=True, do_sample=True,
            bos_token_id=self.text_tokenizer.bos_token_id, eos_token_id=self.text_tokenizer.eos_token_id)
        decoded: List[str] = self.text_tokenizer.batch_decode(
            generated.sequences, skip_special_tokens=True)
        # PROCESSING FOR BLEU SCORE
        # model_predictions.shape = [2, 44]
        # for bleu_metric.compute input
        labels: List[str] = self.text_tokenizer.batch_decode(
            labels, skip_special_tokens=True)
        tokenized_labels = [[label.split()] for label in labels]
        model_predictions = [decode.split() for decode in decoded]
        try:
            bleu_score = self.bleu_metric.compute(
                predictions=model_predictions, references=tokenized_labels)
            self.log('val/bleu_score', bleu_score['bleu'])
        except Exception as e:
            print(e)

        try:
            # PROCESSING FOR ROUGE SCORE
            rouge_score = self.rouge_metric.compute(
                predictions=decoded, references=labels)
            # LOGGING
            # TODO: what rouge score do we want to log? Use print(self.rouge_metric)
            # to see manual
            self.log('val/rouge_score', rouge_score['rouge1'].mid.fmeasure)
        except Exception as e:
            print(e)

        return output.loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
