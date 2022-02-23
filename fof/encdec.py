from typing import List
import pytorch_lightning as pl
import transformers as tr
import torch
import torch.nn as nn


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
    def __init__(self, **kwargs):
        super().__init__()
        encoder = ExtensibleEncoder()
        gpt2 = tr.AutoModelForCausalLM.from_pretrained(
            "gpt2", add_cross_attention=True)

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
        self.text_tokenizer = tr.AutoTokenizer.from_pretrained("gpt2")
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        self.lr = 1e-5

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("EncDecModel")
        return parent_parser

    def preprocess_image(self, figure):
        images = torch.split(figure.to("cpu"), split_size_or_sections=1)
        squeezed_tensors = list(map(torch.squeeze, images))
        image = self.image_processor(images=squeezed_tensors)
        image = torch.stack(image["pixel_values"])
        return image

    def forward(self, image, labels):
        # Tokenize the text into [input_ids]
        text = self.text_tokenizer(
            list(labels), padding=True, truncation=True, return_tensors="pt")
        output = self.model(
            pixel_values=image.to(self.device),
            # Decoder input ids and attention masks are automatically generated
            # by shifting the input ids to the right and adding a start token
            # for causal LM, e.g.
            # inputs: <start> A B C D
            # labels: A       B C D <end>
            labels=text["input_ids"].to(self.device),
        )
        return output

    def training_step(self, batch, batch_idx: int):
        figure, metadata = batch
        image = self.preprocess_image(figure)
        output = self(
            image, metadata)
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
        figure, metadata = batch
        image = self.preprocess_image(figure).to(self.device)
        output = self(image, metadata)
        self.log("val/loss", output.loss)
        # Use sampling to generate sentences
        generated = self.model.generate(
            image.to(self.device), return_dict_in_generate=True, do_sample=True,
            bos_token_id=self.text_tokenizer.bos_token_id, eos_token_id=self.text_tokenizer.eos_token_id)
        decoded: List[str] = self.text_tokenizer.batch_decode(
            generated.sequences, skip_special_tokens=True)
        return output.loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
