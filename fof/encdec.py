from typing import List
import pytorch_lightning as pl
import transformers as tr
import torch


class EncoderDecoderModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        clip = tr.CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32")
        gpt2 = tr.AutoModelForCausalLM.from_pretrained(
            "gpt2", add_cross_attention=True)

        model = tr.VisionEncoderDecoderModel(encoder=clip, decoder=gpt2)
        # use GPT2's eos_token as the pad as well as eos token
        model.config.eos_token_id = model.config.decoder.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

        self.model = model
        self.image_processor = tr.CLIPFeatureExtractor(
            do_resize=False,
            do_center_crop=False,
        )
        self.text_tokenizer = tr.AutoTokenizer.from_pretrained("gpt2")
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        self.lr = 1e-4

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
        B = len(labels)

        text = self.text_tokenizer(
            list(labels), padding=True, truncation=True, return_tensors="pt")
        output = self.model(
            pixel_values=image.to(self.device),
            decoder_input_ids=text["input_ids"].to(self.device),
            decoder_attention_mask=text["attention_mask"].to(self.device),
            labels=text["input_ids"].to(self.device),
            output_attentions=True,
        )

        # Sanity check that cross attention is working
        assert output.cross_attentions[0].mean() != 0
        return output

    def training_step(self, batch, batch_idx):
        figure, metadata = batch
        image = self.preprocess_image(figure)
        output = self(
            image, metadata)
        self.log("train/loss", output.loss)
        return output.loss

    def validation_step(self, batch, batch_idx):
        figure, metadata = batch
        image = self.preprocess_image(figure).to(self.device)
        output = self(image, metadata)
        self.log("val/loss", output.loss)
        # Use top-p sampling with 0.9 as the probability
        generated = self.model.generate(
            image, return_dict_in_generate=True, top_p=0.9)
        decoded: List[str] = self.text_tokenizer.batch_decode(
            generated, skip_special_tokens=True)

        return output.loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
