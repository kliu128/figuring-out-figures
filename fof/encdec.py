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

    def forward(self, figure, labels):
        B = len(labels)

        images = torch.split(figure.to("cpu"), split_size_or_sections=1)
        squeezed_tensors = list(map(torch.squeeze, images))
        image = self.image_processor(images=squeezed_tensors)
        image = torch.stack(image["pixel_values"])

        text = self.text_tokenizer(
            list(labels), padding=True, truncation=True, return_tensors="pt")
        output = self.model(
            pixel_values=image.to(self.device),
            decoder_input_ids=text["input_ids"].to(self.device),
            decoder_attention_mask=text["attention_mask"].to(self.device),
            labels=text["input_ids"].to(self.device),
            output_attentions=True,
        )

        return output.logits, output.loss

    def training_step(self, batch, batch_idx):
        figure, metadata = batch
        _, loss = self(
            figure, metadata)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
