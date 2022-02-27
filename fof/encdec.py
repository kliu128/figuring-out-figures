from typing import List, Tuple
import pytorch_lightning as pl
import transformers as tr
import torch
import torch.nn as nn
from datasets import load_metric
from torchtyping import TensorType


class ExtensibleEncoder(nn.Module):
    def __init__(self, vision_model: str, use_scibert: bool):
        super().__init__()
        if "clip" in vision_model:
            self.clip = tr.CLIPVisionModel.from_pretrained(vision_model)
        else:
            self.clip = tr.AutoModel.from_pretrained(vision_model)
        self.config = self.clip.config
        self.main_input_name = self.clip.main_input_name

        if self.use_scibert:
            # SCIBERT encoder for metadata
            self.metadata_tokenizer = tr.AutoTokenizer.from_pretrained(
                'allenai/scibert_scivocab_cased')
            self.metadata_encoder = tr.AutoModel.from_pretrained(
                'allenai/scibert_scivocab_cased')

    def forward(self, metadata = None, *args, **kwargs):
        image_embedding = self.clip(*args, **kwargs)
        if not self.use_scibert:
            return image_embedding
        
        metadata_embedding = self.metadata_encoder(metadata)
        return metadata_embedding * image_embedding


class EncoderDecoderModel(pl.LightningModule):
    def __init__(self, text_model: str, vision_model: str, tpu_hacks: bool, use_scibert: bool, lr: float, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        encoder = ExtensibleEncoder(vision_model=vision_model, use_scibert=use_scibert)
        decoder = tr.AutoModelForCausalLM.from_pretrained(
            text_model, add_cross_attention=True)

        model = tr.VisionEncoderDecoderModel(
            encoder=encoder.clip, decoder=decoder)
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

        self.tpu_hacks = tpu_hacks
        self.lr = lr

        # Use sacrebleu as a standard BLEU computer.
        self.bleu_metric = load_metric('sacrebleu')
        self.rouge_metric = load_metric('rouge')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("EncDecModel")
        parser.add_argument("--text_model", type=str, default="distilgpt2")
        parser.add_argument("--vision_model", type=str, default="openai/clip-vit-base-patch32")
        parser.add_argument("--use_scibert", type=bool, default=False)
        return parent_parser

    def process_batch(self, batch) -> Tuple[TensorType["b", 3, 224, 224], TensorType["b", "len"]]:
        # (B, 3, 224, 224)
        figure, labels, metadata = batch["figure"], batch["labels"], batch['metadata']
        tokenized_metadata = self.metadata_tokenizer(
            metadata['title'],
            padding="max_length" if self.tpu_hacks else True,
            return_tensors='pt')
        # Returns { "input_ids", "attention_mask" } but we can avoid attn mask
        # because VisionEncoderDecoder will generate it
        labels = self.text_tokenizer(
            labels,
            padding="max_length" if self.tpu_hacks else True,
            return_tensors="pt")["input_ids"].to(self.device)
        
        return figure, labels, tokenized_metadata

    def forward(self, image, labels, metadata):
        output = self.model(
            pixel_values=image,
            # Decoder input ids and attention masks are automatically generated
            # by shifting the input ids to the right and adding a start token
            # for causal LM, e.g.
            # inputs: <start> A B C D
            # labels: A       B C D <end>
            labels=labels,
            metadata=metadata,
        )
        return output

    def training_step(self, batch, batch_idx: int):
        output = self(*self.process_batch(batch))
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
        if self.tpu_hacks:
            return
        
        image, labels = self.process_batch(batch)

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
        if self.tpu_hacks:
            return
        
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
            # "lr_scheduler": scheduler,
            # "monitor": "val/loss"
        }
