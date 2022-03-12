from typing import List, Tuple, Union
import pytorch_lightning as pl
import transformers as tr
import torch
import torch.nn as nn
from datasets import load_metric
from torchtyping import TensorType
import math


def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no_' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default})


def estimated_stepping_batches(self) -> Union[int, float]:
    r"""
    Estimated stepping batches for the complete training inferred from DataLoaders, gradient
    accumulation factor and distributed setup.
    Examples::
        def configure_optimizers(self):
            optimizer = ...
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches
            )
            return [optimizer], [scheduler]
    """
    accumulation_scheduler = self.accumulation_scheduler

    if accumulation_scheduler.epochs != [0]:
        raise Exception(
            "Estimated stepping batches cannot be computed with different"
            " `accumulate_grad_batches` at different epochs."
        )

    # infinite training
    if self.max_epochs == -1 and self.max_steps == -1:
        return float("inf")

    if self.train_dataloader is None:
        print(
            "Loading `train_dataloader` to estimate number of stepping batches.")
        self.reset_train_dataloader()

    total_batches = self.num_training_batches

    # iterable dataset
    if total_batches == float("inf"):
        return self.max_steps

    self.accumulate_grad_batches = accumulation_scheduler.get_accumulate_grad_batches(
        self.current_epoch)
    effective_batch_size = self.accumulate_grad_batches
    max_estimated_steps = math.ceil(
        total_batches / effective_batch_size) * max(self.max_epochs, 1)

    max_estimated_steps = min(
        max_estimated_steps, self.max_steps) if self.max_steps != -1 else max_estimated_steps
    return max_estimated_steps


class ExtensibleEncoder(nn.Module):
    def __init__(self, device: str, vision_model: str, use_vision_encoder: bool,
                 use_scibert: bool, freeze_scibert: bool):
        super().__init__()
        if "clip" in vision_model:
            self.clip = tr.CLIPVisionModel.from_pretrained(vision_model)
        else:
            self.clip = tr.AutoModel.from_pretrained(vision_model)
        self.config = self.clip.config
        self.main_input_name = self.clip.main_input_name
        self.device = device

        self.use_vision_encoder = use_vision_encoder

        self.use_scibert = use_scibert
        if self.use_scibert:
            # SCIBERT encoder for metadata
            self.metadata_encoder = tr.AutoModel.from_pretrained(
                'allenai/scibert_scivocab_uncased')

            self.freeze_scibert = freeze_scibert
            if freeze_scibert:
                # Freeze SCIBERT params
                for param in self.metadata_encoder.base_model.parameters():
                    param.requires_grad = False

    def forward(self, pixel_values, metadata=None, *args, **kwargs):
        if self.use_vision_encoder:
            image_output = self.clip(pixel_values, *args, **kwargs)
        if not self.use_scibert:
            return image_output  # This must work due to the assert placed earlier

        metadata_output = self.metadata_encoder(
            input_ids=metadata["input_ids"],
            attention_mask=metadata["attention_mask"],
            token_type_ids=metadata["token_type_ids"])

        if not self.use_vision_encoder:
            return metadata_output

        # This line should be technically useless but included out of superstition
        image_output.pooler_output *= metadata_output.pooler_output
        # Concatenate on the sequence dimension
        image_output.last_hidden_state = torch.cat(
            [image_output.last_hidden_state, metadata_output.last_hidden_state], dim=1)
        return image_output


class EncoderDecoderModel(pl.LightningModule):
    def __init__(self,
                 text_model: str, vision_model: str, tpu_hacks: bool,
                 use_vision_encoder: bool, use_scibert: bool, freeze_scibert: bool, 
                 lr: float, lr_scheduler: str, use_references: bool,
                 use_top_p_sampling: bool, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        assert use_vision_encoder or use_scibert, \
            'Encoder missing; must use at least some form of vision encoder or text encoder'
        encoder = ExtensibleEncoder(
            self.device, vision_model=vision_model, use_vision_encoder=use_vision_encoder,
            use_scibert=use_scibert, freeze_scibert=freeze_scibert)

        # Freeze encoder
        # for param in encoder.clip.base_model.parameters():
        #     param.requires_grad = False

        decoder = tr.AutoModelForCausalLM.from_pretrained(
            text_model, add_cross_attention=True)

        model = tr.VisionEncoderDecoderModel(
            encoder=encoder.clip, decoder=decoder)
        model.encoder = encoder  # Loophole to make VisionEncoder work with custom encoder
        if not use_vision_encoder:
            del model.encoder.clip
        # use GPT2's eos_aaaatoken as the pad as well as eos token
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

        self.metadata_tokenizer = tr.AutoTokenizer.from_pretrained(
            'allenai/scibert_scivocab_uncased')

        self.tpu_hacks = tpu_hacks
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.use_references = use_references
        self.use_top_p_sampling = use_top_p_sampling
        # self.strategy = strategy

        # Use sacrebleu as a standard BLEU computer.
        self.bleu_metric = load_metric('sacrebleu')
        self.rouge_metric = load_metric('rouge')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("EncDecModel")
        # facebook/bart-large
        parser.add_argument("--text_model", type=str,
                            default="distilgpt2")
        parser.add_argument("--vision_model", type=str,
                            default="openai/clip-vit-base-patch32")
        add_bool_arg(parser, "use_scibert", default=False)
        add_bool_arg(parser, 'freeze_scibert', default=True)
        add_bool_arg(parser, "use_references", default=False)
        add_bool_arg(parser, "use_top_p_sampling", default=False)
        add_bool_arg(parser, 'use_vision_encoder', default=True)
        parser.add_argument("--lr_scheduler", type=str, default=None)
        return parent_parser

    def process_batch(self, batch) -> Tuple[TensorType["b", 3, 224, 224], TensorType["b", "len"], TensorType["b", "len"]]:
        # (B, 3, 224, 224)
        figure, labels, title, abstract, references = batch["figure"], batch[
            "labels"], batch['title'], batch['abstract'], batch['references']

        if self.use_references:
            # Allocate 100 char for title, 150 for abstract, the rest for references
            metadata = [f"{t[:100]} [SEP] {a[:150]} [SEP] {r}" for t,
                        a, r in zip(title, abstract, references)]
        else:
            metadata = [f"{t} [SEP] {a}" for t, a in zip(title, abstract)]

        tokenized_metadata = self.metadata_tokenizer(metadata,
                                                     add_special_tokens=True,
                                                     padding="max_length" if self.tpu_hacks else True,
                                                     max_length=512,
                                                     truncation=True,
                                                     return_tensors='pt').to(self.device)
        # Returns { "input_ids", "attention_mask" } but we can avoid attn mask
        # because VisionEncoderDecoder will generate it
        labels = self.text_tokenizer(
            labels,
            padding="max_length" if self.tpu_hacks else True,
            truncation=True,
            return_tensors="pt").input_ids.to(self.device)

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
        batch_size = len(batch["labels"])
        self.log("train/loss", output.loss, batch_size=batch_size)
        self.log("train/perplexity", torch.exp(output.loss),
                 batch_size=batch_size)

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
        image, labels, title = self.process_batch(batch)
        batch_size = len(labels)

        output = self(image, labels, title)

        sample_args = {
            "top_k": 50
        } if self.use_top_p_sampling else {
            "top_p": 0.9,
            "top_k": 0
        }

        # Use sampling to generate sentences
        generated = self.model.generate(
            image, metadata=title, return_dict_in_generate=True, do_sample=True,
            **sample_args,
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
        self.log("val/loss", output.loss, batch_size=batch_size)
        self.log("val/perplexity", torch.exp(output.loss),
                 batch_size=batch_size)

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
        # if "deepspeed" in self.strategy:
        #     optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.lr)
        # else:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        if self.lr_scheduler == "linear":
            training_steps = estimated_stepping_batches(self.trainer)
            print("Training steps:", training_steps)
            print("Using linear learning rate scheduler")
            scheduler = tr.get_scheduler(
                "linear",
                optimizer,
                num_warmup_steps=0,
                num_training_steps=training_steps)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # linear scheduler decreases learning rate on every step
                    "interval": "step",
                }
            }
        elif self.lr_scheduler == "onecycle":
            training_steps = estimated_stepping_batches(self.trainer)
            print("Training steps:", training_steps)
            print("Using onecycle learning rate scheduler")
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.lr, total_steps=training_steps)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # onecycle scheduler decreases learning rate on every step
                    "interval": "step",
                }
            }
        else:
            return optimizer
