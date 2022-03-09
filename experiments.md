# SciBERT

```
python fof/run.py train --gpus 1 --exp encdec-scibertwhoa --lr 1e-3 --model encdec --batch_size 2 --use_scibert True --pl_logger tb --accumulate_grad_batches 8

python fof/run.py train --gpus 1 --exp encdec-scibertwhoa --lr 5e-5 --model encdec --batch_size 2 --use_scibert True --pl_logger tb --accumulate_grad_batches 4

TPU + GPT2 + SciBERT
TPU: python fof/run.py train --tpu_hacks --exp tpu-gpt2-scibert --lr 5e-5 --model encdec --batch_size 4 --use_scibert True --pl_logger tb --accumulate_grad_batches 4 --text_model gpt2
```

```
LR scheduler
python fof/run.py train --gpus 1 --exp test-lr-scheduler --limit 10 --lr 5e-5 --model encdec --batch_size 2 --use_scibert --pl_logger wandb --lr_scheduler linear --max_epochs 15 --caption_type normalized

Using references + LR scheduler + frozen SCIBERT
https://wandb.ai/figuring-out-figures/figuring-out-figures/runs/1d7mntmw
python fof/run.py train --gpus 1 --exp references-and-lr-scheduler --lr 5e-5 --model encdec --batch_size 4 --accumulate_grad_batches 4 --use_scibert --pl_logger wandb --lr_scheduler linear --max_epochs 15 --caption_type normalized --use_references