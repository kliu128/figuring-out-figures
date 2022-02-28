# SciBERT

```
python fof/run.py train --gpus 1 --exp encdec-scibertwhoa --lr 1e-3 --model encdec --batch_size 2 --use_scibert True --pl_logger tb --accumulate_grad_batches 8

python fof/run.py train --gpus 1 --exp encdec-scibertwhoa --lr 5e-5 --model encdec --batch_size 4 --use_scibert True --pl_logger tb --accumulate_grad_batches 4