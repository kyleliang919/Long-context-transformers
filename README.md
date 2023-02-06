# Long-context-transformers
Exploring finetuning public checkpoints on filter 8K sequences on Pile

## Exmple of running 8K pile on a filtered pile dataset
```bash
CUDA_VISIBLE_DEVICES=0 HF_MODULES_CACHE=./cache/ HF_DATASETS_CACHE=./cache/ TRANSFORMERS_CACHE=./cache/ python finetune.py --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --output_dir pythia-1.5b --gradient_accumulation_steps 8 --fp16 True --evaluation_strategy "epoch" --max_steps 100000
```
Note that this self-contained script holds everything you need to run this finetuning, as long as you set up dependencies, such as flash attention correctly. For a 1.3 B model, it should work on a single A100 80G.

## To do:
* enable multiple GPUs
