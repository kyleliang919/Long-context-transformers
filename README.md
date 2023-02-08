# Long-context-transformers
Exploring finetuning public checkpoints on filtered 8K sequences on Pile

## Exmple of running 8K sequences on Pile

### Single GPU and single node
```bash
CUDA_VISIBLE_DEVICES=0 HF_MODULES_CACHE=./cache/ HF_DATASETS_CACHE=./cache/ TRANSFORMERS_CACHE=./cache/ python finetune.py --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --output_dir pythia-1.5b --gradient_accumulation_steps 8 --fp16 True --evaluation_strategy "epoch" --max_steps 100000
```
Note that this self-contained script holds everything you need to run this finetuning, as long as you set up dependencies, such as flash attention correctly. For a 1.3 B model, it should work on a single A100 80G.

### Multiple GPUs and single node with DeepSpeed
```bash
HF_MODULES_CACHE=./cache/ HF_DATASETS_CACHE=./cache/ TRANSFORMERS_CACHE=./cache/ deepspeed --num_gpus=8 finetune.py --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --output_dir pythia-6.7b --gradient_accumulation_steps 8 --fp16 --evaluation_strategy "epoch" --max_steps 100000 --deepspeed ds_config.json
```

## Dependencies
Not much besides typical pytorch and transformers, the most likely issue will come from flash-attention, where you should follow exactly what the official [repo](https://github.com/HazyResearch/flash-attention.git), in better case, if you have the choice to use the [docker](https://github.com/HazyResearch/flash-attention/blob/main/training/Dockerfile) provided, it will save you from many headaches.

## To do:
* enable multiple GPUs and model parallel
