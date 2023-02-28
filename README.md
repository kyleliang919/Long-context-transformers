# Long-context-transformers
Exploring finetuning public checkpoints on filtered 8K sequences on Pile

## Exmple of running 8K sequences on Pile

### Single GPU and single node
```bash
CUDA_VISIBLE_DEVICES=0 HF_MODULES_CACHE=./cache/ HF_DATASETS_CACHE=./cache/ TRANSFORMERS_CACHE=./cache/ python finetune.py --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --output_dir pythia-1.4b --gradient_accumulation_steps 8 --fp16 --evaluation_strategy "epoch" --max_steps 100000 --model_name_or_path EleutherAI/pythia-1.4b
```
Note that this self-contained script holds everything you need to run this finetuning, as long as you set up dependencies, such as flash attention correctly. For a 1.3 B model, it should work on a single A100 80G.

### Multiple GPUs and Single node with DeepSpeed
```bash
HF_MODULES_CACHE=./cache/ HF_DATASETS_CACHE=./cache/ TRANSFORMERS_CACHE=./cache/ deepspeed --num_gpus=8 finetune.py --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --output_dir pythia-6.9b --gradient_accumulation_steps 8 --fp16 --evaluation_strategy "epoch" --max_steps 100000 --deepspeed ds_config.json --model_name_or_path EleutherAI/pythia-6.9b
```
If you hit "RuntimeError: Tensors must be contiguous" , follow this simple [fix](https://github.com/amyeroberts/transformers/commit/4ea536b45a3fd20ff808a0c236899a66e24bf7fe) and modify your deepSpeed library

### Multiple GPUs and Multiple Nodes with DeepSpeed with Slurm
```bash
sbatch slurm.sh
```
Note that you can launch up to pythia-20B with 16 80GB A100s, aka two nodes. Since the above slurm script relies on openmpi, you should be able to generalize it to more than 2 nodes without problems.

### Warnings
*For OPT, it auto pads 2 in the end, so the max position should be subtracted by 2 (e.g instead of 8192, you will have to put 8190)
*For Bloom, to support Alibi, we had to compute pesudoinverse, its backward is unfriendly to gradient checkpointing, if you see backward precision issue, try to disable gradient checkpointing.

## Dependencies
Not much besides typical pytorch and transformers, the most likely issue will come from flash-attention, where you should follow exactly what the official [repo](https://github.com/HazyResearch/flash-attention.git), in better case, if you have the choice to use the [docker](https://github.com/HazyResearch/flash-attention/blob/main/training/Dockerfile) provided, it will save you from many headaches.

## To do:
- [x] enable multiple GPUs and model parallel
- [x] supporting alibi (Bloom) and normal trainable embedding (OPT) 

## Citation
You can find the citation option under the wedge in the repo. Beyond that please make sure to cite the amazing work by the incredible [Tri Dao](https://tridao.me/). Without his flash-attention this repo will not be possible.
```
Dao, Tri, et al. "Flashattention: Fast and memory-efficient exact attention with io-awareness." arXiv preprint arXiv:2205.14135 (2022).
```
