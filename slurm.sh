#!/bin/bash
#SBATCH --partition=g80n140
#SBATCH --job-name=testlongcontext
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --output=gpt_neox_20.out
#SBATCH --comment=laion
#SBATCH --open-mode=append
#SBATCH --exclusive

module load openmpi
module load cuda/11.7

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES

hostfile="hostfile.txt"
rm -f $hostfile
for node in $HOSTNAMES; do
  echo $node slots=8 >> $hostfile
done

source /fsx/home-kaizhaol/long-context-transformers/venv/bin/activate

HF_MODULES_CACHE=./cache/ HF_DATASETS_CACHE=./cache/ TRANSFORMERS_CACHE=./cache/ deepspeed --master_addr $MASTER_ADDR --hostfile='/fsx/home-kaizhaol/long-context-transformers/hostfile.txt' --launcher OpenMPI finetune.py --model_name_or_path="EleutherAI/gpt-neox-20b" --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --output_dir gpt-neox-20b --gradient_accumulation_steps 8 --fp16 --evaluation_strategy "epoch" --max_steps 100000 --deepspeed ds_config.json --gradient_checkpointing
