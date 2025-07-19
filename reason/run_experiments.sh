#!/bin/bash
#SBATCH --job-name=reason_kvcache
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition cml-furongh
#SBATCH --account cml-furongh
#SBATCH --qos cml-high
#SBATCH -t 06:00:00

# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export NCCL_IB_DISABLE=1

source activate base
conda activate kvcache

export HF_HOME='/nfshomes/minghui/scratch/.cache/huggingface'
export HF_TOKEN=''
huggingface-cli login --token $HF_TOKEN --add-to-git-credential

for dataset in folio logiqa strategyqa drop reclor
do
    for press_name in h2o streaming_llm knorm
    do
        for cache_budget in 128 256 384 512;
        do
            echo "Running evaluation for dataset: $dataset, press_name: $press_name, cache_budget: $cache_budget, max_new_tokens: 2048, num_samples: 100"
            python3 evaluate.py --dataset $dataset --press_name $press_name --cache_budget $cache_budget --max_new_tokens 2048 --num_samples 100
        done
    done
done
