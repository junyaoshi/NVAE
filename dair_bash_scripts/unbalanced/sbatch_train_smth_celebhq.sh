#!/bin/bash
#SBATCH --mem-per-gpu=12G
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dineshj-high
#SBATCH --cpus-per-gpu=4
#SBATCH --time=48:00:00
#SBATCH --gpus=rtx2080ti:1
#SBATCH -w node-2080ti-5
#SBATCH --job-name=hq-beta4
#SBATCH -o celebhq_beta4.out

PYTHONPATH="/home/junyao/LfHV/NVAE"
BETA=4.0
BETA_STR=4
SEED=44
BATCH_SIZE=14
MASTER_PORT=6021

python /home/junyao/LfHV/NVAE/train_dair.py \
--root=/scratch/junyao/nvae/ss_checkpoints \
--dataset=something-something \
--data=/scratch/agelosk/Hands/something_something/ \
--batch_size=${BATCH_SIZE} \
--epochs=300 \
--weight_decay_norm=1e-2 --weight_decay_norm_anneal --weight_decay_norm_init=1. \
--num_nf=2 \
--num_latent_scales=4 --num_groups_per_scale=16 --num_latent_per_group=16 \
--min_groups_per_scale=4 \
--num_channels_enc=30 --num_channels_dec=30 \
--num_preprocess_blocks=1 --num_postprocess_blocks=1 \
--num_preprocess_cells=2 --num_postprocess_cells=2 \
--num_cell_per_cond_enc=2 --num_cell_per_cond_dec=2 \
--num_process_per_node=1 \
--use_se --res_dist --fast_adamax --ada_groups \
--save=celebhq_beta=${BETA_STR} \
--master_port=${MASTER_PORT} \
--seed=${SEED} \
--kl_beta=${BETA} \
--cont_training

wait