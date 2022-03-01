#!/bin/bash
#SBATCH --mem-per-gpu=24G
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dineshj-high
#SBATCH --cpus-per-gpu=9
#SBATCH --time=48:00:00
#SBATCH --gpus=rtx3090:1
#SBATCH -w node-3090-0
#SBATCH --job-name=xm-b1
#SBATCH -o xmagical_vanilla_beta1.out

PYTHONPATH="/home/junyao/LfHV/NVAE"
BETA=1
BETA_STR=1
SEED=43
BATCH_SIZE=200
MASTER_PORT=6024
EPOCHS=100
NUM_WORKERS=8

python /home/junyao/LfHV/NVAE/train_dair.py \
--root=/scratch/junyao/nvae/xmagical_checkpoints \
--dataset=xmagical \
--data=/scratch/junyao/datasets/xmagical/ \
--batch_size=${BATCH_SIZE} \
--epochs=${EPOCHS} \
--weight_decay_norm=1e-2 \
--num_nf=0 \
--num_latent_scales=1 --num_groups_per_scale=1 --num_latent_per_group=256 \
--min_groups_per_scale=4 \
--num_channels_enc=64 --num_channels_dec=64 \
--num_preprocess_blocks=3 --num_postprocess_blocks=3 \
--num_preprocess_cells=2 --num_postprocess_cells=2 \
--num_cell_per_cond_enc=2 --num_cell_per_cond_dec=2 \
--num_process_per_node=1 \
--use_se --res_dist \
--num_workers=${NUM_WORKERS} \
--save=vanilla_mask_process_beta=${BETA_STR}_01260010 \
--master_port=${MASTER_PORT} \
--seed=${SEED} \
--kl_beta=${BETA} \
--cond_robot_mask  --process_cond_info \
--cont_training \

wait