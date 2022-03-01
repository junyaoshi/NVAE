#!/bin/bash
#SBATCH --mem-per-gpu=12G
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dineshj-high
#SBATCH --cpus-per-gpu=4
#SBATCH --time=12:00:00
#SBATCH --gpus=rtx2080ti:1
#SBATCH -w node-2080ti-5
#SBATCH --job-name=xm-b5-l
#SBATCH -o xmagical_vanilla_lite_beta5.out

PYTHONPATH="/home/junyao/LfHV/NVAE"
BETA=5
BETA_STR=5
SEED=3
BATCH_SIZE=200
MASTER_PORT=6021
EPOCHS=300
NUM_WORKERS=4

python /home/junyao/LfHV/NVAE/train_dair.py \
--root=/scratch/junyao/nvae/xmagical_checkpoints/vanilla-lite \
--dataset=xmagical \
--data=/scratch/junyao/datasets/xmagical/ \
--batch_size=${BATCH_SIZE} \
--epochs=${EPOCHS} \
--weight_decay_norm=1e-2 \
--num_nf=0 \
--num_latent_scales=1 --num_groups_per_scale=1 --num_latent_per_group=5 \
--num_channels_enc=16 --num_channels_dec=16 \
--num_preprocess_blocks=1 --num_postprocess_blocks=1 \
--num_preprocess_cells=2 --num_postprocess_cells=2 \
--num_cell_per_cond_enc=1 --num_cell_per_cond_dec=1 \
--num_process_per_node=1 \
--use_se --res_dist \
--num_workers=${NUM_WORKERS} \
--save=vanilla_mask_process_beta=${BETA_STR}_02140200 \
--master_port=${MASTER_PORT} \
--seed=${SEED} \
--kl_beta=${BETA} \
--cond_robot_mask  --process_cond_info \

wait