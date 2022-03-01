#!/bin/bash
#SBATCH --mem-per-gpu=12G
#SBATCH --partition=kostas-compute
#SBATCH --qos=kostas-med
#SBATCH --cpus-per-gpu=4
#SBATCH --time=24:00:00
#SBATCH --gpus=rtx2080ti:1
#SBATCH --account=agelosk-account
#SBATCH --job-name=xm-b10
#SBATCH -o out/xmagical_hierarchical_lite_beta10.out

PYTHONPATH="/home/junyao/LfHV/NVAE"
BETA=10
BETA_STR=10
SEED=9
BATCH_SIZE=200
MASTER_PORT=6026
EPOCHS=300
NUM_WORKERS=4

python /home/junyao/LfHV/NVAE/train_dair.py \
--root=/scratch/junyao/nvae/xmagical_checkpoints/hierarchical-lite \
--dataset=xmagical \
--data=/scratch/junyao/datasets/xmagical/ \
--batch_size=${BATCH_SIZE} \
--epochs=${EPOCHS} \
--weight_decay_norm=1e-1 \
--num_nf=1 \
--num_latent_scales=2 --num_groups_per_scale=2 --num_latent_per_group=5 \
--num_channels_enc=16 --num_channels_dec=16 \
--num_preprocess_blocks=1 --num_postprocess_blocks=1 \
--num_preprocess_cells=2 --num_postprocess_cells=2 \
--num_cell_per_cond_enc=1 --num_cell_per_cond_dec=1 \
--num_process_per_node=1 \
--use_se --res_dist --fast_adamax \
--num_workers=${NUM_WORKERS} \
--save=hierarchical_lite_mask_process_beta=${BETA_STR}_02141930 \
--master_port=${MASTER_PORT} \
--seed=${SEED} \
--kl_beta=${BETA} \
--cond_robot_mask --process_cond_info

wait