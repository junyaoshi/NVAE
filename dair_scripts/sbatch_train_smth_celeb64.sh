#!/bin/bash
#SBATCH --mem-per-gpu=12G
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dineshj-high
#SBATCH --cpus-per-gpu=4
#SBATCH --time=48:00:00
#SBATCH --gpus=rtx2080ti:1
#SBATCH -w node-2080ti-5
#SBATCH --job-name=64-beta4
#SBATCH -o celeb64_beta4.out

PYTHONPATH="/home/junyao/LfHV/NVAE"
BETA=4.0
BETA_STR=4
SEED=45
BATCH_SIZE=6
MASTER_PORT=6022

python /home/junyao/LfHV/NVAE/train_dair.py \
--root=/scratch/junyao/nvae/ss_checkpoints \
--dataset=something-something \
--data=/scratch/agelosk/Hands/something_something/ \
--batch_size=${BATCH_SIZE} \
--epochs=100 \
--weight_decay_norm=1e-1 \
--num_nf=1 \
--num_latent_scales=3 --num_groups_per_scale=20 --num_latent_per_group=20 \
--num_channels_enc=64 --num_channels_dec=64 \
--num_preprocess_blocks=1 --num_postprocess_blocks=1 \
--num_preprocess_cells=2 --num_postprocess_cells=2 \
--num_cell_per_cond_enc=2 --num_cell_per_cond_dec=2 \
--num_process_per_node=1 \
--use_se --res_dist --fast_adamax --ada_groups \
--save=celeb64_beta=${BETA_STR} \
--master_port=${MASTER_PORT} \
--seed=${SEED} \
--kl_beta=${BETA} \
--cont_training

wait