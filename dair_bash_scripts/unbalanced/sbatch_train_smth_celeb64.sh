#!/bin/bash
#SBATCH --mem-per-gpu=24G
#SBATCH --partition=kostas-compute
#SBATCH --qos=kostas-high
#SBATCH --cpus-per-gpu=16
#SBATCH --time=96:00:00
#SBATCH --gpus=rtx3090:1
#SBATCH --account=agelosk-account
#SBATCH --job-name=ss-b100
#SBATCH -o out/ss_cond_celeb64_beta100.out

PYTHONPATH="/home/junyao/LfHV/NVAE"
BETA=100
BETA_STR=100
SEED=46
BATCH_SIZE=16
MASTER_PORT=6010
NUM_WORKERS=15

python /home/junyao/LfHV/NVAE/train_dair.py \
--root=/scratch/junyao/nvae/ss_checkpoints/cond_hand \
--dataset=something-something \
--data=/scratch/agelosk/Hands/something_something_paths.pkl \
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
--cond_hand --process_cond_info \
--num_workers=${NUM_WORKERS}

wait