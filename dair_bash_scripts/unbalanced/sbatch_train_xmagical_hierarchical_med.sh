#!/bin/bash
#SBATCH --mem-per-gpu=12G
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dineshj-high
#SBATCH --cpus-per-gpu=4
#SBATCH --time=24:00:00
#SBATCH --gpus=rtx2080ti:1
#SBATCH -w node-2080ti-5
#SBATCH --job-name=xm-vm
#SBATCH -o out/xmagical_vm_hierarchical_med_beta10.out

PYTHONPATH="/home/junyao/LfHV/NVAE"
BETA=10
BETA_STR=10
SEED=22
BATCH_SIZE=25
MASTER_PORT=6052
EPOCHS=300
NUM_WORKERS=4

python /home/junyao/LfHV/NVAE/train_dair.py \
--root=/scratch/junyao/LfHV/NVAE/xmagical_checkpoints/hierarchical-med/vec_mask \
--dataset=xmagical \
--data=/scratch/junyao/Datasets/xmagical/ \
--batch_size=${BATCH_SIZE} \
--epochs=${EPOCHS} \
--weight_decay_norm=1e-1 \
--num_nf=1 \
--num_latent_scales=2 --num_groups_per_scale=2 --num_latent_per_group=16 \
--num_channels_enc=64 --num_channels_dec=64 \
--num_preprocess_blocks=1 --num_postprocess_blocks=1 \
--num_preprocess_cells=2 --num_postprocess_cells=2 \
--num_cell_per_cond_enc=1 --num_cell_per_cond_dec=1 \
--num_process_per_node=1 \
--use_se --res_dist --fast_adamax \
--num_workers=${NUM_WORKERS} \
--save=beta=${BETA_STR}_bs=${BATCH_SIZE}_02212100 \
--master_port=${MASTER_PORT} \
--seed=${SEED} \
--kl_beta=${BETA} \
--cond_robot_mask --cond_robot_vec --process_cond_info

wait