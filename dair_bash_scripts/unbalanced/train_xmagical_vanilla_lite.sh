PYTHONPATH="/home/junyao/LfHV/NVAE"
BETA=15
BETA_STR=15
SEED=5
BATCH_SIZE=256
MASTER_PORT=6022
EPOCHS=300
NUM_WORKERS=16
CUDA_VISIBLE_DEVICES=2

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python /home/junyao/LfHV/NVAE/train_dair.py \
--root=/home/junyao/LfHV/NVAE/xmagical_checkpoints/vanilla-lite \
--dataset=xmagical --data=/home/junyao/Datasets/xirl/xmagical \
--batch_size=${BATCH_SIZE} \
--epochs=${EPOCHS} \
--weight_decay_norm=1e-2 \
--num_nf=0 \
--num_latent_scales=1 --num_groups_per_scale=1 --num_latent_per_group=5 \
--num_channels_enc=16 --num_channels_dec=16 \
--num_preprocess_blocks=1 --num_postprocess_blocks=1 \
--num_preprocess_cells=2 --num_postprocess_cells=2 \
--num_cell_per_cond_enc=1 --num_cell_per_cond_dec1=1 \
--num_process_per_node=1 \
--use_se --res_dist \
--num_workers=${NUM_WORKERS} \
--save=vanilla_lite_mask_process_beta=${BETA_STR}_02140200 \
--master_port=${MASTER_PORT} \
--seed=${SEED} \
--kl_beta=${BETA} \
--cond_robot_mask --process_cond_info