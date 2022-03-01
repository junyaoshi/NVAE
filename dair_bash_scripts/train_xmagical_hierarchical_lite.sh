export PYTHONPATH="/home/junyao/LfHV/NVAE"
export CUDA_VISIBLE_DEVICES=1
export BETA=4.0
export BETA_STR=4
export SEED=7
export BATCH_SIZE=200
export MASTER_PORT=6022
export EPOCHS=300
export NUM_WORKERS=16

echo "PYTHONPATH: ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "BETA: ${BETA}"
echo "BETA_STR: ${BETA_STR}"
echo "SEED: ${SEED}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "EPOCHS: ${EPOCHS}"
echo "NUM_WORKERS: ${NUM_WORKERS}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python /home/junyao/LfHV/NVAE/train_dair.py \
--root=/home/junyao/LfHV/NVAE/xmagical_checkpoints/hierarchical-lite \
--dataset=xmagical --data=/home/junyao/Datasets/xirl/xmagical \
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
--save=hierarchical_mask_process_beta=${BETA_STR}_02140200 \
--master_port=${MASTER_PORT} \
--seed=${SEED} \
--kl_beta=${BETA} \
--cond_robot_mask --process_cond_info