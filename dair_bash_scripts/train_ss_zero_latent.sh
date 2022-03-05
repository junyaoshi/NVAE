export PYTHONPATH="/home/junyao/LfHV/NVAE"
export BETA=0
export BETA_STR=0
export SEED=13
export BATCH_SIZE=16
export MASTER_PORT=6010
export NUM_WORKERS=8

echo "PYTHONPATH: ${PYTHONPATH}"
echo "BETA: ${BETA}"
echo "BETA_STR: ${BETA_STR}"
echo "SEED: ${SEED}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NUM_WORKERS: ${NUM_WORKERS}"

CUDA_VISIBLE_DEVICES=0 python /home/junyao/LfHV/NVAE/train_dair.py \
--root=/home/junyao/LfHV/NVAE/ss_checkpoints/cond_hand/zero_image \
--dataset=something-something \
--data=/home/junyao/Datasets/something_something_paths_cv.pkl \
--batch_size=${BATCH_SIZE} \
--epochs=100 \
--weight_decay_norm=1e-1 \
--num_nf=1 \
--num_latent_scales=2 --num_groups_per_scale=12 --num_latent_per_group=20 \
--num_channels_enc=64 --num_channels_dec=64 \
--num_preprocess_blocks=3 --num_preprocess_cells=2 \
--num_postprocess_blocks=3 --num_postprocess_cells=20 \
--num_cell_per_cond_enc=1 --num_cell_per_cond_dec=1 \
--num_process_per_node=1 \
--use_se --res_dist --ada_groups \
--save=beta=${BETA_STR}_bs=${BATCH_SIZE} \
--master_port=${MASTER_PORT} \
--seed=${SEED} \
--kl_beta=${BETA} \
--cond_hand --process_cond_info \
--num_workers=${NUM_WORKERS} \
--zero_latent --print_time