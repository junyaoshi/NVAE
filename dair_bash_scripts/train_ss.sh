export PYTHONPATH="/home/junyao/LfHV/NVAE"
export BETA=1
export BETA_STR=1
export SEED=1
export BATCH_SIZE=32
export MASTER_PORT=6000
export EPOCHS=300
export NUM_WORKERS=8
export TYPE="cond_hand"
export TIME="03172300"

echo "PYTHONPATH: ${PYTHONPATH}"
echo "BETA: ${BETA}"
echo "BETA_STR: ${BETA_STR}"
echo "SEED: ${SEED}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "EPOCHS: ${EPOCHS}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "TYPE: ${TYPE}"
echo "TIME: ${TIME}"

CUDA_VISIBLE_DEVICES=1 python /home/junyao/LfHV/NVAE/train_dair.py \
--root=/home/junyao/LfHV/NVAE/ss_checkpoints/${TYPE} \
--dataset=something-something \
--data=/home/junyao/Datasets/something_something_paths_cv.pkl \
--batch_size=${BATCH_SIZE} \
--epochs=${EPOCHS} \
--weight_decay_norm=1e-1 \
--num_nf=1 \
--num_latent_scales=2 --num_groups_per_scale=12 --num_latent_per_group=20 \
--num_channels_enc=64 --num_channels_dec=64 \
--num_preprocess_blocks=2 --num_preprocess_cells=2 \
--num_postprocess_blocks=2 --num_postprocess_cells=8 \
--num_cell_per_cond_enc=1 --num_cell_per_cond_dec=1 \
--num_process_per_node=1 \
--use_se --res_dist --ada_groups \
--save=${BETA_STR}_bs=${BATCH_SIZE}_${TIME} \
--master_port=${MASTER_PORT} \
--seed=${SEED} \
--kl_beta=${BETA} \
--num_workers=${NUM_WORKERS} \
--cond_params_3d --cond_hand_bb --cond_cropped_hand --process_cond_info