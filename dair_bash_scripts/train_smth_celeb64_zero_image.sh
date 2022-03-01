PYTHONPATH="/home/junyao/LfHV/NVAE"
BETA=10
BETA_STR=10
SEED=13
BATCH_SIZE=12
MASTER_PORT=6011
NUM_WORKERS=10
CUDA_VISIBLE_DEVICES=0 python /home/junyao/LfHV/NVAE/train_dair.py \
--root=/home/junyao/LfHV/NVAE/ss_checkpoints/cond_hand/zero_image \
--dataset=something-something \
--data=/home/junyao/Datasets/something_something_paths_cv.pkl \
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
--save=celeb64_beta=${BETA_STR}_bs=${BATCH_SIZE} \
--master_port=${MASTER_PORT} \
--seed=${SEED} \
--kl_beta=${BETA} \
--cond_hand --process_cond_info \
--num_workers=${NUM_WORKERS} \
--zero_image