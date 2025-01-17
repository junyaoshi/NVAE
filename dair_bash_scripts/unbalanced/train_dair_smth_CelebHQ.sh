python train_dair.py \
--root=/scratch/junyao/nvae/ss_checkpoints \
--dataset=something-something \
--data=/scratch/agelosk/Hands/something_something/ \
--batch_size=8 \
--epochs=300 \
--weight_decay_norm=1e-2 --weight_decay_norm_anneal --weight_decay_norm_init=1. \
--num_nf=2 \
--num_latent_scales=4 --num_groups_per_scale=16 --num_latent_per_group=16 \
--min_groups_per_scale=4 \
--num_channels_enc=30 --num_channels_dec=30 \
--num_preprocess_blocks=1 --num_postprocess_blocks=1 \
--num_preprocess_cells=2 --num_postprocess_cells=2 \
--num_cell_per_cond_enc=2 --num_cell_per_cond_dec=2 \
--num_process_per_node=1 \
--use_se --res_dist --fast_adamax --ada_groups \
--save=celebhq_beta=4 \
--master_port=6021 \
--seed=44 \
--kl_beta=4.0 \
