python train_dair.py \
--data /scratch/agelosk/Hands/something_something/ \
--root /scratch/junyao/nvae/ss_checkpoints \
--dataset something-something \
--batch_size 16 \
--epochs 100 \
--num_latent_scales 3 \
--num_groups_per_scale 20 \
--num_postprocess_cells 2 \
--num_preprocess_cells 2 \
--num_cell_per_cond_enc 2 \
--num_cell_per_cond_dec 2 \
--num_latent_per_group 20 \
--num_preprocess_blocks 1 \
--num_postprocess_blocks 1 \
--weight_decay_norm 1e-1 \
--num_channels_enc 64 \
--num_channels_dec 64 \
--num_nf 1 \
--ada_groups \
--num_process_per_node 1 \
--use_se \
--res_dist \
--fast_adamax \
--kl_beta=4.0 \
--save celeb64_beta=4