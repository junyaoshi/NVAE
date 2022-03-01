CUDA_VISIBLE_DEVICES=1 python train_dair.py --root=checkpoints \
	--dataset=xmagical --data=/home/junyao/Datasets/xirl/xmagical --batch_size=128 \
	--epochs=100 --weight_decay_norm=1e-1 --num_nf=0 \
	--num_latent_scales=2 --num_groups_per_scale=2 --num_latent_per_group=20 \
	--num_channels_enc=64 --num_channels_dec=64 \
	--num_preprocess_blocks=2 --num_postprocess_blocks=2 \
	--num_preprocess_cells=2 --num_postprocess_cells=2 \
	--num_cell_per_cond_enc=2 --num_cell_per_cond_dec=2 \
	--num_process_per_node=1 \
	--use_se --res_dist --fast_adamax \
	--save=hierarchical_mask_process_beta=4_01260040 \
	--master_port=6021 \
	--seed=41 \
	--kl_beta=4.0 \
	--cond_robot_mask --process_cond_info