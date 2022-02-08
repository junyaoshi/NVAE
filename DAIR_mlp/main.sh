CUDA_VISIBLE_DEVICES=1 \
python main.py \
--batch_size=128 \
--vae_epochs=20 \
--vae_lr=1e-3 \
--probe_epochs=20 \
--probe_lr=1e-3 \
--log_interval=50 \
--num_samples=5 \
--num_features=2 \
--cond_dim=1 \
--latent_dim=2 \
--save=logs/vae_probe/toy_nonconditional_hidden_mult=4_feat=2_beta=0 \
--seed=31 \
--beta=0 \
