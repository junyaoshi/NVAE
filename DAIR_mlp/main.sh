CUDA_VISIBLE_DEVICES=1 \
python main.py \
--batch_size=128 \
--vae_epochs=40 \
--vae_lr=1e-3 \
--probe_epochs=20 \
--probe_lr=1e-3 \
--log_interval=50 \
--num_samples=5 \
--num_features=2 \
--cond_dim=1 \
--latent_dim=2 \
--save=logs/vae_probe/toy_hidden_mult=4_feat=2_beta=5e-2 \
--seed=17 \
--beta=5e-2 \
