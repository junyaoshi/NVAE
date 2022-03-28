from distributions import Normal
import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(
            self,
            vae,
            vanilla_vae,
            baseline_mode,
            output_dim,
            device
    ):
        super(Probe, self).__init__()
        # pretrained vae encoder
        self.vae = vae
        self.vanilla_vae = vanilla_vae
        self.baseline_mode = baseline_mode
        self.decoder_in_size = None  # size of decoder input, used by baseline mode only
        self.device = device

        # Output: 6 for world_state, 4 for robot_type and 4 for robot_state
        self.output_dim = output_dim
        if self.vanilla_vae:
            self.decoder = nn.Sequential(
                nn.Conv2d(5, 2, (3, 3), (2, 2), 1),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),
                nn.Flatten(),
                nn.Linear(128, self.output_dim)
            )
        else:
            self.decoder = nn.Sequential(
                nn.Conv2d(256, 256, (3, 3), (2, 2), 1),
                nn.ReLU(),
                nn.Conv2d(256, 256, (3, 3), (2, 2), 1),
                nn.ReLU(),
                nn.Conv2d(256, 256, (3, 3), (2, 2), 1),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),
                nn.Flatten(),
                nn.Linear(256, 32),
                nn.Linear(32, self.output_dim)
            )

    def get_vae_latent(self, x):
        """Get latent representation from VAE (adapted from NVAE code)"""
        self.vae.eval()
        with torch.no_grad():
            if self.vanilla_vae:
                s = self.vae.stem(2 * x - 1.0)

                # perform pre-processing
                for cell in self.vae.pre_process:
                    s = cell(s)

                # run the main encoder tower
                combiner_cells_enc = []
                combiner_cells_s = []
                for cell in self.vae.enc_tower:
                    if cell.cell_type == 'combiner_enc':
                        combiner_cells_enc.append(cell)
                        combiner_cells_s.append(s)
                    else:
                        s = cell(s)

                idx_dec = 0
                ftr = self.vae.enc0(s)
                param0 = self.vae.enc_sampler[idx_dec](ftr)
                mu_q, _ = torch.chunk(param0, 2, dim=1)
                latent = mu_q
            else:
                s = self.vae.stem(2 * x - 1.0)
                # perform pre-processing
                for cell in self.vae.pre_process:
                    s = cell(s)
                # run the main encoder tower
                combiner_cells_enc = []
                combiner_cells_s = []
                for cell in self.vae.enc_tower:
                    if cell.cell_type == 'combiner_enc':
                        combiner_cells_enc.append(cell)
                        combiner_cells_s.append(s)
                    else:
                        s = cell(s)

                # reverse combiner cells and their input for decoder
                combiner_cells_enc.reverse()
                combiner_cells_s.reverse()

                idx_dec = 0
                ftr = self.vae.enc0(s)
                param0 = self.vae.enc_sampler[idx_dec](ftr)
                mu_q, log_sig_q = torch.chunk(param0, 2, dim=1)
                dist = Normal(mu_q, log_sig_q)  # for the first approx. posterior
                z, _ = dist.sample()
                log_q_conv = dist.log_p(z)

                # apply normalizing flows
                nf_offset = 0
                for n in range(self.vae.num_flows):
                    z, log_det = self.vae.nf_cells[n](z, ftr)
                    log_q_conv -= log_det
                nf_offset += self.vae.num_flows
                all_q = [dist]
                all_log_q = [log_q_conv]

                # To make sure we do not pass any deterministic features from x to decoder.
                s = 0

                # prior for z0
                dist = Normal(mu=torch.zeros_like(z), log_sigma=torch.zeros_like(z))
                log_p_conv = dist.log_p(z)
                all_p = [dist]
                all_log_p = [log_p_conv]

                idx_dec = 0
                s = self.vae.prior_ftr0.unsqueeze(0)
                batch_size = z.size(0)
                s = s.expand(batch_size, -1, -1, -1)
                for cell in self.vae.dec_tower:
                    if cell.cell_type == 'combiner_dec':
                        if idx_dec > 0:
                            # form prior
                            param = self.vae.dec_sampler[idx_dec - 1](s)
                            mu_p, log_sig_p = torch.chunk(param, 2, dim=1)

                            # form encoder
                            ftr = combiner_cells_enc[idx_dec - 1](combiner_cells_s[idx_dec - 1], s)
                            param = self.vae.enc_sampler[idx_dec](ftr)
                            mu_q, log_sig_q = torch.chunk(param, 2, dim=1)
                            dist = Normal(mu_p + mu_q, log_sig_p + log_sig_q) if self.vae.res_dist else Normal(mu_q,
                                                                                                           log_sig_q)
                            z, _ = dist.sample()
                            log_q_conv = dist.log_p(z)
                            # apply NF
                            for n in range(self.vae.num_flows):
                                z, log_det = self.vae.nf_cells[nf_offset + n](z, ftr)
                                log_q_conv -= log_det
                            nf_offset += self.vae.num_flows
                            all_log_q.append(log_q_conv)
                            all_q.append(dist)

                            # evaluate log_p(z)
                            dist = Normal(mu_p, log_sig_p)
                            log_p_conv = dist.log_p(z)
                            all_p.append(dist)
                            all_log_p.append(log_p_conv)

                        # 'combiner_dec'
                        s = cell(s, z)
                        idx_dec += 1
                    else:
                        s = cell(s)

                latent = s

            return latent

    def forward(self, x):
        if self.baseline_mode:
            if self.decoder_in_size is None:
                out = self.get_vae_latent(x)
                self.decoder_in_size = out.size()
            baseline_latent = torch.zeros(self.decoder_in_size, device=self.device)
            out = self.decoder(baseline_latent)
        else:
            latent = self.get_vae_latent(x)
            out = self.decoder(latent)
        return out
