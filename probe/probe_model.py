from model import AutoEncoder
import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(
            self,
            vae,
            output_dim
    ):
        super(Probe, self).__init__()
        # Define model architecture

        # Output: 6 for target_info, 4 for robot_type and 4 for robot_state
        self.output_dim = output_dim
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 64, (3, 3), (2, 2), 1),
            nn.ReLU(),
            nn.AvgPool2d(4, 1),
            nn.Flatten(),
            nn.Linear(64, self.output_dim)
        )

        # pretrained vae encoder
        self.vae = vae

    def forward(self, x):
        # get latent mu from vae encoder
        self.vae.eval()
        with torch.no_grad():
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

        out = self.decoder(mu_q)
        return out
