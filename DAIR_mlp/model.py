import torch
import torch.nn as nn
from torch.distributions import Normal


# define a simple linear VAE
class LinearVAE(nn.Module):
    def __init__(self,
                 num_features: int = 2,
                 latent_dim: int = 1,
                 beta: float = 4.0,
                 conditional: bool = True,
                 cond_dim: int = 1
                 ):
        super(LinearVAE, self).__init__()
        self.beta = beta
        self.conditional = conditional
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.num_features = num_features
        self.decoder_in_dim = latent_dim
        if self.conditional:
            self.decoder_in_dim += self.cond_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(in_features=64, out_features=latent_dim)
        self.fc_var = nn.Linear(in_features=64, out_features=latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.decoder_in_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=num_features),
        )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def forward(self, x, cond_input=None):
        # encoding
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        if self.conditional:
            z = torch.cat((z, cond_input), dim=1)

        # decoding
        x = self.decoder(z)
        reconstruction = torch.sigmoid(x)
        return reconstruction, mu, log_var

    def sample(self, num_samples, cond_input):
        # sample latent
        dist = Normal(
            torch.zeros(num_samples, self.latent_dim).cuda(),
            torch.ones(num_samples, self.latent_dim).cuda()
        )
        z = dist.sample()
        if self.conditional:
            z = torch.cat((z, cond_input), dim=1)

        # decoding
        x = self.decoder(z)
        generation = torch.sigmoid(x)
        return generation

    def recon_loss(self, recon, input):
        """Reconstruction Loss"""
        mse_criterion = nn.MSELoss()
        return mse_criterion(recon, input)

    def kld_loss(self, mu, log_var):
        """KL Divergence Loss"""
        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())
        return kld_loss
