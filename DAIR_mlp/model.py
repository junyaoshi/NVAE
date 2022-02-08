import torch
import torch.nn as nn
from torch.distributions import Normal


# define a simple linear VAE
class LinearVAE(nn.Module):
    def __init__(
            self,
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

        hidden_features = num_features * 4
        self.encoder = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=hidden_features),
            nn.ReLU(),
            # nn.Linear(in_features=hidden_features, out_features=hidden_features),
            # nn.ReLU(),
        )

        self.fc_mu = nn.Linear(in_features=hidden_features, out_features=latent_dim)
        self.fc_var = nn.Linear(in_features=hidden_features, out_features=latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.decoder_in_dim, out_features=hidden_features),
            nn.ReLU(),
            # nn.Linear(in_features=hidden_features, out_features=hidden_features),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=num_features),
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
        mse_criterion = nn.MSELoss(reduction='sum')
        return mse_criterion(recon, input)

    def kld_loss(self, mu, log_var):
        """KL Divergence Loss"""
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return kld_loss


# define a simple linear MLP probe
class LinearProbe(nn.Module):
    def __init__(
            self,
            vae,
            adversary: bool = False,
            num_features: int = 2,
            latent_dim: int = 1,
            conditional: bool = True,
            cond_dim: int = 1
    ):
        super(LinearProbe, self).__init__()
        self.vae = vae
        self.adversary = adversary
        self.conditional = conditional
        self.latent_dim = latent_dim
        self.num_features = num_features
        self.cond_dim = cond_dim
        self.noncond_dim = self.num_features - self.cond_dim

        hidden_features = num_features * 4
        if self.conditional:
            self.cond_probe = nn.Sequential(
                nn.Linear(in_features=self.latent_dim, out_features=hidden_features),
                nn.ReLU(),
                # nn.Linear(in_features=num_features, out_features=num_features),
                # nn.ReLU(),
                nn.Linear(in_features=hidden_features, out_features=self.cond_dim),
            )
            self.noncond_probe = nn.Sequential(
                nn.Linear(in_features=self.latent_dim, out_features=hidden_features),
                nn.ReLU(),
                # nn.Linear(in_features=num_features, out_features=num_features),
                # nn.ReLU(),
                nn.Linear(in_features=hidden_features, out_features=self.noncond_dim),
            )
        else:
            self.cond_probe = None
            self.noncond_probe = nn.Sequential(
                nn.Linear(in_features=self.latent_dim, out_features=hidden_features),
                nn.ReLU(),
                # nn.Linear(in_features=num_features, out_features=num_features),
                # nn.ReLU(),
                nn.Linear(in_features=hidden_features, out_features=self.num_features),
            )

    def forward(self, x):
        # encoding
        if self.adversary:
            self.vae.train()
            x = self.vae.encoder(x)
            mu = self.vae.fc_mu(x)
        else:
            self.vae.eval()
            with torch.no_grad():
                x = self.vae.encoder(x)
                mu = self.vae.fc_mu(x)

        # decoding
        if self.conditional:
            cond_out = torch.sigmoid(self.cond_probe(mu))
            noncond_out = torch.sigmoid(self.noncond_probe(mu))
        else:
            cond_out = None
            noncond_out = torch.sigmoid(self.noncond_probe(mu))

        return cond_out, noncond_out
