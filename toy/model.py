import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Adapted from https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py

    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    """
    Adapted from https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
    """

    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class LinearVAE(nn.Module):
    """
    A simple linear VAE
    """

    def __init__(
            self,
            num_features: int = 2,
            latent_dim: int = 1,
            beta: float = 1.0,
            gamma: float = 1.0,
            conditional: bool = True,
            cond_dim: int = 1,
            use_adversary=False
    ):
        super(LinearVAE, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.conditional = conditional
        self.latent_dim = latent_dim
        self.num_features = num_features
        self.x2_dim = cond_dim
        self.x1_dim = self.num_features - self.x2_dim
        self.decoder_in_dim = latent_dim
        self.use_adversary = use_adversary
        if self.conditional:
            self.decoder_in_dim += self.x2_dim

        hidden_features = num_features * 4
        self.encoder = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=hidden_features),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(in_features=hidden_features, out_features=latent_dim)
        self.fc_var = nn.Linear(in_features=hidden_features, out_features=latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.decoder_in_dim, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=num_features),
        )

        self.adversary = None
        if self.use_adversary:
            self.adversary = nn.Sequential(
                GradientReversal(),
                nn.Linear(in_features=self.latent_dim * 2, out_features=hidden_features),
                nn.ReLU(),
                nn.Linear(in_features=hidden_features, out_features=self.x2_dim),
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

        # adversary
        adv_out = None
        if self.use_adversary:
            latent = torch.cat((mu, log_var), dim=1)
            adv_out = self.adversary(latent)

        return reconstruction, mu, log_var, adv_out

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


class LinearProbe(nn.Module):
    """
    A simple linear MLP probe
    """

    def __init__(
            self,
            vae,
            num_features: int = 2,
            latent_dim: int = 1,
            conditional: bool = True,
            cond_dim: int = 1
    ):
        super(LinearProbe, self).__init__()
        self.vae = vae
        self.conditional = conditional
        self.latent_dim = latent_dim
        self.num_features = num_features
        self.x2_dim = cond_dim
        self.x1_dim = self.num_features - self.x2_dim

        hidden_features = num_features * 4
        if self.conditional:
            self.x1_probe = nn.Sequential(
                nn.Linear(in_features=self.latent_dim * 2, out_features=hidden_features),
                nn.ReLU(),
                nn.Linear(in_features=hidden_features, out_features=self.x1_dim),
            )
            self.x2_probe = nn.Sequential(
                nn.Linear(in_features=self.latent_dim * 2, out_features=hidden_features),
                nn.ReLU(),
                nn.Linear(in_features=hidden_features, out_features=self.x2_dim),
            )
        else:
            self.x1_probe = nn.Sequential(
                nn.Linear(in_features=self.latent_dim * 2, out_features=hidden_features),
                nn.ReLU(),
                nn.Linear(in_features=hidden_features, out_features=self.num_features),
            )
            self.x2_probe = None

    def forward(self, x):
        # encoding
        self.vae.eval()
        if self.vae.use_adversary:
            self.vae.adversary.eval()

        with torch.no_grad():
            x = self.vae.encoder(x)
            mu = self.vae.fc_mu(x)
            log_var = self.vae.fc_var(x)
            latent = torch.cat((mu, log_var), dim=1)

        # decoding
        if self.conditional:
            x1_out = torch.sigmoid(self.x1_probe(latent))
            x2_out = torch.sigmoid(self.x2_probe(latent))
        else:
            x1_out = torch.sigmoid(self.x1_probe(latent))
            x2_out = None

        return x1_out, x2_out
