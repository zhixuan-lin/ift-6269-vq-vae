import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.distributions import TransformedDistribution, Uniform, SigmoidTransform, AffineTransform
import numpy as np
from vqvae import ResBlock, DiscretizedLogistic



"""General Instructions:
You should first understand VQVAEBase, VQVAEPrior and VQVAE in vqvae.py
For both baselines, you should use the encoder and decoder defined in vqvae.py.
For VanillaVAE:
    However, the latent variable should be 1-D. The length should be equal to
        (image_size // 4). For CIFAR10, this is 8 * 8 = 64.
    So you need an additional linear layer for encoder and decoder.
    For output likelihood, either Gaussian with fixed variance or discretized logistics.

For GumbelSoftmaxVAE:
    Basically the same as VQVAE. The only different is the `quantize` function and loss computation (it doesn't have codebook and commitment loss).
    So you can copy most of the code here.

After you implemented these things:
    Add functions in api.py, like `train_vanilla_vae` and `train_gumbelsoftmax_vae`.
    Then, add four scripts under `scripts`:
        - run_vanilla_mse_cifar10.py
        - run_vanilla_dl_cifar10.py
        - run_gumbelsoftmax_mse_cifar10.py
        - run_gumbelsoftmax_dl_cifar10.py

"""

class Encoder(nn.Module):
    """
    Identical to vqvae.Encoder except for the linear layer that outputs 1D
    embeddings. The latent variable follows a Gaussian distribution.
    """
    def __init__(self, embed_dim, n_hidden, res_hidden, image_width, loss_type):
        super().__init__()
        relu = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv2d(3, n_hidden // 2, 4, 2, 1), # 16 x 16
            relu,
            nn.Conv2d(n_hidden // 2, n_hidden, 4, 2, 1), # 8 x 8
            relu,
            nn.Conv2d(n_hidden, n_hidden, 3, 1, 1), # 8 x 8
            # No relu here because ResBlock has it
            ResBlock(n_hidden, res_hidden, n_hidden),
            ResBlock(n_hidden, res_hidden, n_hidden),
            relu,
            nn.Conv2d(n_hidden, n_hidden, 1, 1, 0),
            relu,
        )

        # Gaussian
        self.fc_mu = nn.Linear(n_hidden*(image_width//4)**2, embed_dim)
        self.fc_logvar = nn.Linear(n_hidden*(image_width//4)**2, embed_dim)

    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    """
    Identical to vqvae.Decoder except for the linear layer that takes 1D
    embeddings.
    """
    def __init__(self, embed_dim, n_hidden, res_hidden, image_width, loss_type):
        super().__init__()
        output_dim = dict(
            mse=3,
            discretized_logistic=3 * 2  # For loc and scale
        )[loss_type]

        self.n_hidden = n_hidden
        self.hid_img_width = image_width//4
        self.decoder_input = nn.Linear(embed_dim, n_hidden*(self.hid_img_width)**2)

        relu = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            # relu,
            nn.Conv2d(n_hidden, n_hidden, 3, 1, 1),
            # No relu here because ResBlock has it
            ResBlock(n_hidden, res_hidden, n_hidden),
            ResBlock(n_hidden, res_hidden, n_hidden),
            relu,
            nn.ConvTranspose2d(n_hidden, n_hidden // 2, 4, 2, 1),
            relu,
            nn.ConvTranspose2d(n_hidden // 2, output_dim, 4, 2, 1),
        )
        self.loss_type = loss_type

    def forward(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, self.n_hidden, self.hid_img_width, self.hid_img_width)
        out = self.net(z)
        return out


class VanillaVAE(nn.Module):
    def __init__(self, beta=1.0, loss_type='mse', pixel_range=256, num_embed=512,
        embed_dim=64, n_hidden=128, res_hidden=32, image_width=32,
        data_variance=0.06327039811675479):
        """
        You are free to add more arguments to ctor, but at include loss_type:

        Args:
            loss_type: 'mse' or 'discretized_logistic'. With 'mse', I really mean Gaussian.
                So don't just use implement mse, but use Gaussian likelihood, with
                a fixed scalar variance given by data_variance
        """
        super().__init__()
        assert loss_type in ['mse', 'discretized_logistic']
        self.loss_type = loss_type
        self.beta = beta    # KL divergence scale
        self.encoder = Encoder(embed_dim, n_hidden, res_hidden, image_width, loss_type)
        self.decoder = Decoder(embed_dim, n_hidden, res_hidden, image_width, loss_type)

        self.pixel_range = pixel_range
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.data_variance = data_variance

        # The linear layer is initialized by Kaiming by default.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std*eps

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def normalize(x):
        """
        [0, 255] -> [-0.5, 0.5]
        """
        return (x / 255.0) - 0.5

    @staticmethod
    def unnormalize(x):
        """
        [-0.5, 0.5] -> [0, 255]
        """
        return (x + 0.5) * 255.0

    def encode(self, x):
        """
        x: (B, C, H, W), in range (0, 255)
        """
        x = self.normalize(x)
        return self.encoder(x)   # (B, D, 8, 8)

    def decode(self, z):
        return self.decoder(z)

    @torch.no_grad()
    def decode_and_unnormalize(self, embeddings):
        """Only used in validation"""
        params = self.decoder(embeddings)
        return torch.clamp(self.unnormalize(params[:, :3, :, :]), min=0.5, max=self.pixel_range-0.5).floor()


    def forward(self, x):
        """
        Instructions:
            You should first normalize the image to [-0.5, 0.5]
            ELBO computation: this is error-prone. You should first get the elbo by summing p(x|z) across pixels
                and summing KL divergence across latent dimensions, sum these two terms together,
                and then average over the number of pixels.
        Args:
            x: float32 torch tensor, in range (0, 255)
        Returns:
            loss: this is supposed to be elbo, but averaged over number of pixels
            log: a dictionary containing scalars to be logged. Following entries are expected:
                recon_loss: mse_loss, computed with *normalized images* (in range [-0.5, 0.5])
                recon_loss_scaled: mse_loss / data_variance
                nll_output: p(x|z), averaged over number of pixels
                nll_lb: negative elbo, average over number of pixels
                bits_per_dim: nll_lb / log(2)
        """
        loss = recons_loss = None
        log = {}

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if self.loss_type == 'mse':
            recons = self.decode(z)
            recons_loss = F.mse_loss(recons, self.normalize(x))

            kld_loss = torch.mean(-0.5*torch.sum(1+logvar-mu**2-logvar.exp(), dim=1), dim=0)
            loss = recons_loss + self.beta*kld_loss
            # p(x|z)
            nll_output = recons_loss / self.data_variance + math.log(
                2 * math.pi * self.data_variance)
        elif self.loss_type == 'discretized_logistic':
            params = self.decode(z)
            # (B, 3, H, W), (B, 3, H, W)
            loc_tmp, scale_tmp = params.chunk(2, dim=1)
            # Let loc_tmp stay in (-0.5, 0.5). loc in [0, 255]
            loc = self.unnormalize(loc_tmp)
            scale = F.softplus(scale_tmp)
            likelihood_dist = DiscretizedLogistic(n_classes=self.pixel_range, loc=loc, scale=scale)
            nll_output = -likelihood_dist.log_prob(x)
            assert nll_output.size() == x.size()
            nll_output = nll_output.mean()
            kld_loss = torch.mean(-0.5*torch.sum(1+logvar-mu**2-logvar.exp(), dim=1), dim=0)
            loss = nll_output + self.beta*kld_loss

            # Logging purpose only
            # Note loc_tmp is in range (-0.5, 0.5)
            recons_loss = F.mse_loss(loc_tmp, self.normalize(x))
        else:
            raise ValueError('Invalid loss_type')

        # p(x|z) + p(z), where p(z) is Gaussian.
        nll_lb = nll_output + (-0.5*self.embed_dim*np.log(2*np.pi)
                -0.5*self.embed_dim*logvar
                -0.5/logvar.exp()*torch.einsum('bd,bd->b', z, z))

        bits_per_dim = nll_lb / np.log(2)
        log.update({
            'loss': loss,
            'recon_loss': recons_loss,
            'recon_loss_scaled': recons_loss / self.data_variance,
            'nll_output': nll_output,
            'nll_lb': nll_lb,
            'bits_per_dim': bits_per_dim,
        })

        assert all(k in log for k in ['recon_loss', 'recon_loss_scaled', 'nll_output', 'nll_lb', 'bits_per_dim'])
        return loss, log

    @torch.no_grad()
    def sample(self, n_samples):
        """
        Args:
            n_samples: number of samples
        Returns:
            torch float images with shape (B, 3, H, W). In range [0, 255]
        """
        z = torch.randn(n_samples, self.embed_dim, device=self.device)
        samples = self.decode_and_unnormalize(z)
        return samples


    @torch.no_grad()
    def reconstruct(self, x):
        """
        Args:
            x: torch float images with shape (B, 3, H, W). In range [0, 255]
        Returns:
            something like x
        """
        embeddings = self.encode(x)
        return self.decode_and_unnormalize(embeddings)


class GumbelSoftmaxVAEBase(nn.Module):
    """
    """
    pass

class GumbelSoftmaxVAEPrior(nn.Module):
    """
    """
    pass

class GumbelSoftmaxVAE(nn.Module):
    """
    """
    pass
