import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.distributions import TransformedDistribution, Uniform, SigmoidTransform, AffineTransform
import numpy as np



"""General Instructions:
You should first understand VQVAEBase, VQVAEPrior and VQVAE in vqvae.py
For VanillaVAE:
    You should use the encoder and decoder defined in vqvae.py.
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

class VanillaVAE(nn.Module):
    def __init__(self, loss_type='mse', data_variance=0.06327039811675479):
        """
        You are free to add more arguments to ctor, but at include loss_type:

        Args:
            loss_type: 'mse' or 'discretized_logistic'. With 'mse', I really mean Gaussian.
                So don't just use implement mse, but use Gaussian likelihood, with
                a fixed scalar variance given by data_variance
        """
        super().__init__()
        pass

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
        loss = None
        log = {}

        assert all(k in log for k in ['recon_loss', 'recon_loss_scaled', 'nll_output', 'nll_lb', 'bit_per_dim'])
        return loss, log

    @torch.no_grad()
    def sample(self, n_samples):
        """
        Args:
            n_samples: number of samples
        Returns:
            torch float images with shape (B, 3, H, W). In range [0, 255]
        """
        pass

    @torch.no_grad()
    def reconstruct(self, x):
        """
        Args:
            x: torch float images with shape (B, 3, H, W). In range [0, 255]
        Returns:
            something like x
        """
        pass

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
