"""
VQ-VAE. Two classes that you will use:
- VQVAE
- VQVAEPrior
"""
import torch
from torch import nn
from torch.nn import functional as F

class LayerNorm(nn.LayerNorm):
    """You need this. Layernorm only handles last several dimensions."""
    def __init__(self, channels, channel_ordered=False):
      nn.LayerNorm.__init__(self, channels // 3 if channel_ordered else channels)
      self.channel_ordered = channel_ordered

    def forward(self, input):
      """
      Args:
        input: (B, C, H, W)
      """
      B, C, H, W = input.size()
      # (B, H, W, C)
      input = input.permute(0, 2, 3, 1)
      if self.channel_ordered:
        # The only difference is that this is applied to each group separately
        input = input.view(B, H, W, 3, C // 3)
      output = nn.LayerNorm.forward(self, input)
      if self.channel_ordered:
        output = output.view(B, H, W, C)
      output = output.permute(0, 3, 1, 2)
      return output


class ResBlock(nn.Module):
    def __init__(self, nin, nout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(nin, nout, 3, stride, 1)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = LayerNorm(nout)
        self.conv2 = nn.Conv2d(nout, nout, 3, 1, 1)
        self.norm2 = LayerNorm(nout)
        self.downsample = None
        if stride != 1 or nin != nout:
            self.downsample = nn.Conv2d(nin, nout, 1, stride, 0)

    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, channel_ordered, in_channels, out_channels, kernel_size):
      """
      Args:
        mask_type: 'A': do not 'B': connects to self
      """
      assert mask_type in ['A', 'B']
      assert kernel_size % 2 == 1
      nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
      (o, i, h, w) = self.weight.size()
      
      mask = torch.zeros(o, i, h, w)
      mask[:, :, :h//2] = 1.0
      mask[:, :, h//2, :w//2] = 1.0
      if channel_ordered:
        # Center point, enforcing that requirement
        # (1, IN)
        num_in = (torch.arange(i) // (i // 3)).unsqueeze(0)
        # (OUT, 1)
        num_out = (torch.arange(o) // (o // 3)).unsqueeze(1)
        if mask_type == 'A':
          # Center pixel only
          mask[:, :, h//2, w//2] = (num_out > num_in)
        else:
          mask[:, :, h//2, w//2] = (num_out >= num_in)
      elif mask_type == 'B':
          mask[:, :, h//2, w//2] = 1.0

      self.register_buffer('mask', mask)


    def forward(self, input):
      self.weight.data *= self.mask
      out = nn.Conv2d.forward(self, input)
      return out


class ResMaskedConvBlock(nn.Module):
    """Type B by default"""
    def __init__(self, channels, kernel_size, channel_ordered):
      assert channels % 2 == 0
      assert kernel_size % 2 == 1
      nn.Module.__init__(self)
      self.conv1 = MaskedConv2d('B', channel_ordered, channels, channels//2, 1)
      # self.conv1 = nn.Conv2d(channels, channels//2, 1)
      self.conv2 = MaskedConv2d('B', channel_ordered, channels//2, channels//2, kernel_size)
      self.conv3 = MaskedConv2d('B', channel_ordered, channels//2, channels, 1)
      self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
      identity = x
      x = self.relu(self.conv1(x))
      x = self.relu(self.conv2(x))
      x = self.conv3(x)
      x += identity
      return x



class PixelCNN(nn.Module):
    def __init__(self, image_shape, channel_ordered, n_colors, n_layers, n_filters):
      nn.Module.__init__(self)
      self.image_shape = image_shape
      self.n_colors = n_colors
      _, _, channels = self.image_shape
      self.n_filters = n_filters

      self.in_conv = MaskedConv2d('A', channel_ordered, channels, n_filters, 3)
      self.in_ln = LayerNorm(n_filters, channel_ordered)
      self.blocks = nn.ModuleList()
      self.block_lns = nn.ModuleList()
      for i in range(n_layers):
        # self.blocks.append(ResMaskedConvBlock(n_filters, 7, channel_ordered))
        self.blocks.append(MaskedConv2d('B', channel_ordered, n_filters, n_filters, 3))
      
        self.block_lns.append(LayerNorm(n_filters, channel_ordered))

      # n channels, n_color possible values each
      self.output_layer1 = MaskedConv2d('B', channel_ordered, n_filters, n_filters, 1)
      self.output_layer = MaskedConv2d('B', channel_ordered, n_filters, n_colors * channels, 1)
      self.relu = nn.ReLU(inplace=True)




    def compute_logits(self, x):
      """
      Args:
        x: (B, C, H, W)
      Returns:
        logits: (B, 4, C, H, W)
      """
      
      B, C, H, W = x.size()
      # Note scaling happens here
      x = self.normalize(x)
      # Preprocessing conditino
      x = self.relu(self.in_ln(self.in_conv(x)))

      for ln, block in zip(self.block_lns, self.blocks):
        x = self.relu(ln(block(x)))
      x = self.output_layer1(x)
      logits = self.output_layer(x)

      # (B, 4C, H, W) -> (B, C, 4, H, W)
      # This step is to keep channels close. This matters for grouping in
      # color conditioned case.
      logits = logits.view(B, C, self.n_colors, H, W)
      # (B, C, 4, H, W) -> (B, 4, C, H, W)
      logits = logits.transpose(1, 2)
      return logits
      

    @property
    def device(self):
      return next(iter(self.parameters())).device
      

    def log_prob(self, data):
      """
      Args: 
        data: (B, C, H, W)
      
      Returns:
        log_prob: (B, C, H, W)
      """
      B, C, H, W = data.size()
      # (B, C, H, W) -> (B, 4, C, H, W)
      logits = self.compute_logits(data)

      # (B, C, H, W)
      log_prob = -F.cross_entropy(logits, data.long(), reduction='none')
      return log_prob
    
    def normalize(self, data):
      """
      Args:
        data: LongTensor, shape (B, C, H, W), in range [0, 3]
      Returns
        normalized: FloatTensor, shape (B, C, H, W), in range [-1, 1]
      """
      return data / (self.n_colors - 1) * 2 - 1 

    def forward(self, data):
      """
      Args: 
        data: (B, H, W, 1)

      Return:
        loss: (B,)
      """
      # (B, C, H, W)
      log_prob = self.log_prob(data)
      # (B, C, H, W) -> (B,)
      loss = -log_prob.flatten(start_dim=1).mean(dim=-1)
      assert loss.size() == (data.size(0),)
      return loss, {'loss': loss}

    @torch.no_grad()
    def sample(self, num_samples):
      H, W, C = self.image_shape
      data = torch.zeros(num_samples, C, H, W, device=self.device)
      # from tqdm import trange, tqdm
      # pbar = tqdm(total=H*W)
      for i in range(H):
        for j in range(W):
          for k in range(C):
            # logit: (B, 4, C, H, W)
            logit = self.compute_logits(data)[:, :, k, i, j]
            # (B)
            value = torch.distributions.Categorical(logits=logit).sample()
            data[:, k, i, j] = value
          # pbar.update(1)
      return data

class VQVAEPrior(PixelCNN):
    """Almost the same, but with an embedding layer"""
    def __init__(self,  image_shape, channel_ordered, n_colors, n_layers, n_filters, embedding_dim=64):
        super().__init__(image_shape, channel_ordered, n_colors, n_layers, n_filters)
        self.embedding = nn.Embedding(num_embeddings=128, embedding_dim=embedding_dim)
        self.in_conv = MaskedConv2d('A', channel_ordered, embedding_dim, n_filters, 3)
      
    def embed(self, x):
        # (B, 1, H, W)
        assert torch.all((0 <= x) & (x <= 127))
        x = x.long().squeeze(dim=1)
        # (B, H, W, D)
        embeddings = self.embedding(x)
        embeddings = embeddings.permute(0, 3, 1, 2)
        return embeddings
    
    def compute_logits(self, x):
        B, C, H, W = x.size()
        x = self.embed(x)
        
        x = self.relu(self.in_ln(self.in_conv(x)))

        for ln, block in zip(self.block_lns, self.blocks):
            x = self.relu(ln(block(x)))
        x = self.output_layer1(x)
        logits = self.output_layer(x)

        # (B, 4C, H, W) -> (B, C, 4, H, W)
        # This step is to keep channels close. This matters for grouping in
        # color conditioned case.
        logits = logits.view(B, C, self.n_colors, H, W)
        # (B, C, 4, H, W) -> (B, 4, C, H, W)
        logits = logits.transpose(1, 2)
        return logits

    def normalize(self, data):
        raise NotImplementedError()

class VQVAEBase(nn.Module):
    def __init__(self, beta=0.25):
        super().__init__()
        relu = nn.ReLU(inplace=True)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 4, 2, 1), # 16 x 16
            LayerNorm(256),
            relu,
            nn.Conv2d(256, 256, 4, 2, 1), # 8 x 8
            LayerNorm(256),
            relu,
            ResBlock(256, 256),
            ResBlock(256, 256),
            nn.Conv2d(256, 256, 1, 1, 0)
        )
        self.decoder = nn.Sequential(
            ResBlock(256, 256),
            ResBlock(256, 256),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            LayerNorm(256),
            relu,
            nn.ConvTranspose2d(256, 3, 4, 2, 1),
            nn.Tanh()
        )
        # (K, D)
        self.codebook = nn.Embedding(128, 256)
        self.codebook.weight.data.uniform_(-1 / 128, 1 / 128)
        self.beta = beta

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, x):
        """
        x: (B, C, H, W), in range (0, 255)
        """
        # (B, D, 8, 8)
        x = (x - 128) / 128
        encoded = self.encoder(x)
        B, D, H, W = encoded.size()
        encoded = encoded.view(B, D, H * W, 1)
        # NN
        weight = self.codebook.weight.view(1, D, 1, 128)
        # (B, D, H*W, K) -> (B, H*W, K)
        squared_distance = ((encoded - weight) ** 2).sum(dim=1)
        # (B, H*W, K) -> (B, H, W)
        indices = torch.argmin(squared_distance, dim=2).view(B, H, W)
        # (B, H, W, D)
        embeddings = self.codebook(indices)
        embeddings = embeddings.permute(0, 3, 1, 2)
        encoded = encoded.view(B, D, H, W)
        assert embeddings.size() == (B, D, H, W)
        return indices, encoded, embeddings

    
    def decode(self, encoded, embeddings):
        # ST gradient. Value is embeddings, but grad flows to encoded.
        input = encoded + (embeddings - encoded).detach()
        return self.decoder(input)
    
    @torch.no_grad()
    def reconstruct(self, x):
        _, encoded, embeddings = self.encode(x)
        return 128 * self.decode(encoded, embeddings) + 128

    def forward(self, x):
        
        _, encoded, embeddings = self.encode(x)
        recon = self.decode(encoded, embeddings)
        recon_loss = F.mse_loss(recon, (x - 128) / 128, reduction='none').mean(dim=[1, 2, 3])

        codebook_loss = F.mse_loss(embeddings, encoded.detach(), reduction='none').mean(dim=[1, 2, 3])
        commitment_loss = F.mse_loss(encoded, embeddings.detach(), reduction='none').mean(dim=[1, 2, 3])

        assert recon_loss.size() == codebook_loss.size() == commitment_loss.size()

        loss = recon_loss + codebook_loss + self.beta * commitment_loss
        log = {
            'loss': loss,
            'recon_loss': recon_loss,
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss
        }
        return loss, log



class VQVAE:
    def __init__(self, base, prior):
        # (H, W, C)
        self.base = base
        self.prior = prior
    
    @torch.no_grad()
    def sample(self, n_samples):
        # (B, 1, H, W)
        indices = self.prior.sample(n_samples)
        assert torch.all((0 <= indices) & (indices <= 128 - 1))
        # (B, H, W)
        indices = indices.long().squeeze(dim=1)
        # (B, H, W, D)
        embeddings = self.base.codebook(indices)
        # (B, D, H, W)
        embeddings = embeddings.permute(0, 3, 1, 2)
        samples = self.base.decoder(embeddings) * 128 + 128
        return samples

    @torch.no_grad()
    def reconstruct(self, *args, **kargs):
        return self.base.reconstruct(*args, **kargs)

@torch.no_grad()
def create_indices_dataset(dataloader, vqvae_base, device):
    indices_data = []
    for imgs in dataloader:
        # indices: long (B, H, W)
        indices, _, _ = vqvae_base.encode(imgs.to(device))
        indices_data.append(indices.unsqueeze(dim=1).cpu().float())
    indices_data = torch.cat(indices_data, dim=0)
    return indices_data
