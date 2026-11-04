"""Conditional VAE that generates extra wafer maps for rare defect classes.

The model only learns where the failing dies are (a binary 64x64 "fail map"),
conditioned on the defect class. The wafer outline (which pixels are on the
wafer at all) is borrowed from a random real wafer of the same class when
sampling, so every synthetic map is a valid 0/1/2 map with a realistic shape.

Sampling happens at the borrowed wafer's real die resolution: the decoder's
failure probabilities are averaged over the pixels of each die, every die passes
or fails once, and the die grid is scaled up to 64x64 exactly like the real maps.
That reproduces the blocky look of real maps (most wafers have ~25x27 dies)
instead of pixel-level salt-and-pepper noise.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from waferwise.data import nearest_index
from waferwise.train import random_dihedral


def down(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


def up(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class ConditionalVAE(nn.Module):
    """Encoder 64 -> 4 pixels, latent vector, decoder 4 -> 64 pixels; the class is fed to both."""

    def __init__(self, num_classes: int, latent_dim: int = 32, channels=(32, 64, 128, 256)):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        c1, c2, c3, c4 = channels
        self.encoder = nn.Sequential(down(1 + num_classes, c1), down(c1, c2), down(c2, c3), down(c3, c4), nn.Flatten())
        self.to_mu = nn.Linear(c4 * 16, latent_dim)
        self.to_logvar = nn.Linear(c4 * 16, latent_dim)
        self.from_latent = nn.Linear(latent_dim + num_classes, c4 * 16)
        self.decoder = nn.Sequential(
            up(c4, c3), up(c3, c2), up(c2, c1), nn.ConvTranspose2d(c1, 1, 4, stride=2, padding=1)
        )
        self.top_channels = c4

    def one_hot(self, y: torch.Tensor) -> torch.Tensor:
        return F.one_hot(y, self.num_classes).float()

    def encode(self, fail_map: torch.Tensor, y: torch.Tensor):
        # Give the encoder the class as extra constant image channels.
        cls = self.one_hot(y)[:, :, None, None].expand(-1, -1, *fail_map.shape[-2:])
        h = self.encoder(torch.cat([fail_map, cls], dim=1))
        return self.to_mu(h), self.to_logvar(h)

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return per-pixel failure logits (N, 1, 64, 64)."""
        h = self.from_latent(torch.cat([z, self.one_hot(y)], dim=1))
        return self.decoder(h.view(-1, self.top_channels, 4, 4))

    def forward(self, fail_map: torch.Tensor, y: torch.Tensor):
        mu, logvar = self.encode(fail_map, y)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)  # reparameterisation trick
        return self.decode(z, y), mu, logvar


def vae_loss(logits, target, mu, logvar, beta: float = 1.0):
    """Per-wafer reconstruction loss (summed over pixels) + beta * KL divergence, averaged over the batch."""
    recon = F.binary_cross_entropy_with_logits(logits, target, reduction="sum") / len(target)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(target)
    return recon + beta * kl, recon, kl


def fail_maps(X: np.ndarray) -> np.ndarray:
    """0/1/2 wafer maps -> binary fail maps (1 where a die failed)."""
    return (X == 2).astype(np.float32)


def balanced_epoch(y: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    """Indices for one epoch with the same number of samples from every class (with replacement)."""
    picks = [rng.choice(np.flatnonzero(y == c), size=per_class, replace=True) for c in np.unique(y)]
    return rng.permutation(np.concatenate(picks))


def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Independent random rotation/flip for every map in a batch (patterns are rotation invariant)."""
    return torch.stack([random_dihedral(m) for m in x])


def sample_on_die_grid(probs: np.ndarray, die_shape, rng: np.random.Generator) -> np.ndarray:
    """Turn a (S, S) pixel failure-probability map into a boolean fail map drawn die by die.

    ``die_shape`` is the original (rows, cols) of dies of the wafer whose outline is used.
    """
    size = probs.shape[0]
    h, w = int(die_shape[0]), int(die_shape[1])
    rows, cols = nearest_index(h, size), nearest_index(w, size)
    die_id = (rows[:, None] * w + cols[None, :]).ravel()  # which die each pixel shows
    sums = np.bincount(die_id, weights=probs.ravel(), minlength=h * w)
    counts = np.bincount(die_id, minlength=h * w)
    die_prob = sums / np.maximum(counts, 1)
    die_fail = rng.random(h * w) < die_prob
    return die_fail[die_id].reshape(size, size)


@torch.no_grad()
def sample_wafers(
    model: ConditionalVAE,
    class_idx: int,
    masks: np.ndarray,
    die_shapes: np.ndarray,
    n: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
    sharpen: float = 1.0,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``n`` wafer maps of one class.

    ``masks`` are boolean on-wafer masks (N_real, 64, 64) from real wafers of that class and
    ``die_shapes`` their original die-grid sizes (N_real, 2). Returns the 0/1/2 maps (uint8)
    and which real wafer each one borrowed its outline and die grid from.

    ``temperature`` scales the latent noise (lower = more typical samples). ``sharpen`` > 1 pushes
    the decoder's blurry probabilities towards 0 or 1 (p^s / (p^s + (1-p)^s)), which makes
    patterns more coherent; 1 leaves them unchanged.
    """
    model.eval()
    out, mask_ids = [], []
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        z = torch.randn(b, model.latent_dim) * temperature
        y = torch.full((b,), class_idx, dtype=torch.long)
        probs = torch.sigmoid(sharpen * model.decode(z, y)).squeeze(1).numpy()  # same as the formula above
        ids = rng.integers(0, len(masks), size=b)
        fail = np.stack([sample_on_die_grid(p, die_shapes[i], rng) for p, i in zip(probs, ids)])
        on = masks[ids]
        out.append(on.astype(np.uint8) + (fail & on).astype(np.uint8))
        mask_ids.append(ids)
    return np.concatenate(out), np.concatenate(mask_ids)
