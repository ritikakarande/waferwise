"""Tests for the conditional VAE and synthetic sampling (no dataset needed)."""

import numpy as np
import torch

from waferwise.generative import (
    ConditionalVAE,
    balanced_epoch,
    fail_maps,
    sample_on_die_grid,
    sample_wafers,
    vae_loss,
)


def small_model():
    return ConditionalVAE(num_classes=9, latent_dim=8, channels=(8, 8, 16, 16))


def test_forward_shapes_and_finite_loss():
    model = small_model()
    x = torch.randint(0, 2, (4, 1, 64, 64)).float()
    y = torch.tensor([0, 3, 5, 8])
    logits, mu, logvar = model(x, y)
    assert logits.shape == (4, 1, 64, 64)
    assert mu.shape == logvar.shape == (4, 8)
    loss, recon, kl = vae_loss(logits, x, mu, logvar)
    assert torch.isfinite(loss) and recon > 0 and kl >= 0


def test_samples_are_valid_maps_inside_the_borrowed_wafer_outline():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:64, :64]
    disc = np.hypot(yy - 31.5, xx - 31.5) < 28
    masks = np.stack([disc, np.roll(disc, 3, axis=1)])
    die_shapes = np.array([[26, 26], [40, 30]])
    X, mask_ids = sample_wafers(small_model(), class_idx=5, masks=masks, die_shapes=die_shapes, n=10, rng=rng,
                                batch_size=4)
    assert X.shape == (10, 64, 64) and X.dtype == np.uint8
    assert set(np.unique(X)) <= {0, 1, 2}
    for wafer, m in zip(X, mask_ids):
        assert ((wafer > 0) == masks[m]).all()  # on-wafer pixels are exactly the real outline


def test_balanced_epoch_draws_every_class_equally():
    y = np.array([0] * 100 + [1] * 5 + [2] * 20)
    idx = balanced_epoch(y, per_class=50, rng=np.random.default_rng(0))
    assert np.array_equal(np.bincount(y[idx]), [50, 50, 50])


def test_fail_maps_marks_only_failing_dies():
    X = np.array([[[0, 1, 2], [2, 1, 0]]], dtype=np.uint8)
    assert np.array_equal(fail_maps(X), [[[0, 0, 1], [1, 0, 0]]])


def test_die_grid_sampling_is_blocky_like_real_maps():
    # A 16x16-die wafer shown at 64x64: every 4x4 block of pixels is one die and must be all-or-nothing.
    fail = sample_on_die_grid(np.full((64, 64), 0.5), (16, 16), np.random.default_rng(0))
    blocks = fail.reshape(16, 4, 16, 4)
    assert (blocks.min(axis=(1, 3)) == blocks.max(axis=(1, 3))).all()
    assert 0.3 < fail.mean() < 0.7
