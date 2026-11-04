"""Train the conditional VAE on the (cleaned) training split. Results go to outputs/<experiment_name>/.

Usage:
    python scripts/train_vae.py [--config configs/cvae.yaml] [--epochs N] [--threads N]
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from waferwise.generative import ConditionalVAE, augment_batch, balanced_epoch, fail_maps, vae_loss
from waferwise.train import drop_listed_wafers, load_data
from waferwise.utils import get_logger, load_config, repo_path, save_json, set_seed

log = get_logger("train_vae")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/cvae.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    out_dir = repo_path(f"outputs/{cfg['experiment_name']}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data, splits, classes = load_data(cfg["data_config"])
    train_idx = drop_listed_wafers(splits["train"], data["source_index"], cfg["drop_list"])
    fails = fail_maps(data["X"][train_idx])
    y = data["y"][train_idx]

    model = ConditionalVAE(len(classes), **cfg["model"]).to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    log.info("cVAE: %d params, %d training wafers, %d samples per class per epoch",
             sum(p.numel() for p in model.parameters()), len(y), tcfg["samples_per_class"])

    history = []
    for epoch in range(1, tcfg["epochs"] + 1):
        start = time.time()
        beta = tcfg["beta"] * min(1.0, epoch / max(tcfg["kl_warmup_epochs"], 1))
        model.train()
        totals = np.zeros(3)
        order = balanced_epoch(y, tcfg["samples_per_class"], rng)
        for b in range(0, len(order), tcfg["batch_size"]):
            idx = order[b:b + tcfg["batch_size"]]
            x = torch.from_numpy(fails[idx]).unsqueeze(1)
            if tcfg["augment"]:
                x = augment_batch(x)
            x = x.contiguous(memory_format=torch.channels_last)
            target_y = torch.from_numpy(y[idx])
            logits, mu, logvar = model(x, target_y)
            loss, recon, kl = vae_loss(logits, x, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals += np.array([loss.item(), recon.item(), kl.item()]) * len(idx)
        loss, recon, kl = totals / len(order)
        history.append({"epoch": epoch, "beta": beta, "loss": loss, "recon": recon, "kl": kl,
                        "seconds": round(time.time() - start, 1)})
        log.info("epoch %2d  beta %.2f  loss %.1f  recon %.1f  kl %.1f  (%.0fs)",
                 epoch, beta, loss, recon, kl, history[-1]["seconds"])

    torch.save({"model_state": model.state_dict(), "config": cfg, "classes": classes}, out_dir / "model.pt")
    save_json(history, out_dir / "history.json")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    epochs = [h["epoch"] for h in history]
    axes[0].plot(epochs, [h["recon"] for h in history], linewidth=2)
    axes[0].set_title("Reconstruction loss (per wafer)", loc="left")
    axes[1].plot(epochs, [h["kl"] for h in history], linewidth=2, color="#c0392b")
    axes[1].set_title("KL divergence (per wafer)", loc="left")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e5e5e5", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)
    log.info("Saved %s", out_dir / "model.pt")


if __name__ == "__main__":
    main()
