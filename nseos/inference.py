"""Posterior sampling restricted to the prior box (rejection instead of v1's clipping,
which piled probability mass onto the box edges and distorted the marginals)."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import EOSCatalogDataset, collate_fn, move_to_device


@torch.no_grad()
def sample_posterior_batch(model, batch, lo, hi, n_samples=1000, n_steps=50, seed=None, max_rounds=6):
    """Returns (samples (B, n, 4) float32, frac_outside_box (B,))."""
    model.eval()
    dev = model.theta_mu.device
    g = torch.Generator(device=dev).manual_seed(int(seed)) if seed is not None else None
    lo_t = torch.as_tensor(lo, dtype=torch.float32, device=dev)
    hi_t = torch.as_tensor(hi, dtype=torch.float32, device=dev)
    B = batch["num_catalogs"]
    kept = [[] for _ in range(B)]
    n_kept, n_drawn, n_in = np.zeros(B, int), np.zeros(B, int), np.zeros(B, int)
    for _ in range(max_rounds):
        need = n_samples - n_kept
        if need.max() <= 0:
            break
        n_draw = int(max(need.max() * 1.15, 64))
        post = model.sample(batch, n_samples=n_draw, n_steps=n_steps, generator=g)
        inside = ((post >= lo_t) & (post <= hi_t)).all(-1)
        post, inside = post.cpu().numpy(), inside.cpu().numpy()
        for i in range(B):
            if need[i] > 0:
                ok = post[i][inside[i]][: need[i]]
                kept[i].append(ok)
                n_kept[i] += len(ok)
                n_drawn[i] += n_draw
                n_in[i] += int(inside[i].sum())
    out = np.zeros((B, n_samples, 4), np.float32)
    for i in range(B):
        arr = np.concatenate(kept[i]) if kept[i] else np.zeros((0, 4), np.float32)
        if len(arr) < n_samples:  # practically never; flagged through frac_outside ~ 1
            fill = np.random.default_rng(i).uniform(lo, hi, size=(n_samples - len(arr), 4))
            arr = np.concatenate([arr, fill])
        out[i] = arr[:n_samples]
    return out, 1.0 - n_in / np.maximum(n_drawn, 1)


def posterior_for_samples(model, samples, lo, hi, n_samples=1000, n_steps=50, batch_size=64, seed=0,
                          allowed_types=None, reduce=None, progress=False):
    """Posterior for a list of dataset samples (their fixed catalogs).

    reduce: optional function(samples (B, n, 4), batch_thetas (B, 4)) -> dict of arrays; when given,
            only reduced summaries are kept (memory-friendly for the full test split)."""
    loader = DataLoader(EOSCatalogDataset(samples, augment=False, allowed_types=allowed_types),
                        batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    it = loader
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(loader, desc="posterior")
    posts, fouts, red = [], [], {}
    dev = model.theta_mu.device
    for bi, batch in enumerate(it):
        batch = move_to_device(batch, dev)
        p, fo = sample_posterior_batch(model, batch, lo, hi, n_samples, n_steps, seed=seed + bi)
        fouts.append(fo)
        if reduce is None:
            posts.append(p)
        else:
            for k, v in reduce(p, batch["theta"].cpu().numpy()).items():
                red.setdefault(k, []).append(v)
    fout = np.concatenate(fouts)
    if reduce is None:
        return np.concatenate(posts), fout
    return {k: np.concatenate(v) for k, v in red.items()}, fout


def posterior_for_catalogs(model, catalogs, lo, hi, n_samples=1000, n_steps=50, batch_size=64, seed=0):
    samples = [{"theta_eos": [0.0] * 4, "catalog": c} for c in catalogs]
    return posterior_for_samples(model, samples, lo, hi, n_samples, n_steps, batch_size, seed)
