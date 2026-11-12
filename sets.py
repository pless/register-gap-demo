"""The two maps over sets: the same two maps act on each phrase of a caption and on each patch of a frame.

score(caption, frame) = <T(caption), I(frame)> + lam * mean over the phrases p of softmax_j <T(p), I(patch_j)>
softmax_j s_j = tau * logsumexp_j(s_j / tau) - tau * log(J), with J patches.
T and I are the text map and the image map of maps.TwoMaps. lam and tau are learned; they start at 1 and 0.05.
"""
import math

import numpy as np
import torch

from evaluate import recall_of_scores
from maps import TwoMaps


def pick_device():
    """CUDA, then Apple MPS, then the CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def as_tensor(array, device, dtype=torch.float32):
    """A numpy array, a float16 array or a memory map, as a tensor on the device."""
    return torch.tensor(np.asarray(array), dtype=dtype, device=device)


class SetMaps(torch.nn.Module):
    """TwoMaps and the two learned numbers of the set score: lam (weight of the phrase-patch term) and tau (softness)."""

    def __init__(self, dim):
        super().__init__()
        self.maps = TwoMaps(dim)
        self.log_lam = torch.nn.Parameter(torch.tensor(0.0))
        self.log_tau = torch.nn.Parameter(torch.tensor(math.log(0.05)))

    def score(self, captions, phrases, mask, images, patches):
        """captions (B, d), phrases (B, P, d), mask (B, P) bool, images (C, d), patches (C, J, d). Returns (B, C)."""
        pooled = self.maps.text(captions) @ self.maps.image(images).T
        text = self.maps.text(phrases)
        image = self.maps.image(patches)
        b, p, d = text.shape
        c, j, _ = image.shape
        similarity = (text.reshape(b * p, d) @ image.reshape(c * j, d).T).reshape(b, p, c, j)
        tau = self.log_tau.exp()
        soft_max = tau * torch.logsumexp(similarity / tau, dim=3) - tau * math.log(j)
        weights = mask.float() / mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        return pooled + self.log_lam.exp() * torch.einsum("bpc,bp->bc", soft_max, weights)


def batch_score(model, captions, phrases, mask, images, patches, query_rows, frame_rows):
    """The set score (len(query_rows), len(frame_rows)) as a tensor. Phrases are cut to the longest set of the query rows."""
    device = model.log_lam.device
    p = max(1, int(mask[query_rows].sum(axis=1).max()))
    return model.score(
        as_tensor(captions[query_rows], device),
        as_tensor(phrases[query_rows, :p], device),
        as_tensor(mask[query_rows, :p], device, torch.bool),
        as_tensor(images[frame_rows], device),
        as_tensor(patches[frame_rows], device),
    )


@torch.no_grad()
def set_scores(model, captions, phrases, mask, images, patches, rows):
    """Score matrix (len(rows), len(rows)) as numpy: row i is the caption of rows[i], column j is the frame of rows[j]."""
    scores = np.zeros((len(rows), len(rows)), np.float32)
    for f in range(0, len(rows), 256):
        for q in range(0, len(rows), 64):
            block = batch_score(model, captions, phrases, mask, images, patches, rows[q:q + 64], rows[f:f + 256])
            scores[q:q + 64, f:f + 256] = block.cpu().numpy()
    return scores


@torch.no_grad()
def word_scores(model, words, images, patches):
    """words (w, d), images (n, d), patches (n, J, d). Returns (w, n) numpy: for each word and frame, the cosine with
    the pooled frame plus lam times the cosine with the best patch, after the maps."""
    device = model.log_lam.device
    text = model.maps.text(as_tensor(words, device))
    out = []
    for start in range(0, len(images), 256):
        pooled = text @ model.maps.image(as_tensor(images[start:start + 256], device)).T
        best = torch.einsum("wd,njd->wnj", text, model.maps.image(as_tensor(patches[start:start + 256], device))).amax(dim=2)
        out.append((pooled + model.log_lam.exp() * best).cpu().numpy())
    return np.concatenate(out, axis=1)


def train_set_maps(captions, phrases, mask, images, patches, train_rows, val_rows):
    """captions (n, d), phrases (n, P, d) with mask (n, P) bool, images (n, d), patches (n, J, d): float32, float16
    or a memory map. train_rows, val_rows: row numbers of the training and validation pairs.
    AdamW: the maps with learning rate 1e-4 and weight decay 1e-4, lam and tau with learning rate 1e-2.
    Batch 128, 30 epochs, contrastive loss on score / 0.05. After each epoch, measure R@10 on all validation pairs.
    Returns the SetMaps of the best epoch, on the device that trained it.
    """
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    model = SetMaps(captions.shape[1]).to(pick_device())
    optimizer = torch.optim.AdamW([
        {"params": model.maps.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
        {"params": [model.log_lam, model.log_tau], "lr": 1e-2, "weight_decay": 0.0},
    ])
    best_r10 = -1.0
    best_state = None
    for epoch in range(30):
        order = rng.permutation(train_rows)
        for start in range(0, len(order), 128):
            rows = np.sort(order[start:start + 128])
            logits = batch_score(model, captions, phrases, mask, images, patches, rows, rows) / 0.05
            target = torch.arange(len(rows), device=logits.device)
            loss = 0.5 * (torch.nn.functional.cross_entropy(logits, target) + torch.nn.functional.cross_entropy(logits.T, target))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        r10 = recall_of_scores(set_scores(model, captions, phrases, mask, images, patches, val_rows))["R@10"]
        print(f"epoch {epoch + 1:2d}: validation R@10 {r10:.3f}, lam {model.log_lam.exp().item():.2f}, tau {model.log_tau.exp().item():.4f}", flush=True)
        if r10 > best_r10:
            best_r10 = r10
            best_state = {name: value.clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model
