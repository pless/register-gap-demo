"""Measures: retrieval ranks and recall, word AUC, and the heat map of a word on the patches of a frame."""
import numpy as np
from scipy.stats import rankdata


def ranks(text, image):
    """text, image: (n, 768) unit length. Row i of text is the caption of frame i.
    Returns (n,) int: the rank of the true frame for each caption among all n frames. 1 is the best.
    A tie counts against the true frame.
    """
    scores = text @ image.T
    true_scores = np.diag(scores)
    return (scores >= true_scores[:, None]).sum(axis=1)


def recall(text, image):
    """text, image: (n, 768), row i of each is one pair. Returns R@1, R@10, R@50 and the median rank."""
    r = ranks(text, image)
    return {
        "R@1": float(np.mean(r <= 1)),
        "R@10": float(np.mean(r <= 10)),
        "R@50": float(np.mean(r <= 50)),
        "median rank": float(np.median(r)),
    }


def word_auc(words, frames, labels):
    """words: (w, 768). frames: (n, 768). labels: (w, n) bool, True if a caption of frame j contains word i.
    Returns (w,) float: for each word, the chance that a positive frame scores above a negative frame (rank AUC).
    """
    scores = words @ frames.T
    auc = np.zeros(len(words))
    for i in range(len(words)):
        r = rankdata(scores[i])
        positive = labels[i]
        n_pos = positive.sum()
        n_neg = len(positive) - n_pos
        auc[i] = (r[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def heatmap(patches, text_vector):
    """patches: (256, 768) of one frame. text_vector: (768,). Returns (16, 16): the cosine of the text with each patch."""
    return (patches.astype(np.float32) @ text_vector).reshape(16, 16)
