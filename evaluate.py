"""Measures: retrieval ranks and recall, word AUC, and the heat map of a word on the patches of a frame."""
import numpy as np
from scipy.stats import rankdata


def ranks(scores):
    """scores: (n, n), row i is caption i and column j is frame j, for example text @ image.T.
    Returns (n,) int: the rank of frame i for caption i among all n frames. 1 is the best.
    A tie counts against the true frame.
    """
    true_scores = np.diag(scores)
    return (scores >= true_scores[:, None]).sum(axis=1)


def recall(text, image):
    """text, image: (n, 768), row i of each is one pair. Returns R@1, R@10, R@50 and the median rank."""
    return recall_of_scores(text @ image.T)


def recall_of_scores(scores):
    """scores: (n, n), as for ranks. Returns R@1, R@10, R@50 and the median rank."""
    r = ranks(scores)
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
    return auc_of_scores(words @ frames.T, labels)


def auc_of_scores(scores, labels):
    """scores, labels: (w, n), the score and the label of frame j for word i. Returns (w,) float, as for word_auc."""
    auc = np.zeros(len(scores))
    for i in range(len(scores)):
        r = rankdata(scores[i])
        positive = labels[i]
        n_pos = positive.sum()
        n_neg = len(positive) - n_pos
        auc[i] = (r[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def heatmap(patches, text_vector):
    """patches: (256, 768) of one frame. text_vector: (768,). Returns (16, 16): the cosine of the text with each patch."""
    return (patches.astype(np.float32) @ text_vector).reshape(16, 16)
