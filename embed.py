"""Frozen SigLIP2 encoder. Text, image and per-patch embeddings, all with unit length."""
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip2-base-patch16-256"


def load_encoder(device=None):
    """Return (model, processor). If device is None, use CUDA, then Apple MPS, then the CPU."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def as_tensor(output):
    """get_text_features returns a tensor in some transformers versions and an output object in others."""
    if torch.is_tensor(output):
        return output
    return output.pooler_output


@torch.no_grad()
def embed_texts(model, processor, texts, batch_size=128):
    """texts: list of n strings, used as written. Returns (n, 768) float32, unit length."""
    out = []
    for start in range(0, len(texts), batch_size):
        tokens = processor(text=texts[start:start + batch_size], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        features = as_tensor(model.get_text_features(**tokens.to(model.device)))
        out.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def embed_images(model, processor, images, with_patches=False, batch_size=32):
    """images: list of n file paths or PIL images.
    Returns pooled (n, 768) float32, unit length.
    If with_patches is True, returns (pooled, patches). patches: (n, 256, 768) float16, unit length, one row for each 16 x 16 patch.

    SigLIP2 makes the pooled embedding with an attention-pooling head over the 256 patch tokens.
    The same head, applied to one token alone, puts that patch in the shared text-image space.
    """
    pooled = []
    patches = []
    head = model.vision_model.head
    for start in range(0, len(images), batch_size):
        batch = []
        for image in images[start:start + batch_size]:
            if not isinstance(image, Image.Image):
                image = Image.open(image)
            batch.append(image.convert("RGB"))
        pixels = processor(images=batch, return_tensors="pt")["pixel_values"].to(model.device)
        tokens = model.vision_model(pixel_values=pixels).last_hidden_state
        pooled.append(torch.nn.functional.normalize(head(tokens), dim=-1).cpu().numpy())
        if with_patches:
            n, length, dim = tokens.shape
            one_token = head(tokens.reshape(n * length, 1, dim)).reshape(n, length, dim)
            patches.append(torch.nn.functional.normalize(one_token, dim=-1).half().cpu().numpy())
    if with_patches:
        return np.concatenate(pooled), np.concatenate(patches)
    return np.concatenate(pooled)
