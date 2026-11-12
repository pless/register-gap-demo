"""Two residual linear maps on frozen embeddings: one for text, one for images."""
import torch

from evaluate import recall


class TwoMaps(torch.nn.Module):
    """phi(x) = normalize(x + W x + b). W and b start at zero, so training starts at the frozen encoder."""

    def __init__(self, dim):
        super().__init__()
        self.text_map = torch.nn.Linear(dim, dim)
        self.image_map = torch.nn.Linear(dim, dim)
        for layer in (self.text_map, self.image_map):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def text(self, x):
        """x: (..., dim) text embeddings. Returns the same shape, unit length."""
        return torch.nn.functional.normalize(x + self.text_map(x), dim=-1)

    def image(self, x):
        """x: (..., dim) image or patch embeddings. Returns the same shape, unit length."""
        return torch.nn.functional.normalize(x + self.image_map(x), dim=-1)


def contrastive_loss(text, image, temperature=0.05):
    """text, image: (batch, dim). Row i of text matches row i of image."""
    logits = text @ image.T / temperature
    target = torch.arange(len(text), device=text.device)
    return 0.5 * (torch.nn.functional.cross_entropy(logits, target) + torch.nn.functional.cross_entropy(logits.T, target))


@torch.no_grad()
def apply_maps(maps, text, image):
    """text: (..., dim), image: (..., dim) float arrays. Returns the two mapped arrays (numpy float32, unit length)."""
    device = maps.text_map.weight.device
    text = torch.tensor(text, dtype=torch.float32, device=device)
    image = torch.tensor(image, dtype=torch.float32, device=device)
    return maps.text(text).cpu().numpy(), maps.image(image).cpu().numpy()


def train_maps(text_train, image_train, text_val, image_val):
    """text_train, image_train: (n, dim) float32. text_val, image_val: (m, dim). Row i of text matches row i of image.
    AdamW, learning rate 1e-4, weight decay 1e-4, batch 128, 30 epochs, a new shuffle each epoch.
    After each epoch, measure R@10 on all validation pairs. Return the TwoMaps of the best epoch.
    """
    torch.manual_seed(0)
    maps = TwoMaps(text_train.shape[1])
    optimizer = torch.optim.AdamW(maps.parameters(), lr=1e-4, weight_decay=1e-4)
    text_train = torch.tensor(text_train, dtype=torch.float32)
    image_train = torch.tensor(image_train, dtype=torch.float32)
    best_r10 = -1.0
    best_state = None
    for epoch in range(30):
        order = torch.randperm(len(text_train))
        for start in range(0, len(order), 128):
            rows = order[start:start + 128]
            loss = contrastive_loss(maps.text(text_train[rows]), maps.image(image_train[rows]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        r10 = recall(*apply_maps(maps, text_val, image_val))["R@10"]
        if r10 > best_r10:
            best_r10 = r10
            best_state = {name: value.clone() for name, value in maps.state_dict().items()}
    maps.load_state_dict(best_state)
    return maps
