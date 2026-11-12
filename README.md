# A camera's own words

A camera has its own words. A frozen image-text model does not know them. Two small linear maps, trained on a few thousand (caption, frame) pairs of one camera, teach the model those words.

- Demo page: https://pless.github.io/register-gap-demo/
- Notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pless/register-gap-demo/blob/main/demo.ipynb)

## The method

- The encoder is SigLIP2-base (`google/siglip2-base-patch16-256`). It does not change.
- We add two residual linear maps, one for text and one for images: `phi(x) = normalize(x + W x + b)`. W and b start at zero, so the maps start as the frozen encoder.
- We train only the maps on the camera's own (caption, frame) pairs. The loss is a symmetric contrastive loss with temperature 0.05. The optimizer is AdamW (learning rate 1e-4, weight decay 1e-4, batch 128, 30 epochs).
- After each epoch, we measure R@10 on all validation pairs and keep the best epoch.
- We split the pairs by day, never at random. Frames from one day look alike, so a random split puts near copies of the test frames in the training data.

The maps are not a new kind of model. They are a small linear adapter on frozen embeddings, in the family of CLIP-Adapter.

## Three numbers

| Camera | Measure | Frozen | Two maps |
|---|---|---|---|
| Brooks Falls bear cam (explore.org) | AUC of the viewer word "begging" on 1,203 frames from 16 held-out days | 0.38 | 0.72 |
| Duluth harbor cam, 2012 to 2026 | R@10 on 742 test frames | 0.12 | 0.42 |
| Roatán coral cam (explore.org) | R@10 on 983 test frames | 0.06 | 0.23 |

AUC 0.5 is chance. R@10 is the share of test captions whose own frame is in the top 10 of all test frames.

## How to run

On Colab, click the badge above. Part 1 needs no GPU. On a T4, parts 1 to 4 take about 5 minutes. Part 5 (optional) trains the maps over phrases and patches.

On your own computer:

```bash
git clone https://github.com/pless/register-gap-demo
cd register-gap-demo
pip install torch transformers scipy matplotlib pillow jupyter
jupyter notebook demo.ipynb
```

Part 2 of the notebook downloads the Duluth frames (about 270 MB) from https://github.com/pless/duluth-harbor-cam-captions.

To use the code on your own pairs:

```python
from embed import embed_images, embed_texts, load_encoder
from evaluate import recall
from maps import apply_maps, train_maps

model, processor = load_encoder()
text = embed_texts(model, processor, captions)   # (n, 768)
image = embed_images(model, processor, paths)    # (n, 768)
maps = train_maps(text[train], image[train], text[val], image[val])
print(recall(*apply_maps(maps, text[test], image[test])))
```

Part 4 of the notebook has one function, `run(pairs_jsonl)`, for a JSON-lines file with the fields `image`, `caption` and `time`.

## Files

| File | What it has |
|---|---|
| `embed.py` | The frozen encoder: text, image and per-patch embeddings. |
| `maps.py` | The two maps, the loss and the training loop. |
| `evaluate.py` | Ranks, recall, word AUC and heat maps. |
| `sets.py` | The maps over phrases and patches (the set score) and their training loop. |
| `demo.ipynb` | Five parts: the bear cam, Duluth from raw frames, where a word points, your own camera, and the set score (optional). |
| `data/` | Bear cam embeddings and captions (no viewer names), 13 query words, 12 frames for the heat maps, and the set-score maps. |
| `docs/` | The demo page. |

## Limits

- Tower radio and airport cameras: one day, 123 radio transmissions. The model was at chance before and after the maps.
- Chat captions: on a kitten cam and an eagle nest cam, the captions do not tell what the frame shows. R@50 goes only from 0.13 to 0.23 and from 0.07 to 0.11.
- Plain words: "sleeping" goes from 0.87 to 0.91. The model knew this word.
- Some words lose: on the Duluth cam, "coal" goes from 0.56 to 0.52 and "iron ore pellets" from 0.58 to 0.47.
- On-screen text: about 1 in 3 Duluth frames carries the ship name as an overlay. The maps can learn to read it.
- Maps trained on full frames do not point at the part of the frame that a word names. Maps trained with the set score (`sets.py`) do.

## Maps over phrases and patches (`sets.py`)

The same two maps can also act on parts. The text map acts on each phrase of a caption, and the image map acts on each patch of a frame. The set score adds, for each phrase, its best match among the 256 patches:

```
score(caption, frame) = <T(caption), I(frame)> + lam * mean over the phrases p of softmax_j <T(p), I(patch_j)>
softmax_j s_j = tau * logsumexp_j(s_j / tau) - tau * log(J)
```

The training learns `lam` and `tau`. They start at 1 and 0.05. The rest of the recipe is the same as for the two maps, and R@10 on all validation pairs chooses the epoch. `sets.py` does not make the phrases. Part 5 of the notebook makes them with the noun chunks of spaCy.

Maps trained with the set score point at the part of a frame that a word names. On 355 held-out bear cam frames with one bear, the hottest patch for the bear code is on the bear in 79% of the frames. It is 18% for the frozen model and 19% for maps trained on full frames (chance 31%).

For full captions, the set score helps a little on the reef cams and loses on Duluth. The table gives R@10 / R@50 on the test frames:

| Camera | Two maps | Set maps |
|---|---|---|
| Roatán coral cam | 0.23 / 0.51 | 0.25 / 0.53 |
| Roatán dock cam | 0.26 / 0.59 | 0.27 / 0.60 |
| Duluth harbor cam | 0.43 / 0.75 | 0.40 / 0.71 |

On a laptop with Apple MPS, the set maps train in 2 to 4 minutes for about 4,000 pairs.

## Data

- Brooks Falls bear cam and Roatán reef cams: snapshots and viewer captions from explore.org. We removed viewer names and handles. Bear codes: Katmai National Park.
- Duluth Harbor Cam / LSMMA: frames and posts, from https://github.com/pless/duluth-harbor-cam-captions.
- To ask us to remove an item, open an issue: https://github.com/pless/register-gap-demo/issues
