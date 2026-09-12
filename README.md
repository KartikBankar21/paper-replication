# CLIP → SigLIP: Contrastive Vision-Language Pretraining From Scratch

[![Open In Colab (PyTorch)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KartikBankar21/paper-replication/blob/main/notebooks/clip_siglip_colab.ipynb)
[![Open In Colab (TensorFlow)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KartikBankar21/paper-replication/blob/main/notebooks/clip_siglip_colab_tensorflow.ipynb)

A from-scratch implementation and head-to-head comparison of **CLIP** (Radford et al., 2021)
and **SigLIP** (Zhai et al., 2023) — trained on Flickr8k, in both **PyTorch** and
**TensorFlow**, with no pretrained vision or text backbones.

**[→ View the rendered project site](https://KartikBankar21.github.io/paper-replication/)**
(static, read-only — no execution needed. Run it yourself via the Colab badges above.)

---

## Motivation

This project is the direct continuation of an earlier one: a from-scratch replication of
**Vision Transformer** ("An Image is Worth 16x16 Words: Transformers for Image Recognition
at Scale," Dosovitskiy et al., 2020), which I built while learning PyTorch through
[learnpytorch.io](https://www.learnpytorch.io) — specifically its paper-replicating module.

ViT answers "how do you represent an image with a Transformer." The natural next question
is "how do you connect that representation to language" — which is exactly what CLIP does:
two independent encoders (image, text) trained jointly so that matching image-caption pairs
land near each other in a shared embedding space, with no per-task labels required.

That question also connects directly to a parallel project I was building: an LLM-agent
system that reasons over geospatial data using LangGraph. Multimodal grounding — tying
vision to language — is the same underlying idea that makes an agent capable of reasoning
about images, maps, or documents rather than text alone. CLIP/SigLIP was the natural bridge
between "I understand how a vision Transformer works" and "I understand how modern
multimodal/agentic systems are actually built."

**Why also implement SigLIP?** CLIP's loss is a softmax over the whole batch — every
image's "correct" caption is chosen from among all captions in the batch, which means the
loss quality depends heavily on batch size (more negatives = harder, more informative task).
SigLIP (from Google DeepMind) replaces this with a per-pair sigmoid loss, removing the
batch-wide normalization entirely. Implementing both, on identical architectures and data,
was a chance to verify that difference isn't just a claim in a paper — it's something you
can watch happen in your own training curves.

## What's in this repo

Two complete, independent implementations of the same architecture and experiment:

- **PyTorch** (`notebooks/clip_siglip_colab.ipynb`)
- **TensorFlow / Keras** (`notebooks/clip_siglip_colab_tensorflow.ipynb`)

Both notebooks, from scratch:
1. Build a small **Vision Transformer** image encoder (patch embedding, learned position
   embeddings, multi-head self-attention blocks, CLS-token pooling).
2. Build a small **Transformer text encoder** (token + position embeddings, self-attention
   with padding masks, mean-pooling over non-pad tokens).
3. Project both into a shared, L2-normalized embedding space.
4. Train once with **CLIP's softmax InfoNCE loss**, once with **SigLIP's sigmoid loss** —
   same architecture, same data, only the loss function changes.
5. Evaluate with zero-shot **text→image / image→text retrieval** (Recall@1/5/10).
6. Produce a qualitative demo: type a caption, see the top-matching images.

Only the tokenizer (`distilbert-base-uncased`, via 🤗 `transformers`) is reused off the
shelf — that's text splitting, not learned weights. Every trained parameter in both
encoders starts from scratch.

## Results

Trained on [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) (28,500 train /
1,500 val image-caption pairs), 8 epochs, batch size 128, on a single Colab T4.

| Metric (text → image retrieval) | CLIP | SigLIP |
|---|---|---|
| Final train loss | 2.17 | 3.69 |
| Final val loss | 3.05 | 4.24 |
| Recall@1 | **4.80%** | 3.53% |
| Recall@5 | **17.67%** | 14.13% |
| Recall@10 | **26.53%** | 22.00% |

| Metric (image → text retrieval) | CLIP | SigLIP |
|---|---|---|
| Recall@1 | **5.67%** | 3.40% |
| Recall@5 | **16.67%** | 13.07% |
| Recall@10 | **26.87%** | 20.87% |

**CLIP outperformed SigLIP at this scale — which is worth sitting with rather than
smoothing over.** SigLIP's headline claim is that removing the softmax's batch-wide
normalization lets it train efficiently at *very* large batch sizes (the original paper
uses batches up to 32k+), where CLIP's in-batch negative sampling starts to bottleneck.
At batch size 128 and 8 epochs, that regime never kicks in — CLIP's harder, more
informative per-batch classification objective likely just converges faster with this
little data and this few steps. My read: this isn't SigLIP "losing," it's a reminder that
a technique's advantage is conditional on the regime it was designed for, and reproducing
a paper's result requires reproducing its actual operating conditions, not just its loss
function. Worth revisiting at a larger batch size as a follow-up.

## Repo structure

```
.
├── notebooks/
│   ├── clip_siglip_colab.ipynb              # PyTorch implementation
│   └── clip_siglip_colab_tensorflow.ipynb   # TensorFlow/Keras implementation
├── results/
│   └── results.json                          # metrics + config from the PyTorch run
├── docs/                                      # static site source (mkdocs)
├── mkdocs.yml
└── README.md
```

## Running it yourself

Click a Colab badge above, or:
1. Open [Google Colab](https://colab.research.google.com), `File → Upload notebook`,
   select the `.ipynb` you want.
2. `Runtime → Change runtime type → T4 GPU`.
3. Run all cells top to bottom. Flickr8k downloads automatically from the Hugging Face
   Hub on first run.

## What I learned building this

- **Contrastive learning mechanics, not just the concept.** Implementing both CLIP's
  softmax InfoNCE loss and SigLIP's sigmoid loss by hand — rather than importing either —
  made the actual difference between them (global batch normalization vs. independent
  per-pair decisions) concrete instead of abstract.
- **Debugging a silent production-style crash.** The TensorFlow version initially
  OOM-crashed on Colab with no Python traceback — just the kernel silently restarting.
  Root-causing it meant reading Colab's system logs, recognizing the OOM-kill signature,
  and tracing it to a data-pipeline design flaw: eagerly decoding the entire dataset into
  memory *and* caching a second full copy via `tf.data`'s in-memory `.cache()`. The fix
  (lazy per-row decoding + disk-based caching) is a small code change, but finding it
  required systems-level debugging, not just ML knowledge.
- **Framework differences that actually matter.** Porting the same architecture from
  PyTorch to TensorFlow surfaced real, non-cosmetic differences: channels-first vs.
  channels-last tensor layout, `autograd`/`.backward()` vs. `GradientTape`, and — the one
  that caused the crash above — how each framework's data-loading defaults handle memory
  very differently (`DataLoader`'s per-item laziness vs. `tf.data`'s eager caching).
- **A paper's claimed advantage is conditional, not universal.** Reproducing SigLIP's
  architecture doesn't reproduce its result unless you're also operating in the regime
  the paper is describing — a distinction that's easy to state and easy to overlook until
  your own numbers contradict the abstract.

## Acknowledgements

- [ViT paper](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al., 2020
- [CLIP paper](https://arxiv.org/abs/2103.00020) — Radford et al., 2021
- [SigLIP paper](https://arxiv.org/abs/2303.15343) — Zhai et al., 2023
- [learnpytorch.io](https://www.learnpytorch.io) — Daniel Bourke, for the PyTorch +
  paper-replication foundation this project builds on
- [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset
