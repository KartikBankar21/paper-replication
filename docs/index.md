# CLIP → SigLIP: Contrastive Vision-Language Pretraining From Scratch

A from-scratch implementation and head-to-head comparison of **CLIP** and **SigLIP**,
trained on Flickr8k, in both PyTorch and TensorFlow — no pretrained vision or text
backbones.

This page is a static, read-only view of the project. Nothing here executes — the
notebooks below are rendered exactly as they ran on Google Colab. To run either one
yourself, open it in Colab directly from the notebook page (top-right badge) or from the
[GitHub repo](https://github.com/KartikBankar21/paper-replication).

## Why this project

Built as the direct continuation of a from-scratch **Vision Transformer** replication
(learned via [learnpytorch.io](https://www.learnpytorch.io)'s paper-replicating module).
ViT answers how to represent an image with a Transformer; CLIP answers how to connect
that representation to language. The comparison against SigLIP — same architecture, same
data, only the loss function changes — turns a claim from a paper ("sigmoid loss removes
the batch-size bottleneck of softmax contrastive loss") into something you can watch
happen, or not happen, in your own training curves.

## Results summary

Flickr8k, 28,500 train / 1,500 val pairs, 8 epochs, batch size 128, single Colab T4.

| Metric (text → image) | CLIP | SigLIP |
|---|---|---|
| Recall@1 | **4.80%** | 3.53% |
| Recall@5 | **17.67%** | 14.13% |
| Recall@10 | **26.53%** | 22.00% |

CLIP outperformed SigLIP here — see the full writeup in the repo
[README](https://github.com/KartikBankar21/paper-replication#results) for why that's expected
at this batch size, not a bug.

## Notebooks

Use the tabs above to open either implementation:

- **PyTorch implementation** — the primary run, with full training curves and retrieval
  results.
- **TensorFlow / Keras implementation** — same architecture and experiment, ported to
  `tf.keras` + `tf.data`.
