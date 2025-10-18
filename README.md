# GeneJEPA: A Predictive World Model of the Transcriptome

[![Hugging Face - Model Card](https://img.shields.io/badge/HuggingFace-Model%20Card-ffcc4d?logo=huggingface&logoColor=white&labelColor=000000)](https://huggingface.co/elonlit/GeneJEPA) [![Paper](https://img.shields.io/badge/Paper-bioRxiv-1e90ff?labelColor=000000)](https://doi.org/10.1101/2025.10.14.682378)

GeneJEPA is a self-supervised foundation model for single-cell RNA-seq that learns to predict representations of masked gene sets from visible context (Joint-Embedding Predictive Architecture, JEPA). Instead of reconstructing noisy counts, the model infers latent structure over cellular state using:

- Perceiver-style encoder over unordered gene sets (fixed compute vs. variable input)
- Continuous-value tokenizer (gene identity + Fourier features for expression)
- EMA teacher targets with variance–covariance regularization

Trained on the large-scale Tahoe-100M atlas, GeneJEPA produces transferable embeddings for downstream tasks (annotation, drug response, perturbation reasoning) and supports test-time scaling by rereading more genes while keeping latent computation fixed.


## Install

Prerequisites:
- Python ≥ 3.11, PyTorch ≥ 2.6 (CUDA strongly recommended)

Using pip (editable install):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
# Optional visualizations used by callbacks
pip install umap-learn
```

Using uv (strongly recommended):

```bash
uv sync
```


## Quickstart

1) Authenticate with the Hugging Face Hub (Tahoe-100M manifests are pulled automatically):

```bash
huggingface-cli login  # or set HUGGINGFACE_HUB_TOKEN
```

2) Train (single node; Lightning auto-detects devices):

```bash
uv run -m genejepa.train
```

Notes:
- Checkpoints are saved under `checkpoints/gene_jepa_tahoe/`.
- W&B logging is enabled if `wandb` is configured; otherwise training proceeds without it.


## Export gene vocabulary map (optional)

Write the foundation gene map (and global stats) without training:

```bash
uv run -m genejepa.train \
  --export-foundation-map hf_data_cache/foundation_gene_map.parquet \
  --export-global-stats hf_data_cache/global_stats.json \
  --foundation-meta hf_data_cache/data/gene_metadata.parquet \
  --export-only
```


## Inference: embedding cells from a checkpoint

```python
import torch
from lightning.pytorch import seed_everything
from genejepa.train import JepaLightningModule

# Load your checkpoint (path under `checkpoints/gene_jepa_tahoe/`)
ckpt_path = "checkpoints/gene_jepa_tahoe/last.ckpt"
module = JepaLightningModule.load_from_checkpoint(ckpt_path)
model = module.model.eval()  # GenePerceiverJEPA

# Ragged inputs for a small batch of cells
# indices: concat of token_ids; values: log1p-standardized expression; offsets: prefix sums
indices = torch.tensor([10, 42, 7, 3, 9,  1, 2], dtype=torch.long)
values  = torch.tensor([0.1, 0.5, 0.3, 2.1, -0.2, 0.0, 1.1], dtype=torch.float32)
offsets = torch.tensor([0, 5, 7], dtype=torch.long)  # 2 samples: 5 tokens, then 2 tokens

with torch.no_grad():
    emb = model.get_embedding(indices, values, offsets, use_teacher=True)  # [batch, d]
print(emb.shape)
```


## Project structure

- `genejepa/models.py`: Perceiver encoder, predictor, JEPA wiring
- `genejepa/tokenizer.py`: Gene identity + Fourier expression tokenizer
- `genejepa/data.py`: Tahoe-100M streaming DataModule (HF Hub)
- `genejepa/train.py`: PyTorch Lightning training script and CLI utilities
- `genejepa/configs.py`: Model, training, and data configurations


## Citation

If you use GeneJEPA, please cite:

```bibtex
@article{GeneJEPA2025,
  title     = {GeneJEPA: A Predictive World Model of the Transcriptome},
  author    = {Litman, E. and Myers, T. and Agarwal, V. and Gopinath, A. and Li, O. and Mittal, E. and Kassis, T.},
  journal   = {bioRxiv},
  year      = {2025},
  publisher = {Cold Spring Harbor Laboratory},
  note      = {preprint},
  doi       = {10.1101/2025.10.14.682378},
}
```
