import os
import math
import logging
import itertools
import io
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse

from lightning.pytorch.strategies import DDPStrategy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Callback

from torch.utils.data import DataLoader, IterableDataset
from ema_pytorch import EMA
import numpy as np
from lightning.pytorch.loggers import WandbLogger
import wandb

try:
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import umap
except ImportError:
    print("WARNING: Libs for EmbeddingQualityValidator not found. 'pip install pandas seaborn matplotlib-base umap-learn'. Callback will be disabled.")
    pd, sns, plt, umap = None, None, None, None

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

import os
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import lightning as L
from dataclasses import dataclass
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
import json
import logging
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import lightning as L

# --- Dependency and Configuration Placeholders ---
try:
    from datasets import load_dataset, DownloadConfig
    from huggingface_hub import HfApi, hf_hub_url
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    raise ImportError("This module requires `datasets` and `huggingface_hub`. Please install them.")

try:
    from datasets import load_dataset, Dataset, Features, Value, Sequence, ClassLabel
    from huggingface_hub import HfApi, hf_hub_url, hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    raise ImportError(
        "This module requires `datasets` and `huggingface_hub`. "
        "Please install them with 'pip install datasets huggingface_hub'"
    )
    
from dataclasses import dataclass


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

from .configs import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    ExperimentConfig,
)
from .tokenizer import scRNATokenizer
from .data import Tahoe100MDataModule
from .models import (
    GenePerceiverEncoder,
    MLPPredictor,
    GenePerceiverJEPA,
)

# ==============================================================================
# 2. MODEL ARCHITECTURE
# ==============================================================================

from .callbacks import (
    LinearProbeMLP,
    EmbeddingQualityValidator,
    SupervisedValidatorCallback,
)


# ==============================================================================
# 5. PYTORCH LIGHTNING MODULE
# ==============================================================================
class JepaLightningModule(L.LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainingConfig,
        exp_config: ExperimentConfig,
        total_steps: int
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = GenePerceiverJEPA(model_config)
        self.total_steps = total_steps

        # Cache for validation callback (can be disabled by setting to 0)
        self._validation_cache: List[Dict[str, Any]] = []
        self.num_batches_to_cache = max(0, int(self.hparams.exp_config.validation_num_batches))
        if self.num_batches_to_cache > 0:
            log.info(f"JepaLightningModule configured to cache {self.num_batches_to_cache} batches for validation callbacks.")

        # -----------------------------
        # STABILITY GATE – ROBUST DESIGN
        # -----------------------------
        # Buffers move with device and get saved in checkpoints.
        self.register_buffer("teacher_dispersion_ema", torch.tensor(0.0))  # EMA of dispersion metric (scale-invariant)
        self.register_buffer("stable_score", torch.tensor(0.0))            # patience accumulator (float)
        self.register_buffer("is_teacher_stable", torch.tensor(False, dtype=torch.bool))

        # Gate hyperparams (tuned to be conservative but not sticky)
        self.pre_stability_sim_scale = 0.0    # 10% of the final sim weight before stability
        self.dispersion_threshold = 0.07       # gate opens when dispersion (1 - mean cos) > 0.10
        self.patience_steps = 300              # number of "good" steps required to flip
        self.patience_decay = 0.95             # decay instead of hard reset on bad steps
        self.ema_momentum = 0.99               # EMA for stability metric
        self.safety_flip_after = max(1, int(0.10 * max(self.total_steps, 1)))  # force-open after 10% steps

        # Make sure teacher is a true copy and frozen at train start (device-safe)
        self.register_buffer("did_initial_teacher_sync", torch.tensor(False, dtype=torch.bool))

    def _log_tokenizer_grad_balance(self):
        """
        Logs gradient norms flowing through the tokenizer's identity vs value branches.
        Use this to check whether the model is actually learning from expression values.
        """
        try:
            tok = self.model.student_encoder.tokenizer

            # Identity branch gradient (embedding table)
            id_g = getattr(tok.identity_embed.weight, "grad", None)
            id_norm = float(id_g.norm().item()) if (id_g is not None) else 0.0

            # Value branch gradient (sum of parameter grads in the value encoder)
            val_sq = 0.0
            for p in tok.value_encoder.parameters():
                if p.grad is not None:
                    val_sq += float(p.grad.detach().pow(2).sum().item())
            val_norm = val_sq ** 0.5

            # Log to Lightning (and W&B if enabled)
            self.log("debug/grad_norm_identity", id_norm, on_step=True, prog_bar=False, sync_dist=True)
            self.log("debug/grad_norm_value",    val_norm, on_step=True, prog_bar=False, sync_dist=True)

            # Occasional human-readable print on rank 0
            if (self.global_step % 50 == 0) and self.trainer.is_global_zero:
                ratio = id_norm / (val_norm + 1e-12)
                print(f"[GRAD] identity:{id_norm:.3e} value:{val_norm:.3e} ratio:{ratio:.2f}")
        except Exception:
            # Never break training because of debug logs
            pass

    def on_after_backward(self):
        # Called after loss.backward(); safe place to read gradients
        self._log_tokenizer_grad_balance()

    def _cosine_loss(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # stop-grad target occurs at call site
        p = F.normalize(p.float(), dim=1, eps=1e-6)
        t = F.normalize(t.float(), dim=1, eps=1e-6)
        return 1.0 - (p * t).sum(dim=1).mean()

    def _collapse_metrics(self, x: torch.Tensor):
        """
        Returns (avg_pairwise_cosine, std_of_vector_norms) for a batch of embeddings.
        Subsamples to <=1024 rows for speed.
        """
        if x.numel() == 0:
            z = torch.tensor(0.0, device=x.device)
            return z, z

        x = x.float()
        n = x.shape[0]
        if n > 1024:
            idx = torch.randperm(n, device=x.device)[:1024]
            x = x.index_select(0, idx)
            n = x.shape[0]

        # Norm dispersion (should NOT go to 0)
        norms = torch.linalg.norm(x, dim=1)
        std_norm = norms.std(unbiased=False)

        # Avg pairwise cosine (should NOT go to ~1.0)
        x = F.normalize(x, p=2, dim=1, eps=1e-6)
        if n > 1:
            sims = x @ x.T
            avg_cos = (sims.sum() - n) / (n * (n - 1))
        else:
            avg_cos = torch.tensor(0.0, device=x.device)

        return avg_cos, std_norm


    def _calculate_vicreg_loss(self, p: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = p.to(torch.float32)
        t = t.to(torch.float32)

        # Invariance
        sim_loss = F.mse_loss(p, t)

        # Variance
        p_std = torch.sqrt(p.var(dim=0, unbiased=False) + 1e-4)
        var_loss = torch.mean(F.relu(1.0 - p_std))

        # Covariance (off-diagonal of p only, normalized by dim)
        batch_size = p.shape[0]
        if batch_size < 2:
            cov_loss = torch.tensor(0.0, device=p.device, dtype=p.dtype)
        else:
            p_centered = p - p.mean(dim=0)
            p_cov = (p_centered.T @ p_centered) / (batch_size - 1)
            cov_loss = (p_cov.pow(2).sum() - torch.diagonal(p_cov).pow(2).sum()) / p.shape[1]

        return sim_loss, var_loss, cov_loss

    def on_train_start(self):
        """
        Ensures the teacher model is an exact copy of the student at the beginning of training.
        """
        if self.global_step == 0:
            log.info("Performing initial hard sync of teacher model from student.")
            teacher_model = self.model.teacher_encoder.ema_model
            student_model = self.model.student_encoder

            teacher_model.load_state_dict(student_model.state_dict())
            teacher_model.to(self.device)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False

            self.did_initial_teacher_sync.fill_(True)
            log.info(f"Teacher model synced, frozen, and moved to {self.device}.")

    def training_step(self, batch: Dict, batch_idx: int) -> Dict[str, torch.Tensor]:
        # Remove metadata from loss path
        _ = batch.pop("metadata", None)

        # Forward
        pred, target_raw, student_ctx = self.model(**batch)

        # Handle empty micro-batches (after masking/filtering)
        if pred.numel() == 0 or target_raw.numel() == 0:
            # Ensure all ranks hit the same distributed logging calls this step.
            zero = torch.tensor(0.0, device=self.device)

            # Always-on training metrics
            self.log_dict(
                {
                    "train/loss": zero,
                    "train_loss/sim": zero,
                },
                on_step=True, on_epoch=False, prog_bar=True, sync_dist=True
            )

            # The periodic block also uses sync_dist=True on other ranks every ~20 steps.
            if self.global_step % 20 == 0:
                self.log_dict(
                    {
                        "train_quality/avg_cosine_pred": zero,
                        "train_quality/norm_std_pred": zero,
                        "train_quality/avg_cosine_tgt_precenter": zero,
                        "train_quality/norm_std_tgt_precenter": zero,
                    },
                    on_step=True, on_epoch=False, sync_dist=True
                )

            # Touch a param so optimizer/comm buckets align, but keep loss = 0
            any_param = next(self.model.student_encoder.parameters())
            return {"loss": any_param.sum() * 0.0}

        # ---------- Target prep ----------
        target_for_loss = target_raw

        # ---------- Losses ----------
        sim_loss = self._cosine_loss(pred, target_for_loss.detach())

        # VICReg anti-collapse on predictions
        _, var_loss, cov_loss = self._calculate_vicreg_loss(pred, pred)

        # Extra stability on student context (as in your original)
        _, var_s, cov_s = self._calculate_vicreg_loss(student_ctx, student_ctx)

        # >>> DEBUG 6b: log teacher target dispersion
        cache = getattr(self.model, "_debug_cache", {})
        if "teacher_avg_cos_targets" in cache:
            # Log locally on rank 0 only; avoid distributed reduction on conditional keys
            self.log(
                "debug/teacher_avg_cos_on_targets",
                float(cache["teacher_avg_cos_targets"]),
                on_step=True, prog_bar=False, sync_dist=False, rank_zero_only=True
            )
            self.log(
                "debug/teacher_stdnorm_on_targets",
                float(cache["teacher_stdnorm_targets"]),
                on_step=True, prog_bar=False, sync_dist=False, rank_zero_only=True
            )

        total_loss = (
            self.hparams.train_config.sim_coeff * sim_loss
            + self.hparams.train_config.var_coeff * var_loss
            + self.hparams.train_config.cov_coeff * cov_loss
            + 20.0 * var_s
            + 1.0 * cov_s
        )

        self.log_dict(
            {"train/loss": total_loss, "train_loss/sim": sim_loss},
            on_step=True, on_epoch=False, prog_bar=True, sync_dist=True
        )

        # Student diagnostics
        avg_cos_s, std_s = self._collapse_metrics(student_ctx.detach())
        self.log_dict({
            "train_quality/avg_cosine_student_ctx": avg_cos_s,
            "train_quality/norm_std_student_ctx":   std_s,
        }, on_step=True, prog_bar=False)

        # ---------- Diagnostics (every ~20 steps) ----------
        if self.global_step % 20 == 0:
            # (1) Pred metrics
            avg_cos_p, std_p = self._collapse_metrics(pred.detach())
            # (2) Teacher metrics BEFORE any centering
            avg_cos_t_pre, std_t_pre = self._collapse_metrics(target_raw.detach())

            logs = {
                "train_quality/avg_cosine_pred": avg_cos_p,
                "train_quality/norm_std_pred":   std_p,
                "train_quality/avg_cosine_tgt_precenter": avg_cos_t_pre,
                "train_quality/norm_std_tgt_precenter":   std_t_pre,
            }

            self.log_dict(logs, on_step=True, on_epoch=False, sync_dist=True)

        try:
            lin0 = self.model.predictor.head[1]  # first Linear after LayerNorm
            w = lin0.weight.detach()
            self.log_dict({
                "debug/pred_w_norm_input_cols": w.norm().item(),
            }, on_step=True, prog_bar=False, sync_dist=True)
        except Exception:
            pass


        return {"loss": total_loss}

    def validation_step(self, batch: Dict, batch_idx: int):
        metadata = batch.get("metadata", [])
        if len(self._validation_cache) < self.num_batches_to_cache:
            if batch["indices"].numel() > 0:
                with torch.no_grad():
                    embeddings = self.model.get_embedding(batch["indices"], batch["values"], batch["offsets"], use_teacher=True)
                self._validation_cache.append({"embeddings": embeddings.cpu(), "metadata": metadata})

        _ = batch.pop("metadata", None)
        pred, target, _student_context = self.model(**batch)

        if pred.numel() == 0 or target.numel() == 0:
            # Log zeros so all ranks emit the same metric keys for the epoch reduction
            zero = torch.tensor(0.0, device=self.device)
            self.log_dict(
                {
                    "val/loss": zero,
                    "val_loss": zero,        # checkpoint monitor
                    "val_loss/sim": zero,
                },
                on_step=False, on_epoch=True, sync_dist=True
            )
            return

        # No centering in validation
        with torch.no_grad():
            target = target

        # BYOL-style cosine loss for validation
        val_sim = self._cosine_loss(pred, target.detach())
        self.log_dict(
            {
                "val/loss": val_sim,
                "val_loss": val_sim,  # checkpoint monitor
                "val_loss/sim": val_sim,
            },
            on_step=False, on_epoch=True, sync_dist=True
        )
        
    def on_train_batch_start(self, batch: Any, batch_idx: int):
        if self.total_steps > 0:
            progress = self.global_step / self.total_steps
            ema_end = min(self.hparams.model_config.ema_end_decay, 0.9999)
            decay = ema_end - \
                    (ema_end - self.hparams.model_config.ema_start_decay) * \
                    (math.cos(math.pi * progress) + 1) / 2
            self.model.teacher_encoder.beta = decay
            self.log("hyper/ema_decay", decay, on_step=True, on_epoch=False, rank_zero_only=True)

    def on_train_batch_end(self, outputs: Dict[str, Any], batch: Any, batch_idx: int):
        if (not self.did_initial_teacher_sync.item()) and (self.global_step == 0):
            log.info("Performing initial hard sync of teacher model from student.")
            teacher_model = self.model.teacher_encoder.ema_model
            student_model = self.model.student_encoder
            teacher_model.load_state_dict(student_model.state_dict())
            teacher_model.to(self.device).eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
            self.did_initial_teacher_sync.fill_(True)

        # --- EMA warmup gate ---
        if self.global_step >= self.hparams.model_config.ema_warmup_steps:
            with torch.no_grad():
                self.model.update_teacher()

    def on_validation_epoch_start(self):
        self._validation_cache.clear()

    def configure_optimizers(self) -> Dict:
        """
        AdamW with NO weight decay on:
        - all biases
        - all normalization parameters (e.g., LayerNorm weight/bias)
        - embedding tables (names containing 'embed')
        Weight decay is applied to everything else.

        We ONLY optimize trainable parts of the student pathway:
        - self.model.student_encoder
        - self.model.predictor
        
        The LR scheduler matches your previous cosine schedule with warmup.
        """
        # -------- 1) Collect the trainable modules we actually optimize --------
        modules_to_optimize = [
            ("student_encoder", self.model.student_encoder),
            ("predictor",       self.model.predictor),
        ]

        # -------- 2) Split parameters into decay / no_decay groups ------------
        decay_params: List[torch.nn.Parameter] = []
        no_decay_params: List[torch.nn.Parameter] = []

        # We may visit names across multiple modules; ensure uniqueness
        seen: set = set()

        def is_no_decay_param(param_name: str, p: torch.nn.Parameter) -> bool:
            name_l = param_name.lower()
            # No weight decay if:
            #  - 1D parameter (includes norm scale & ALL biases)
            #  - explicit bias (safety check; most biases are 1D already)
            #  - any normalization param (layer names containing 'norm')
            #  - any embedding table (names containing 'embed')
            return (
                p.ndim == 1
                or param_name.endswith("bias")
                or ("norm" in name_l)
                or ("embed" in name_l)
            )

        for prefix, module in modules_to_optimize:
            for name, p in module.named_parameters(recurse=True):
                if not p.requires_grad:
                    continue
                full_name = f"{prefix}.{name}"
                if full_name in seen:
                    continue
                seen.add(full_name)

                if is_no_decay_param(full_name, p):
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)

        # Safety: make sure we actually collected something
        if len(decay_params) + len(no_decay_params) == 0:
            raise RuntimeError("No parameters collected for optimization. "
                            "Check requires_grad flags and module wiring.")

        # -------- 3) Build AdamW with param groups ----------------------------
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params,    "weight_decay": self.hparams.train_config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.train_config.learning_rate,
            betas=self.hparams.train_config.adam_betas,
        )

        # -------- 4) Cosine schedule with linear warmup -----------
        warmup_steps = int(self.hparams.train_config.warmup_ratio * self.total_steps)

        def lr_lambda(step: int) -> float:
            # Guard: if total_steps not known, keep LR = 1.0
            if self.total_steps <= 0:
                return 1.0
            # Linear warmup
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            # Cosine decay
            progress = float(step - warmup_steps) / float(max(1, self.total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["ema_state_dict"] = self.model.teacher_encoder.state_dict()
        # Persist the vocab order if available from the DataModule
        try:
            dm_gene_map = getattr(self.trainer.datamodule, "gene_map", None)
            if dm_gene_map:
                if isinstance(dm_gene_map, dict):
                    # store as ordered list by token_id
                    ordered = [sym for sym, _id in sorted(dm_gene_map.items(), key=lambda kv: kv[1])]
                else:
                    ordered = list(dm_gene_map)
                checkpoint["foundation_gene_list"] = [str(s).upper() for s in ordered]
        except Exception:
            pass
        log.info("Saved EMA teacher state to checkpoint.")

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "ema_state_dict" in checkpoint:
            self.model.teacher_encoder.load_state_dict(checkpoint["ema_state_dict"])
            log.info("Loaded EMA teacher state from checkpoint.")
        else:
            log.warning("Could not find 'ema_state_dict' in checkpoint. Teacher model may be randomly initialized.")

# ==============================================================================
# 6. EXPORT HELPERS (metadata-only path)
# ==============================================================================
def _normalize_symbol(s: str) -> str:
    return str(s).upper()

def _write_foundation_map(out_path: str, gene_map_or_list, global_mean=None, global_std=None):
    import os as _os, json as _json
    import pandas as _pd
    if isinstance(gene_map_or_list, dict):
        items = sorted(((k, int(v)) for k, v in gene_map_or_list.items()), key=lambda kv: kv[1])
        symbols = [_normalize_symbol(k) for k, _ in items]
        token_ids = [v for _, v in items]
    else:
        symbols = [_normalize_symbol(s) for s in list(gene_map_or_list)]
        token_ids = list(range(len(symbols)))

    ext = _os.path.splitext(out_path)[1].lower()
    if ext == ".parquet":
        _pd.DataFrame({"gene_symbol": symbols, "token_id": token_ids}).to_parquet(out_path, index=False)
    elif ext == ".json":
        payload = {"gene_symbol": symbols, "token_id": token_ids,
                   "global_mean": float(global_mean) if global_mean is not None else None,
                   "global_std": float(global_std) if global_std is not None else None}
        with open(out_path, "w", encoding="utf-8") as f:
            _json.dump(payload, f)
    else:
        raise ValueError("out_path must end with .parquet or .json")

def _build_gene_map_from_metadata(meta_path: str):
    """Create SYMBOL->token_id without touching the dataset."""
    import os as _os, json as _json
    try:
        import pandas as _pd
    except Exception:
        _pd = None
    ext = _os.path.splitext(meta_path)[1].lower()
    if ext == ".parquet":
        if _pd is None:
            raise ImportError("pandas is required to read parquet metadata")
        df = _pd.read_parquet(meta_path)
        if {"gene_symbol", "token_id"}.issubset(df.columns):
            return { _normalize_symbol(s): int(t) for s, t in zip(df["gene_symbol"], df["token_id"]) }
        elif "gene_symbol" in df.columns:
            symbols = [ _normalize_symbol(s) for s in df["gene_symbol"] ]
            return { s: i for i, s in enumerate(symbols) }
        else:
            raise ValueError("gene_metadata.parquet must have 'gene_symbol' (+ optional 'token_id').")
    else:
        with open(meta_path, "r", encoding="utf-8") as f:
            j = _json.load(f)
        if isinstance(j, dict) and "gene_symbol" in j and "token_id" in j:
            return { _normalize_symbol(s): int(t) for s, t in zip(j["gene_symbol"], j["token_id"]) }
        if isinstance(j, dict):
            return { _normalize_symbol(k): int(v) for k, v in j.items() }
        if isinstance(j, list):
            syms = [ _normalize_symbol(s) for s in j ]
            return { s: i for i, s in enumerate(syms) }
        raise ValueError("Unsupported metadata JSON shape.")

# ==============================================================================
# 7. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    # --- export-only argparse shim (non-invasive to existing dataclass configs) ---
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--export-foundation-map", type=str, default=None,
                        help="Write the current foundation gene map (parquet|json) and exit.")
    parser.add_argument("--export-global-stats", type=str, default=None,
                        help="Write global mean/std JSON (optional) and exit if used with --export-foundation-map.")
    parser.add_argument("--foundation-meta", type=str, default=None,
                        help="Path to training gene metadata (parquet/json) used to build the vocab order.")
    parser.add_argument("--export-only", action="store_true",
                        help="If set with --export-foundation-map, do not train; just export and exit.")
    cli, _ = parser.parse_known_args()

    # EARLY EXPORT: do NOT touch DataModule or load_dataset
    if cli.export_foundation_map and cli.export_only:
        if not cli.foundation_meta:
            raise SystemError("Provide --foundation-meta pointing to training gene metadata (parquet/json).")
        gene_map = _build_gene_map_from_metadata(cli.foundation_meta)
        _write_foundation_map(cli.export_foundation_map, gene_map)
        if cli.export_global_stats:
            import json as _json
            payload = {"mean": 0.8255, "std": 0.3135}
            with open(cli.export_global_stats, "w", encoding="utf-8") as f:
                _json.dump(payload, f)
        print(f"[export] Wrote foundation map to: {cli.export_foundation_map}")
        import sys as _sys; _sys.exit(0)
    # --- STEP 1: INITIALIZE CONFIGURATIONS AND SET GLOBAL SEED ---
    # Instantiate all configuration objects from their dataclasses. This provides a
    # single, type-safe source of truth for all hyperparameters.
    model_config = ModelConfig()
    train_config = TrainingConfig()
    data_config = DataConfig()
    exp_config = ExperimentConfig()

    # Set the global random seed for reproducibility across all libraries (PyTorch,
    # NumPy, Python's random, etc.). The `workers=True` flag ensures that dataloader
    # workers also inherit the seed, which is crucial for reproducible data loading.
    L.seed_everything(exp_config.random_seed, workers=True)
    log.info(f"Set global random seed to {exp_config.random_seed}")

    # --- STEP 2: PREREQUISITE CHECKS & HARDWARE SETUP ---
    # The script is fundamentally dependent on the `datasets` library for data loading.
    # We must exit early with an informative error if it's not installed.
    if load_dataset is None:
        log.error("FATAL: Hugging Face 'datasets' library is required to run this script. Please run 'pip install datasets'.")
        exit(1)

    # Determine the number of available GPUs to correctly calculate distributed
    # batch sizes and configure the trainer. Defaults to 1 for CPU training.
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    log.info(f"Discovered {num_devices} available device(s).")
    
    # Calculate the true, effective batch size, which is the number of samples
    # the model processes before a single optimizer step (weight update).
    effective_batch_size = data_config.batch_size * num_devices * train_config.accumulate_grad_batches
    log.info(f"Running on {num_devices} devices. Effective batch size: {effective_batch_size} "
             f"({data_config.batch_size} per device * {train_config.accumulate_grad_batches} accumulation steps).")

    # --- STEP 3: PREPARE DATA AND FINALIZE MODEL CONFIGURATION ---
    # This sequence is critical due to data-dependent model parameters.
    
    # 3a. Instantiate the DataModule. It encapsulates all data-related logic.
    datamodule = Tahoe100MDataModule(data_config, exp_config)
    
    # 3b. Manually trigger the data preparation and setup steps. This must be done
    # before model initialization.
    # `prepare_data()`: Runs ONCE on rank 0. Downloads the file manifest from HF Hub
    #                  to prevent rate-limiting errors.
    # `setup("fit")`:   Runs on ALL ranks. Loads the local manifest and, most importantly,
    #                  processes the gene metadata to create the `gene_map` and determine
    #                  the vocabulary size.
    log.info("Preparing and setting up the DataModule...")
    datamodule.prepare_data()
    datamodule.setup("fit") 

    # --- export mapping (no training required if export-only not set earlier) ---
    if cli.export_foundation_map:
        if not getattr(datamodule, "gene_map", None):
            raise RuntimeError("datamodule.gene_map not initialized; cannot export.")
        _write_foundation_map(
            cli.export_foundation_map,
            datamodule.gene_map,
            getattr(datamodule, "global_mean", None),
            getattr(datamodule, "global_std", None),
        )
        log.info(f"Wrote foundation map to: {cli.export_foundation_map}")

        if cli.export_global_stats:
            import json as _json
            stats_path = cli.export_global_stats
            payload = {
                "mean": float(getattr(datamodule, "global_mean", 0.0)),
                "std": float(getattr(datamodule, "global_std", 1.0)),
            }
            with open(stats_path, "w", encoding="utf-8") as f:
                _json.dump(payload, f)
            log.info(f"Wrote global stats to: {stats_path}")

        if cli.export_only:
            log.info("Export-only requested; exiting before training.")
            import sys as _sys; _sys.exit(0)
    
    # 3c. Finalize the model configuration. Now that the datamodule has processed the
    # metadata, we can get the exact vocabulary size. This value is essential for
    # initializing the `nn.Embedding` layer in the model.
    model_config.gene_vocab_size = datamodule.gene_vocab_size
    log.info(f"Finalized model configuration: Set gene vocabulary size to {model_config.gene_vocab_size}")

    # --- STEP 4: CALCULATE TOTAL TRAINING STEPS FOR SCHEDULER ---
    # The learning rate scheduler (cosine decay) needs the total number of optimizer
    # steps in advance to correctly map its decay curve over the training duration.
    steps_per_epoch = math.ceil(data_config.train_samples / (data_config.batch_size * num_devices))
    updates_per_epoch = math.ceil(steps_per_epoch / train_config.accumulate_grad_batches)  # FIXED: ceil, not floor
    total_steps = updates_per_epoch * train_config.max_epochs

    log.info(
        f"Estimated training schedule: {steps_per_epoch} raw batches/epoch, "
        f"{updates_per_epoch} optimizer updates/epoch, for {total_steps} total updates over "
        f"{train_config.max_epochs} epochs."
    )

    # --- STEP 5: INITIALIZE THE PYTORCH LIGHTNING MODULE ---
    # With the finalized model configuration and total training steps, we can now
    # create our main JepaLightningModule.
    trainer_module = JepaLightningModule(model_config, train_config, exp_config, total_steps)

    # --- STEP 6: CONFIGURE THE LOGGER ---
    # Set up Weights & Biases for experiment tracking, visualization, and logging.
    logger = None
    if WandbLogger and wandb:
        logger = WandbLogger(
            project=exp_config.wandb_project, 
            entity=exp_config.wandb_entity, 
            name=exp_config.wandb_run_name,
            # Log all hyperparameter configurations for complete reproducibility.
            config={
                "model": asdict(model_config), 
                "training": asdict(train_config), 
                "data": asdict(data_config), 
                "experiment": asdict(exp_config)
            },
            save_dir="logs/"
        )
        # Use wandb.watch to track model gradients and parameter distributions.
        # 'log_freq' should be tuned based on `log_every_n_steps`.
        if logger.experiment:
            logger.watch(trainer_module.model, log="all", log_freq=max(100, exp_config.log_every_n_steps * 5))
            log.info("Weights & Biases logger initialized and watching the model.")
    else:
        log.warning("`wandb` not found. Logging is disabled. Advanced validation callbacks will also be disabled.")
    

    # --- STEP 7: CONFIGURE CALLBACKS ---
    # Ensure checkpoint directory exists for saving and potential resume
    os.makedirs(exp_config.checkpoint_dir, exist_ok=True)
    log.info(f"Checkpoints directory: {exp_config.checkpoint_dir}")
    callbacks: List[Callback] = [
        ModelCheckpoint(
            dirpath=exp_config.checkpoint_dir, 
            monitor="val_loss",
            save_top_k=2, 
            mode="min", 
            save_last=True,
            filename='scjepa-{epoch:02d}-{val_loss:.3f}'
        ),
        LearningRateMonitor(logging_interval="step")
    ]
    
    # Always add the supervised probe callback
    log.info("Adding SupervisedValidatorCallback.")
    if not datamodule.gene_map:
        raise RuntimeError("DataModule's `gene_map` is not initialized. Cannot create SupervisedValidatorCallback.")
    
    callbacks.append(SupervisedValidatorCallback(
        foundation_gene_map=datamodule.gene_map,
        embedding_dim=model_config.d,
        global_mean=datamodule.global_mean,
        global_std=datamodule.global_std,
        probe_dataset_path="scvi-tools/human-lung-cell-atlas-scanvi",
        probe_cell_type_col="scanvi_label",
        run_every_n_epochs=exp_config.validation_plot_every_n_epochs,
        max_probe_cells=10_000
    ))

    # Conditionally add visualization callbacks
    if logger:
        log.info("W&B logger detected. Adding UMAP visualization callback (EmbeddingQualityValidator).")
        callbacks.append(EmbeddingQualityValidator(
            num_batches=exp_config.validation_num_batches, 
            plot_every_n_epochs=exp_config.validation_plot_every_n_epochs
        ))
    else:
        log.warning("WandbLogger not available. UMAP plotting will be disabled.")

    strategy = "auto"
    if num_devices > 1:
        log.info("Multiple devices detected. Using DDPStrategy(find_unused_parameters=True).")
        strategy = DDPStrategy(find_unused_parameters=True)

    supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    resolved_precision = "bf16-mixed" if supports_bf16 else "16-mixed"
    log.info(f"Hardware support for bfloat16: {supports_bf16}. Using precision: '{resolved_precision}'")

    use_cuda = torch.cuda.is_available()
    multi_gpu = use_cuda and (num_devices > 1)
    sync_bn = False
    log.info(f"sync_batchnorm enabled: {sync_bn}")

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        strategy=strategy,
        precision=resolved_precision,
        max_epochs=train_config.max_epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=exp_config.log_every_n_steps,
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        gradient_clip_val=train_config.gradient_clip_val,
        limit_train_batches=steps_per_epoch,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        limit_val_batches=exp_config.validation_num_batches,
        sync_batchnorm=sync_bn,
        reload_dataloaders_every_n_epochs=1,
    )
    
    # --- STEP 9: START THE TRAINING ---
    # Log the final, resolved configurations before starting the run for clarity.
    log.info("--- All components initialized. Starting training. ---")
    log.info(f"Final Model Config: {asdict(model_config)}")
    log.info(f"Final Training Config: {asdict(train_config)}")
    
    # The `fit` method starts the training and validation loops, handling all the
    # underlying complexity of distributed training, optimization, and logging.
    # We pass the datamodule to handle train/val dataloader creation.
    ckpt_path = None
    try:
        # Prefer the user-specified checkpoint filename if present
        preferred_ckpt = os.path.join(
            exp_config.checkpoint_dir, "scjepa-epoch=34-val_loss=0.116.ckpt"
        )
        if os.path.isfile(preferred_ckpt):
            ckpt_path = preferred_ckpt
            log.info(f"Resuming from specified checkpoint: {ckpt_path}")
        else:
            last_ckpt = os.path.join(exp_config.checkpoint_dir, "last.ckpt")
            if os.path.isfile(last_ckpt):
                ckpt_path = last_ckpt
                log.info(f"Resuming from checkpoint: {ckpt_path}")
            else:
                # Fallback: resume from most recent .ckpt if available
                candidates = [
                    os.path.join(exp_config.checkpoint_dir, f)
                    for f in os.listdir(exp_config.checkpoint_dir)
                    if f.endswith(".ckpt")
                ]
                if candidates:
                    ckpt_path = max(candidates, key=os.path.getmtime)
                    log.info(f"Resuming from latest checkpoint: {ckpt_path}")
    except Exception as e:
        log.warning(f"Checkpoint resume discovery failed: {e}")

    trainer.fit(trainer_module, datamodule=datamodule, ckpt_path=ckpt_path)
    
    log.info("--- Training complete. ---")