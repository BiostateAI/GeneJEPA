import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import logging
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import LightningModule as L
from huggingface_hub import hf_hub_url
from datasets import load_dataset
import wandb
from typing import TYPE_CHECKING
import lightning.pytorch as L
from lightning.pytorch.callbacks import Callback

if TYPE_CHECKING:  # avoid runtime access before import time
    from lightning.pytorch import Trainer, LightningModule
    
log = logging.getLogger(__name__)

class LinearProbeMLP(nn.Module):
    """A simple MLP for the linear probing task."""
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 512),
            nn.GELU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# ==============================================================================
# 4. VALIDATION CALLBACKS (CORRECTED FOR DDP AND STREAMING)
# ==============================================================================
class EmbeddingQualityValidator(Callback):
    """
    Monitors embedding quality via collapse metrics and UMAP visualization.
    
    --- CORRECTED & ROBUST VERSION ---
    This version is compatible with PyTorch Lightning 2.x and robust to
    streaming/iterable datasets. It works by reading a cache of embeddings
    and metadata that is populated by the JepaLightningModule during the
    main validation loop. This avoids the error of re-iterating a consumed
    streaming dataloader.
    """
    def __init__(self, num_batches: int, plot_every_n_epochs: int, color_by: str = "drug"):
        super().__init__()
        self.enabled = all([pd, sns, plt, umap, wandb])
        if not self.enabled:
            log.warning("Missing libraries for EmbeddingQualityValidator (pandas, seaborn, umap-learn, wandb). Disabling callback.")
        
        # This parameter is now mainly for documentation and to configure the LightningModule's cache size.
        self.num_batches = num_batches
        self.plot_every_n_epochs = plot_every_n_epochs
        self.color_by_key = color_by
        
        if self.enabled:
            log.info(f"EmbeddingQualityValidator initialized. It will process cached data and plot every {plot_every_n_epochs} epochs.")

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if not self.enabled:
            return
        
        # Check if the cache exists and is populated.
        if not hasattr(pl_module, '_validation_cache') or not pl_module._validation_cache:
            if trainer.global_rank == 0:
                log.warning("Validation cache not found or is empty. Skipping EmbeddingQualityValidator for this epoch.")
            return

        world_size = trainer.strategy.world_size

        # STEP 1: All processes retrieve their LOCAL cached data.
        cached_data = pl_module._validation_cache
        local_embeddings = [item['embeddings'] for item in cached_data]
        # The metadata is a list of lists of dicts; it needs to be flattened.
        local_metadata = [meta for item in cached_data for meta in item['metadata']]

        # STEP 2: Gather embeddings from all processes to the main process (rank 0).
        if world_size > 1:
            local_tensor = torch.cat(local_embeddings).to(pl_module.device) if local_embeddings else torch.empty(0, pl_module.model.config.d, device=pl_module.device)
            local_size = torch.tensor([local_tensor.shape[0]], device=pl_module.device, dtype=torch.long)
            all_sizes = trainer.strategy.all_gather(local_size).flatten()
            max_size = int(all_sizes.max().item())
            
            if local_tensor.shape[0] < max_size:
                pad_shape = (max_size - local_tensor.shape[0], local_tensor.shape[1])
                pad = torch.zeros(pad_shape, device=local_tensor.device, dtype=local_tensor.dtype)
                local_tensor_padded = torch.cat([local_tensor, pad], dim=0)
            else:
                local_tensor_padded = local_tensor
            
            gathered_padded = trainer.strategy.all_gather(local_tensor_padded)
            
            if trainer.global_rank == 0:
                unpadded_tensors = [gathered_padded[r, :all_sizes[r]] for r in range(world_size)]
                embeddings_tensor = torch.cat(unpadded_tensors, dim=0)
                rank_0_embeddings = unpadded_tensors[0] if len(unpadded_tensors) > 0 else torch.empty(0)
            else:
                embeddings_tensor = torch.empty(0)
                rank_0_embeddings = torch.empty(0)
        else:
            embeddings_tensor = torch.cat(local_embeddings) if local_embeddings else torch.empty(0)
            rank_0_embeddings = embeddings_tensor

        # STEP 3: Only rank 0 performs metrics calculation and plotting.
        if trainer.global_rank == 0 and embeddings_tensor.numel() > 0:
            log.info(f"Processing {embeddings_tensor.shape[0]} collected embeddings on rank 0...")
            self._log_collapse_metrics(embeddings_tensor, pl_module)
            if rank_0_embeddings.numel() > 0 and (trainer.current_epoch + 1) % self.plot_every_n_epochs == 0:
                self._log_umap_plot(rank_0_embeddings, local_metadata, pl_module)
                
        # STEP 4: All processes must wait here to stay in sync.
        if world_size > 1:
            trainer.strategy.barrier()

    def _log_collapse_metrics(self, embeddings: torch.Tensor, pl_module: L.LightningModule):
        embeddings_f32 = embeddings.float()
        norm_embeds = F.normalize(embeddings_f32, p=2, dim=1)
        subset_size = min(4096, len(norm_embeds))
        indices = torch.randperm(len(norm_embeds))[:subset_size]
        sim_matrix = norm_embeds[indices] @ norm_embeds[indices].T
        if subset_size > 1:
            avg_cosine_sim = (sim_matrix.sum() - subset_size) / (subset_size * (subset_size - 1))
        else:
            avg_cosine_sim = torch.tensor(0.0, device=embeddings.device)
        output_norm_std = torch.std(torch.linalg.norm(embeddings_f32, dim=1), unbiased=False)

        if wandb and wandb.run:
            try:
                wandb.log({
                    "val_quality/cosine_hist": wandb.Histogram(sim_matrix.detach().flatten().cpu().numpy()),
                    "val_quality/embedding_norms_hist": wandb.Histogram(torch.linalg.norm(embeddings_f32, dim=1).detach().cpu().numpy()),
                })
            except Exception as _e:
                pass  # stay silent if histogram creation fails on edge cases

        pl_module.log_dict(
            {"val_quality/avg_cosine_sim": avg_cosine_sim.item(), "val_quality/output_norm_std": output_norm_std.item()},
            on_step=False, on_epoch=True, rank_zero_only=True
        )

    def _log_umap_plot(self, embeddings: torch.Tensor, metadata: List[Dict], pl_module: L.LightningModule):
        log.info("Generating UMAP plot on rank 0...")
        max_points_for_plot = 5_000
        if len(embeddings) > max_points_for_plot:
            log.warning(f"UMAP data has {len(embeddings)} points (> {max_points_for_plot}). Subsampling to prevent OOM.")
            perm = torch.randperm(len(embeddings))[:max_points_for_plot]
            plot_embeddings = embeddings[perm]
            plot_metadata = [metadata[i] for i in perm.cpu().numpy()]
        else:
            plot_embeddings, plot_metadata = embeddings, metadata
            
        reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1, low_memory=True)
        try:
            embedding_2d = reducer.fit_transform(plot_embeddings.float().cpu().numpy())
        except Exception as e:
            log.error(f"UMAP transformation failed: {e}. Skipping plot.")
            return
            
        df = pd.DataFrame(embedding_2d, columns=['UMAP1', 'UMAP2'])
        df = pd.concat([df, pd.DataFrame(plot_metadata).reset_index(drop=True)], axis=1)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if self.color_by_key in df.columns and 1 < df[self.color_by_key].nunique() < 100:
            top_cats = df[self.color_by_key].value_counts().nlargest(10).index
            df_plot = df[df[self.color_by_key].isin(top_cats)]
            sns.scatterplot(data=df_plot, x='UMAP1', y='UMAP2', hue=self.color_by_key, ax=ax, s=5, alpha=0.7, linewidth=0)
            ax.legend(title=self.color_by_key, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            sns.scatterplot(data=df, x='UMAP1', y='UMAP2', ax=ax, s=5, alpha=0.5, linewidth=0)
            
        ax.set_title(f"UMAP of Cell Embeddings (Epoch {pl_module.current_epoch}, Rank 0 data)")
        plt.tight_layout()
        
        if wandb and wandb.run:
            try:
                # FIX: Pass the figure object 'fig' directly to wandb.Image
                wandb.log({"val_quality/embedding_UMAP": wandb.Image(fig)}, step=pl_module.global_step)
                log.info("UMAP plot logged successfully by rank 0.")
            except Exception as e:
                log.error(f"Failed to log UMAP plot to wandb: {e}")
        
        # Always close the figure to free up memory
        plt.close(fig)


# Helper function for robust gene symbol matching
def _norm_sym(s: Any) -> str:
    """
    Normalize a gene symbol: strip, drop Ensembl version, uppercase. Do not alter hyphens.
    """
    if s is None or not isinstance(s, str):
        return ""
    s2 = str(s).strip()
    if s2.upper().startswith("ENSG"):
        s2 = s2.split(".")[0]
    return s2.upper()

class SupervisedValidatorCallback(Callback):
    """
    A robust callback to perform linear probing on a supervised downstream task.

    --- PRODUCTION HARDENED VERSION ---
    This version includes multiple hardening improvements for real-world use:
    1.  **Flexible Data Source Resolution**: Intelligently finds raw counts in
        adata.layers['counts'], adata.raw.X, or adata.X.
    2.  **Minified AnnData Fallback**: If a minified AnnData is detected (zero
        expression), it uses the scvi-hub model to generate denoised expression
        values, making it compatible with the Human Lung Cell Atlas.
    3.  **Correct Gene Identifiers**: Passes `adata.var_names` to the scvi-hub
        model API, which is the correct key space, not gene symbols.
    4.  **Flexible Gene Symbol Resolution**: Finds gene names in 'feature_name',
        'gene_symbol', or the .var_names index for vocabulary mapping.
    5.  **Memory Capping**: Limits the number of cells used to prevent OOM.
    """
    def __init__(
        self,
        foundation_gene_map: Dict[int, int],
        embedding_dim: int,
        global_mean: Optional[float],
        global_std: Optional[float],
        probe_dataset_path: str,
        probe_cell_type_col: str,
        foundation_repo_id: str = "vevotx/Tahoe-100M",
        foundation_metadata_filename: str = "metadata/gene_metadata.parquet",
        foundation_token_id_col: str = "token_id",
        foundation_gene_symbol_col: str = "gene_symbol",
        probe_train_epochs: int = 5,
        probe_batch_size: int = 32,
        probe_lr: float = 1e-3,
        run_every_n_epochs: int = 1,
        max_probe_cells: Optional[int] = 50_000,
    ):
        super().__init__()
        self.foundation_gene_map = foundation_gene_map
        self.embedding_dim = embedding_dim
        self.global_mean = global_mean
        self.global_std = global_std
        self.probe_dataset_path = probe_dataset_path
        self.probe_cell_type_col = probe_cell_type_col
        self.foundation_repo_id = foundation_repo_id
        self.foundation_metadata_filename = foundation_metadata_filename
        self.foundation_token_id_col = foundation_token_id_col
        self.foundation_gene_symbol_col = foundation_gene_symbol_col
        self.probe_train_epochs = probe_train_epochs
        self.probe_batch_size = probe_batch_size
        self.probe_lr = probe_lr
        self.run_every_n_epochs = run_every_n_epochs
        self.max_probe_cells = max_probe_cells
        self.is_initialized = False
        self.probe_model: Optional[LinearProbeMLP] = None
        self.probe_train_loader: Optional[DataLoader] = None
        self.probe_val_loader: Optional[DataLoader] = None
        self.num_classes = 0
        if self.global_mean is None or self.global_std is None:
            log.warning("[Probe Validator] Global mean/std not provided. Probe data will only be log1p transformed.")
        log.info("SupervisedValidatorCallback initialized.")

    def _initialize_on_rank_zero(self, device: torch.device):
        log.info("[Probe Validator] Initializing on rank 0 (HARDENED & DIAGNOSTIC MODE)...")
        self.is_initialized = True
        
        from huggingface_hub import hf_hub_download
        import anndata as ad
        from datasets import Dataset
        import scipy.sparse as sp
        import pandas as pd

        # PART A: LOAD FOUNDATION VOCABULARY (Unchanged)
        try:
            log.info(f"[Probe Validator] Loading foundation gene metadata from '{self.foundation_repo_id}'...")
            metadata_url = hf_hub_url(self.foundation_repo_id, self.foundation_metadata_filename, repo_type="dataset")
            foundation_metadata_ds = load_dataset("parquet", data_files=metadata_url, split="train")
            foundation_symbol_to_token_map = {
                _norm_sym(e[self.foundation_gene_symbol_col]): e[self.foundation_token_id_col]
                for e in foundation_metadata_ds if _norm_sym(e.get(self.foundation_gene_symbol_col))
            }
            log.info(f"[Probe Validator] Built foundation vocabulary map with {len(foundation_symbol_to_token_map)} unique normalized symbols.")
            log.info(f"[Probe Validator] Foundation vocab sample: {list(foundation_symbol_to_token_map.keys())[:5]}")
        except Exception as e:
            log.error(f"[Probe Validator] FATAL: FAILED to load foundation gene metadata: {e}. Disabling callback.")
            return

        # PART B: LOAD PROBE ANNDATA (Unchanged)
        try:
            log.info(f"[Probe Validator] Loading probe dataset from HF Hub: '{self.probe_dataset_path}'...")
            local_h5ad = hf_hub_download(repo_id=self.probe_dataset_path, filename="adata.h5ad")
            adata = ad.read_h5ad(local_h5ad, backed=None)
            log.info(f"[Probe Validator] Loaded probe dataset with {adata.n_obs} cells and {adata.n_vars} genes.")
        except Exception as e:
            log.error(f"[Probe Validator] FATAL: FAILED to load probe anndata object: {e}. Disabling callback.")
            return

        # PART C: BUILD VOCABULARY OVERLAP MAP (Unchanged, but uses resolved symbols)
        if 'feature_name' in adata.var.columns:
            probe_gene_symbols_for_map = adata.var['feature_name']
        else:
            probe_gene_symbols_for_map = adata.var_names.to_series()
        
        probe_idx_to_foundation_idx_map = {}
        for probe_idx, raw_symbol in enumerate(probe_gene_symbols_for_map):
            norm_symbol = _norm_sym(raw_symbol)
            if not norm_symbol: continue
            foundation_token_id = foundation_symbol_to_token_map.get(norm_symbol)
            if foundation_token_id is None: continue
            foundation_contiguous_idx = self.foundation_gene_map.get(foundation_token_id)
            if foundation_contiguous_idx is not None:
                probe_idx_to_foundation_idx_map[probe_idx] = foundation_contiguous_idx
        
        log.info(f"--- [OVERLAP REPORT] ---")
        log.info(f"Successfully mapped {len(probe_idx_to_foundation_idx_map)} genes between probe and foundation vocabularies.")

        # PART D: SUBSAMPLE, RESOLVE COUNTS, AND CREATE DATALOADERS
        if self.probe_cell_type_col not in adata.obs.columns:
            log.error(f"[Probe Validator] Label column '{self.probe_cell_type_col}' not found. Disabling.")
            return
        probe_labels = adata.obs[self.probe_cell_type_col].astype(str).to_numpy()
        unique_labels = sorted(np.unique(probe_labels).tolist())
        label_map = {name: i for i, name in enumerate(unique_labels)}
        self.num_classes = len(label_map)
        log.info(f"Found {self.num_classes} classes for probing.")

        rng = np.random.default_rng(42)
        n_obs = adata.n_obs
        take = min(self.max_probe_cells or n_obs, n_obs)
        sel_indices = np.sort(rng.choice(n_obs, size=take, replace=False))
        log.info(f"Subsampling probe data to {take} cells (from {n_obs}).")

        counts_like = None
        if hasattr(adata, "layers") and "counts" in adata.layers:
            log.info("[Probe Validator] Using `adata.layers['counts']` as counts.")
            counts_like = adata.layers["counts"]
        elif getattr(adata, "raw", None) is not None and getattr(adata.raw, "X", None) is not None:
            log.info("[Probe Validator] Using `adata.raw.X` as counts.")
            counts_like = adata.raw.X
        else:
            log.info("[Probe Validator] No 'counts' layer or .raw found. Falling back to `adata.X`.")
            counts_like = adata.X

        counts_csr = sp.csr_matrix(counts_like[sel_indices, :]) if counts_like is not None else sp.csr_matrix((len(sel_indices), adata.n_vars))
        nnz_total = int(counts_csr.nnz)

        if nnz_total == 0:
            log.warning("[Probe Validator] Detected minified or empty AnnData. Attempting to generate denoised expression via scvi-hub.")
            try:
                import scvi
                import scvi.hub
            except ImportError as e:
                log.error(f"[Probe Validator] scvi-tools is required for minified data fallback. `pip install scvi-tools`. Error: {e}")
                return

            try:
                overlap_probe_cols = sorted(probe_idx_to_foundation_idx_map.keys())
                hubmodel = scvi.hub.HubModel.pull_from_huggingface_hub(self.probe_dataset_path)
                model = hubmodel.model
                
                adata_manager = model.adata_manager
                batch_key = adata_manager.get_state_registry(scvi.REGISTRY_KEYS.BATCH_KEY).original_key
                target_batch = hubmodel.adata.obs[batch_key].mode()[0] if batch_key in hubmodel.adata.obs else None
                if target_batch:
                    log.info(f"[Probe Validator] Using most frequent batch '{target_batch}' for transformation.")

                # scVI expects var_names from the model's AnnData, not display symbols.
                gene_list = list(adata.var_names[overlap_probe_cols])
                log.info(f"[Probe Validator] Querying hub model for {len(gene_list)} genes on {len(sel_indices)} cells using var_names.")

                den_expr = model.get_normalized_expression(
                    adata=adata,
                    indices=sel_indices,
                    gene_list=gene_list,
                    transform_batch=target_batch,
                )

                TOPK = 256
                data, rows, cols = [], [], []
                for i, row in enumerate(den_expr.itertuples(index=False, name=None)):
                    arr = np.asarray(row, dtype=np.float32)
                    if arr.size == 0: continue
                    topk_idx = np.argpartition(arr, -TOPK)[-TOPK:] if arr.size > TOPK else np.arange(arr.size)
                    vals = arr[topk_idx]
                    keep = vals > 1e-6
                    for j_local, v in zip(topk_idx[keep], vals[keep]):
                        rows.append(i)
                        cols.append(overlap_probe_cols[j_local])
                        data.append(float(v))
                counts_csr = sp.csr_matrix((data, (rows, cols)), shape=(len(sel_indices), adata.n_vars))
                log.info(f"[Probe Validator] Hub-generated sparse matrix NNZ: {counts_csr.nnz}")
            except Exception as e:
                log.error(f"[Probe Validator] FATAL: Failed to generate expression from hub model: {e}. Disabling callback.", exc_info=True)
                return

        log.info("[Probe Validator] Converting probe data to foundation index space...")
        records = []
        y_sub = [label_map[probe_labels[i]] for i in sel_indices]
        for i in range(counts_csr.shape[0]):
            row = counts_csr.getrow(i)
            mapped_indices, mapped_values = [], []
            for probe_gene_idx, value in zip(row.indices, row.data):
                foundation_idx = probe_idx_to_foundation_idx_map.get(probe_gene_idx)
                if foundation_idx is not None and value > 0.0:
                    mapped_indices.append(int(foundation_idx))
                    mapped_values.append(float(value))
            if mapped_indices:
                records.append({"gene_indices": mapped_indices, "counts": mapped_values, "label": int(y_sub[i])})
        
        if not records:
            log.error(f"[Probe Validator] FATAL: No cells remained after mapping genes. Disabling callback.")
            return
        log.info(f"Successfully converted {len(records)} cells to the foundation vocabulary space.")

        from datasets import Features, Value, Sequence, ClassLabel

        # Define the schema, explicitly marking 'label' as a ClassLabel
        features = Features({
            'gene_indices': Sequence(Value('int64')),
            'counts': Sequence(Value('float32')),
            'label': ClassLabel(names=unique_labels)
        })
        probe_ds = Dataset.from_list(records, features=features)

        split_dataset = probe_ds.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
        train_ds, val_ds = split_dataset["train"], split_dataset["test"]
        class ProbeDataset(torch.utils.data.Dataset):
            def __init__(self, hf_ds): self.hf_ds = hf_ds; self.hf_ds.set_format(type='torch', columns=['gene_indices', 'counts', 'label'])
            def __len__(self): return len(self.hf_ds)
            def __getitem__(self, idx): return self.hf_ds[idx]

        self.probe_train_loader = DataLoader(ProbeDataset(train_ds), batch_size=self.probe_batch_size, shuffle=True, collate_fn=self._collate_fn, num_workers=0)
        self.probe_val_loader = DataLoader(ProbeDataset(val_ds), batch_size=self.probe_batch_size, collate_fn=self._collate_fn, num_workers=0)
        self.probe_model = LinearProbeMLP(self.embedding_dim, self.num_classes).to(device)
        log.info(f"[Probe Validator] Initialization complete. Ready for probing.")

    def _collate_fn(self, batch: List[Dict]) -> Dict:
        indices = torch.cat([s["gene_indices"] for s in batch])
        values = torch.cat([s["counts"] for s in batch])
        labels = torch.stack([s["label"] for s in batch])
        values_norm = torch.log1p(values.float())
        if self.global_mean is not None and self.global_std is not None:
            values_norm = (values_norm - self.global_mean) / (self.global_std + 1e-6)
        offsets = torch.tensor([0] + [len(s["gene_indices"]) for s in batch], dtype=torch.long).cumsum(0)
        return {"indices": indices, "values": values_norm, "offsets": offsets, "labels": labels}

    def _get_embeddings_and_labels(self, dataloader: DataLoader, pl_module: L.LightningModule) -> Tuple[torch.Tensor, torch.Tensor]:
        all_embeddings, all_labels = [], []
        pl_module.eval()
        with torch.no_grad():
            for batch in dataloader:
                embeddings = pl_module.model.get_embedding(
                    batch["indices"].to(pl_module.device),
                    batch["values"].to(pl_module.device),
                    batch["offsets"].to(pl_module.device),
                    use_teacher=True,
                )
                all_embeddings.append(embeddings.cpu())
                all_labels.append(batch["labels"].cpu())
        pl_module.train()
        return torch.cat(all_embeddings), torch.cat(all_labels)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        # 1. The most important fix: Skip the entire callback during the sanity check.
        if trainer.sanity_checking:
            log.info("[Probe Validator] In sanity check, skipping probe evaluation.")
            return

        # 2. Do all work only on the global rank 0 process. No manual barriers needed.
        if not trainer.is_global_zero:
            return

        # 3. Respect the user-defined frequency
        if (trainer.current_epoch + 1) % self.run_every_n_epochs != 0:
            return

        # --- The rest of the original logic can now run safely on rank 0 ---
        log.info(f"--- [SupervisedValidator] Starting probe evaluation on epoch {trainer.current_epoch} ---")
        if not self.is_initialized:
            self._initialize_on_rank_zero(pl_module.device)
        
        if not self.probe_model or not self.probe_train_loader or not self.probe_val_loader:
            log.warning("[Probe Validator] Skipping evaluation as initialization failed or was incomplete.")
            return

        # Extract frozen embeddings
        log.info("[Probe Validator] Extracting frozen embeddings for probe training...")
        train_embeddings, train_labels = self._get_embeddings_and_labels(self.probe_train_loader, pl_module)

        # Train the probe
        log.info(f"[Probe Validator] Training probe for {self.probe_train_epochs} epochs on {len(train_embeddings)} samples...")
        self.probe_model.train()
        optimizer = torch.optim.Adam(self.probe_model.parameters(), lr=self.probe_lr)
        loss_fn = nn.CrossEntropyLoss()
        with torch.enable_grad():
            for epoch in range(self.probe_train_epochs):
                perm = torch.randperm(len(train_embeddings))
                train_embeddings_shuffled = train_embeddings[perm]
                train_labels_shuffled = train_labels[perm]
                
                for i in range(0, len(train_embeddings), self.probe_batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    batch_embeds = train_embeddings_shuffled[i:i+self.probe_batch_size].to(pl_module.device)
                    batch_labels = train_labels_shuffled[i:i+self.probe_batch_size].to(pl_module.device)
                    
                    preds = self.probe_model(batch_embeds)
                    loss = loss_fn(preds, batch_labels)
                    loss.backward()
                    optimizer.step()

        # Evaluate the probe
        log.info("[Probe Validator] Evaluating probe on validation set...")
        self.probe_model.eval()
        val_embeddings, val_labels = self._get_embeddings_and_labels(self.probe_val_loader, pl_module)
        with torch.no_grad():
            val_preds = self.probe_model(val_embeddings.to(pl_module.device))
            correct = (val_preds.argmax(dim=1) == val_labels.to(pl_module.device)).sum().item()
            accuracy = correct / len(val_labels) if len(val_labels) > 0 else 0.0
        
        log.info(f"[Probe Validator] Probe validation accuracy: {accuracy:.4f}")
        pl_module.log("val_probe/accuracy", accuracy, on_step=False, on_epoch=True, rank_zero_only=True)
