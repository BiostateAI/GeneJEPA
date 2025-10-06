import os
import json
import math
import time
import itertools
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
import lightning as L

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from .configs import DataConfig, ExperimentConfig


log = logging.getLogger(__name__)


class Tahoe100MDataset(IterableDataset):
    def __init__(self, hf_dataset, gene_map: Dict[int, int]):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.gene_map = gene_map

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        dataset_instance = self.hf_dataset

        # --- SAFE WORKER SHARDING FOR HF STREAMING ---
        if worker_info is not None and worker_info.num_workers > 1:
            # How many shards are actually available after any prior DDP sharding
            n_shards = getattr(dataset_instance, "n_shards", 1)

            if n_shards <= 1:
                # Only worker 0 should iterate; others yield nothing
                if worker_info.id != 0:
                    return
            else:
                # Clamp shards to what's available and map worker id into that range
                num_shards_to_use = min(worker_info.num_workers, n_shards)
                idx = worker_info.id % num_shards_to_use
                dataset_instance = dataset_instance.shard(
                    num_shards=num_shards_to_use, index=idx
                )

        for cell in dataset_instance:
            if 'genes' not in cell or 'expressions' not in cell:
                continue
            genes, expressions = cell['genes'], cell['expressions']
            if not genes or not expressions:
                continue
            if isinstance(expressions, list) and len(expressions) > 0 and expressions[0] < 0:
                genes, expressions = genes[1:], expressions[1:]
            if not genes:
                continue

            mapped_indices = [self.gene_map[g] for g in genes if g in self.gene_map]
            valid_expressions = [e for g, e in zip(genes, expressions) if g in self.gene_map]
            if not mapped_indices:
                continue

            metadata = {"drug": cell.get("drug", "N/A"), "cell_line_name": cell.get("cell_line_name", "N/A")}
            yield {
                "gene_indices": np.array(mapped_indices, dtype=np.int64),
                "counts": np.array(valid_expressions, dtype=np.float32),
                "metadata": metadata
            }


class Tahoe100MDataModule(L.LightningDataModule):
    """
    Final, robust DataModule for Tahoe-100M with two key improvements:
    1.  Download-First Strategy: Downloads all files in `prepare_data` to prevent
        rate-limiting errors during DDP training.
    2.  Global Statistics Normalization: Computes the global mean and standard
        deviation of log1p-transformed expression values once, saves them, and
        applies these fixed stats to all batches. This provides a stable
        normalization and removes batch-to-batch statistical noise.
    """
    def __init__(self, data_config: DataConfig, exp_config: ExperimentConfig):
        super().__init__()
        self.save_hyperparameters()
        self.data_config = data_config
        self.exp_config = exp_config

        # Define local cache paths
        self.data_cache_dir = os.path.join(os.getcwd(), "hf_data_cache")
        self.manifest_path = os.path.join(self.data_cache_dir, "local_file_manifest.json")
        
        self.stats_path = os.path.join(self.data_cache_dir, "global_stats.json")
        self.repo_id = "vevotx/Tahoe-100M"
        
        # These will be populated with LOCAL file paths and stats
        self.gene_map: Optional[Dict[int, int]] = None
        self.train_files: Optional[List[str]] = None
        self.val_files: Optional[List[str]] = None
        self.metadata_file: Optional[str] = None
        
        self.global_mean: Optional[float] = None
        self.global_std: Optional[float] = None
        
        os.makedirs(self.data_cache_dir, exist_ok=True)

    @property
    def gene_vocab_size(self) -> int:
        if self.gene_map is None:
             raise RuntimeError("gene_map is not initialized. Call setup() before accessing gene_vocab_size.")
        return len(self.gene_map)

    def prepare_data(self):
        """
        Runs ONCE on rank 0. Downloads all necessary files from the Hub to a
        local cache and creates a manifest of the local paths.
        """
        log.info("--- [prepare_data] Starting (runs on RANK 0 only) ---")
        if os.path.exists(self.manifest_path):
            log.info(f"Local file manifest found at {self.manifest_path}. Skipping downloads.")
            return

        log.info("Local manifest not found. Starting download process...")
        api = HfApi()
        
        try:
            repo_files = list(api.list_repo_tree(self.repo_id, repo_type="dataset", recursive=True))
        except HfHubHTTPError as e:
            log.error(f"Failed to list files in repo {self.repo_id}. Check connection and token. Error: {e}")
            raise e

        data_file_paths = sorted([f.path for f in repo_files if f.path.startswith("data/") and f.path.endswith(".parquet")])
        metadata_file_path = next((f.path for f in repo_files if f.path.endswith("gene_metadata.parquet")), None)

        if not data_file_paths or not metadata_file_path:
            raise FileNotFoundError(f"Required .parquet files not found in {self.repo_id}.")
        
        files_to_download = data_file_paths + [metadata_file_path]
        local_file_manifest = {"data_files": [], "metadata_file": ""}

        log.info(f"Downloading {len(files_to_download)} files to {self.data_cache_dir}...")
        for i, filepath in enumerate(files_to_download):
            log.info(f"  ({i+1}/{len(files_to_download)}) Downloading {filepath}...")
            try:
                local_path = hf_hub_download(
                    repo_id=self.repo_id, filename=filepath, repo_type="dataset",
                    cache_dir=self.data_cache_dir,
                    local_dir=os.path.join(self.data_cache_dir, os.path.dirname(filepath)),
                    local_dir_use_symlinks=False
                )
                if filepath == metadata_file_path:
                    local_file_manifest["metadata_file"] = local_path
                else:
                    local_file_manifest["data_files"].append(local_path)
            except HfHubHTTPError as e:
                log.error(f"Failed to download {filepath}. Error: {e}")
                raise e
        
        local_file_manifest["data_files"].sort()
        with open(self.manifest_path, 'w') as f:
            json.dump(local_file_manifest, f)
        log.info(f"Successfully downloaded all files and saved local manifest to {self.manifest_path}")

    def setup(self, stage: Optional[str] = None):
        """
        Called on every DDP process. This method now includes the logic to
        calculate (on rank 0) or load (on all ranks) the global normalization statistics.
        """
        log.info(f"--- [setup] Starting for stage '{stage}' on PID: {os.getpid()} ---")

        # STEP 1: Load manifest and determine train/val file splits
        with open(self.manifest_path, 'r') as f:
            manifest = json.load(f)
        all_local_data_files = manifest["data_files"]
        self.metadata_file = manifest["metadata_file"]
        
        if not all_local_data_files:
            raise ValueError("The local file manifest is empty. prepare_data may have failed.")
        
        # Use hardcoded numbers for Tahoe-100M for stable calculation
        total_files = len(all_local_data_files)
        samples_per_file = 100_000_000 / 3388 
        num_files_for_samples = math.ceil(self.data_config.val_samples / samples_per_file)
        min_files_for_parallelism = self.data_config.num_workers * 2
        num_val_files = max(int(num_files_for_samples), int(min_files_for_parallelism))
        num_val_files = min(num_val_files, total_files // 2)
        if self.data_config.num_workers > 0 and num_val_files == 0 and total_files > 1:
            num_val_files = 1
        
        log.info(f"Data Splitting: {num_val_files} files for validation, {total_files - num_val_files} for training.")
        self.val_files = all_local_data_files[:num_val_files]
        self.train_files = all_local_data_files[num_val_files:]

        # STEP 2: Load or compute global statistics
        self._setup_global_stats()

        # STEP 3: Load gene metadata
        self._load_metadata(self.metadata_file)
        log.info(f"--- [setup] Finished on PID: {os.getpid()} ---")

    def _setup_global_stats(self):
        """Orchestrates the loading or computation of global statistics, ensuring DDP safety."""
        if self.global_mean is not None and self.global_std is not None:
            return

        is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        is_rank_zero = not is_ddp or torch.distributed.get_rank() == 0

        if os.path.exists(self.stats_path):
            if is_rank_zero:
                log.info(f"Loading pre-computed global stats from {self.stats_path}")
            self._load_stats()
            return

        if not is_rank_zero:
            log.info(f"Rank {torch.distributed.get_rank()} waiting for rank 0 to compute stats...")
            torch.distributed.barrier()
        else:
            log.info(f"Global stats file not found. Computing on rank 0 from a subset of training data...")
            self._compute_and_save_stats()
            if is_ddp:
                torch.distributed.barrier()

        if is_rank_zero:
            log.info("All ranks will now load the newly computed stats.")
        self._load_stats()

    def _load_stats(self):
        with open(self.stats_path, 'r') as f:
            stats = json.load(f)
        self.global_mean = float(stats['mean'])
        self.global_std = float(stats['std'])
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            log.info(f"Successfully loaded stats: mean={self.global_mean:.4f}, std={self.global_std:.4f}")
            
    def _compute_and_save_stats(self):
        num_samples_for_stats = 1_000_000
        samples_per_file = 100_000_000 / 3388
        num_files_to_load = math.ceil(num_samples_for_stats / samples_per_file)
        files_for_stats = self.train_files[:num_files_to_load]

        if not files_for_stats:
            raise RuntimeError("No training files available to compute statistics.")

        log.info(f"Using first {len(files_for_stats)} training files for streaming stats calculation...")
        temp_ds = load_dataset("parquet", data_files=files_for_stats, split="train")

        n = 0
        mean = 0.0
        M2 = 0.0

        log.info("Starting streaming calculation of mean/std...")
        data_iterator = itertools.islice(temp_ds, num_samples_for_stats)
        
        for i, cell in enumerate(data_iterator):
            if (i + 1) % 100_000 == 0:
                log.info(f"  ...processed {i+1}/{num_samples_for_stats} cells")

            if 'expressions' not in cell or not cell['expressions']:
                continue
            
            expressions = cell['expressions']
            if expressions[0] < 0:
                expressions = expressions[1:]
            if not expressions:
                continue

            log1p_values = np.log1p(np.asarray(expressions, dtype=np.float64))
            
            new_count = log1p_values.size
            if new_count == 0:
                continue
            
            n_old = float(n)
            n_new = n_old + new_count
            
            delta = np.mean(log1p_values) - mean
            
            mean += delta * (new_count / n_new)
            
            M2 += np.sum((log1p_values - np.mean(log1p_values)) ** 2) + (delta**2) * (n_old * new_count / n_new)

            n += new_count
            
        if n < 2:
            raise RuntimeError("Not enough data points (< 2) to compute standard deviation.")

        variance = M2 / (n - 1)
        std = np.sqrt(variance)

        log.info(f"Computed global stats from {n} total expression values: mean={mean:.4f}, std={std:.4f}")

        with open(self.stats_path, 'w') as f:
            json.dump({'mean': float(mean), 'std': float(std)}, f)
        log.info(f"Saved global stats to {self.stats_path}")

    def _load_metadata(self, metadata_file_path: Optional[str]):
        if self.gene_map:
            return
        if not metadata_file_path:
            raise ValueError("Metadata file path not set.")
        log.info(f"Loading gene metadata from LOCAL path: {metadata_file_path}")
        gene_metadata_ds = load_dataset("parquet", data_files=metadata_file_path, split="train")
        sorted_genes = sorted(list(gene_metadata_ds), key=lambda x: x["token_id"])
        self.gene_map = {entry["token_id"]: i for i, entry in enumerate(sorted_genes)}

    def _collate_fn(self, batch: List[Dict]) -> Dict:
        if not batch:
            return { "indices": torch.empty(0, dtype=torch.long), "values": torch.empty(0, dtype=torch.float),
                     "offsets": torch.tensor([0], dtype=torch.long), "metadata": [] }

        indices_list = [torch.from_numpy(s["gene_indices"]) for s in batch]
        values_list = [torch.from_numpy(s["counts"]) for s in batch]

        indices = torch.cat(indices_list)
        values = torch.cat(values_list)

        values = torch.log1p(values.float())

        if self.global_mean is None or self.global_std is None:
            raise RuntimeError(
                "FATAL: Global normalization statistics are not available at collate time. "
                "This will lead to training instability. Check the DataModule's setup process."
            )

        if self.global_mean is not None and self.global_std is not None:
            values = (values - self.global_mean) / (self.global_std + 1e-6)
            if torch.rand(()) < 0.01:
                vstd = values.std().item()
                if not torch.isfinite(values).all() or vstd < 1e-6:
                    print(f"[NORM] WARNING: values std={vstd:.3e} finite={bool(torch.isfinite(values).all())}")
        else:
            if values.numel() > 1:
                mean = values.mean()
                std = values.std()
                values = (values - mean) / (std + 1e-6)

        offsets = torch.tensor([0] + [len(s["gene_indices"]) for s in batch], dtype=torch.long).cumsum(0)
        metadata = [s.get("metadata", {}) for s in batch]
        return {"indices": indices, "values": values, "offsets": offsets, "metadata": metadata}

    def _create_dataloader(self, file_list: List[str], *, is_train: bool) -> DataLoader:
        if not file_list:
            log.warning("Received an empty file list for dataloader creation. Returning an empty loader.")
            return DataLoader([], batch_size=self.data_config.batch_size)
            
        hf_dataset = load_dataset("parquet", data_files=file_list, streaming=True, split="train")
        
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            hf_dataset = hf_dataset.shard(num_shards=world_size, index=rank)

        if is_train:
            rank = torch.distributed.get_rank() if (torch.distributed.is_available() and torch.distributed.is_initialized()) else 0
            shuffle_seed = int(time.time()) + rank
            shuffle_buffer_size = 50_000
            hf_dataset = hf_dataset.shuffle(
                seed=shuffle_seed,
                buffer_size=shuffle_buffer_size
            )

        dataset = Tahoe100MDataset(hf_dataset, self.gene_map)

        n_shards_after_ddp = getattr(hf_dataset, "n_shards", 1)
        
        if is_train:
            num_workers = 0  # min(self.data_config.num_workers, n_shards_after_ddp)
            if num_workers < self.data_config.num_workers:
                log.warning(f"Capping train dataloader workers from {self.data_config.num_workers} to {num_workers} to match available data shards.")
        else:
            log.info("Setting validation dataloader workers to 0 for stability with streaming.")
            num_workers = 0

        return DataLoader(
            dataset,
            batch_size=self.data_config.batch_size,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.train_files, is_train=True)

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.val_files, is_train=False)


