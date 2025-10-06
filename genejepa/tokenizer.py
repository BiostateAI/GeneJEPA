import math
from typing import Optional

import torch
import torch.nn as nn

from .configs import ModelConfig


class scRNATokenizer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        d_identity = int(config.d * config.identity_value_split_ratio)
        d_value = config.d - d_identity

        actual_fourier_dim = config.fourier_num_frequencies * 2
        # Use log-space for numerical stability
        freqs = torch.logspace(
            start=math.log10(config.fourier_min_freq), 
            end=math.log10(config.fourier_max_freq), 
            steps=config.fourier_num_frequencies, 
            dtype=torch.float32
        )
        fourier_freqs = config.fourier_freq_scale * freqs
        self.register_buffer("fourier_freqs", fourier_freqs.view(1, -1), persistent=False)

        # --- MLP to process the Fourier features into the target dimension ---
        self.value_encoder = nn.Sequential(
            nn.Linear(actual_fourier_dim, d_value * 2),
            nn.GELU(),
            nn.Linear(d_value * 2, d_value)
        )
        
        self.identity_embed = nn.Embedding(config.gene_vocab_size, d_identity)
        
        combined_dim = d_identity + d_value
        self.final_proj = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, config.d),
            nn.GELU(),
            nn.LayerNorm(config.d)
        )

    def forward(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        identity_embedding = self.identity_embed(indices)
        
        # --- Create Fourier features ---
        # values shape: [TotalTokens], fourier_freqs shape: [1, NumFreqs]
        # Broadcasting creates fourier_args with shape [TotalTokens, NumFreqs]
        fourier_args = values.unsqueeze(-1) * self.fourier_freqs
        fourier_embedding_raw = torch.cat([torch.sin(fourier_args), torch.cos(fourier_args)], dim=-1)
        
        # --- Process features with the MLP ---
        value_embedding = self.value_encoder(fourier_embedding_raw)
        
        combined_embedding = torch.cat([identity_embedding, value_embedding], dim=-1)
        final_tokens = self.final_proj(combined_embedding)

        # >>> DEBUG 8: occasionally compare channel magnitudes
        if self.training and torch.rand(()) < 0.01:
            with torch.no_grad():
                id_std  = identity_embedding.float().std().item()
                val_std = value_embedding.float().std().item()
                print(f"[TOK] std(identity)={id_std:.4f} std(value)={val_std:.4f}")

        return final_tokens


