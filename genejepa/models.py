import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from ema_pytorch import EMA

from .tokenizer import scRNATokenizer
from .configs import ModelConfig


class LatentTransformerBlock(nn.Module):
    """A standard Transformer block for self-attention over the latent array."""
    def __init__(self, config: ModelConfig, heads: Optional[int] = None):
        super().__init__()
        num_heads = heads if heads is not None else config.heads_h
        self.norm1 = nn.LayerNorm(config.d)
        self.attn = nn.MultiheadAttention(embed_dim=config.d, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.norm2 = nn.LayerNorm(config.d)
        self.ffn = nn.Sequential(nn.Linear(config.d, config.d * 4), nn.GELU(), nn.Linear(config.d * 4, config.d), nn.Dropout(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(*[self.norm1(x)] * 3)[0]
        x = x + self.ffn(self.norm2(x))
        return x

class GenePerceiverEncoder(nn.Module):
    """
    A Perceiver-style encoder that maps a variable-length set of gene tokens
    to a fixed-size representation.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d % config.heads_h == 0, f"Model dimension d ({config.d}) must be divisible by the number of heads h ({config.heads_h})."
        self.config = config
        self.tokenizer = scRNATokenizer(config)
        self.latents = nn.Parameter(torch.randn(config.latents_L, config.d))
        self.q_proj = nn.Linear(config.d, config.d, bias=False)
        self.kv_proj = nn.Linear(config.d, config.d * 2, bias=False)
        self.cross_attn_norm_q = nn.LayerNorm(config.d)
        self.cross_attn_norm_kv = nn.LayerNorm(config.d)
        self.cross_attn_out_proj = nn.Linear(config.d, config.d)
        self.latent_blocks_seq = nn.Sequential(*[LatentTransformerBlock(config) for _ in range(config.blocks_D)])
        # Using a LayerNorm here is more standard and stable than Identity.
        self.final_norm = nn.LayerNorm(config.d, elementwise_affine=True)

    def _chunked_cross_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes memory-efficient cross-attention using chunking and the online softmax trick,
        with enhanced numerical stability for mixed-precision training.

        This method avoids materializing the full (L, N) attention matrix. It performs all
        internal softmax calculations and accumulations in float32 precision to prevent
        underflow/overflow issues common in bfloat16/float16, before casting the final
        result back to the original input dtype. It is also robust to the edge case where
        a query attends only to padded tokens, preventing NaN propagation.

        Args:
            q (torch.Tensor): Queries, shape [B, h, L, d_head]. (L = num_latents)
            k (torch.Tensor): Keys, shape [B, h, N, d_head]. (N = num_input_tokens)
            v (torch.Tensor): Values, shape [B, h, N, d_head].
            key_padding_mask (torch.Tensor): Mask for padded keys, shape [B, N]. True where padded.

        Returns:
            torch.Tensor: The output of the attention mechanism, shape [B, h, L, d_head].
        """
        # --- STEP 0: SETUP AND PRECISION HANDLING ---
        B, h, L, d_head = q.shape

        # Store the original dtype to cast back to at the end.
        orig_dtype = q.dtype
        
        # Upcast all inputs to float32 for stable computation.
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)

        # Scale queries for dot product attention (now in float32).
        q = q / math.sqrt(d_head)

        # --- STEP 1 (PASS 1): Find the maximum score per query for stable softmax calculation ---
        # This is the 'm' in the online softmax algorithm. We iterate through key/value
        # chunks to find the max score without storing the whole attention matrix.
        # Initialize the max_scores tensor in float32.
        max_scores = torch.full(
            (B, h, L, 1), -torch.inf, device=q.device, dtype=torch.float32
        )
        
        # Split keys and the padding mask into chunks along the sequence dimension.
        k_chunks = k.split(self.config.cross_attn_chunk_size, dim=2)
        mask_chunks = key_padding_mask.to(torch.bool).split(
            self.config.cross_attn_chunk_size, dim=1
        )

        for k_chunk, mask_chunk in zip(k_chunks, mask_chunks):
            # Calculate scores for the current chunk: [B, h, L, d_head] @ [B, h, d_head, N_chunk] -> [B, h, L, N_chunk]
            scores_chunk = q @ k_chunk.transpose(-1, -2)
            
            # Apply the padding mask. Padded positions become -inf.
            # Mask shape [B, N_chunk] -> [B, 1, 1, N_chunk] for broadcasting.
            scores_chunk.masked_fill_(mask_chunk.view(B, 1, 1, -1), -torch.inf)
            
            # Find the maximum score within this chunk for each query.
            chunk_max, _ = scores_chunk.max(dim=-1, keepdim=True)
            
            # Update the overall maximum score seen so far.
            max_scores = torch.max(max_scores, chunk_max)

        # Detach to prevent gradients from flowing through this stability-related calculation.
        max_scores = max_scores.detach()

        # --- STEP 2: THE CRITICAL NAN GUARD ---
        # If a query attended only to padding, its max_score will be -inf.
        # This would cause `exp(scores - max_scores)` to become `exp(-inf - (-inf)) = exp(NaN)`.
        # We replace -inf with 0.0 for the subtraction. This is safe because the corresponding
        # `exp(scores_chunk - 0.0)` will be `exp(-inf)`, which correctly evaluates to 0.
        safe_max_scores = torch.where(torch.isinf(max_scores), 0.0, max_scores)
        
        # --- STEP 3 (PASS 2): Compute weighted values and the normalizer (denominator) ---
        # This is the second part of the online softmax, calculating the numerator and
        # denominator of the final attention formula.
        # Initialize accumulators in float32.
        weighted_values = torch.zeros_like(q, dtype=torch.float32)
        normalizer = torch.zeros((B, h, L, 1), device=q.device, dtype=torch.float32)

        # We need to re-iterate through the chunks.
        v_chunks = v.split(self.config.cross_attn_chunk_size, dim=2)
        # Re-create iterators for k and mask for the second pass.
        k_chunks_pass2 = k.split(self.config.cross_attn_chunk_size, dim=2)
        mask_chunks_pass2 = key_padding_mask.to(torch.bool).split(
            self.config.cross_attn_chunk_size, dim=1
        )
        
        for k_chunk, v_chunk, mask_chunk in zip(k_chunks_pass2, v_chunks, mask_chunks_pass2):
            # Re-calculate scores for the chunk.
            scores_chunk = q @ k_chunk.transpose(-1, -2)
            
            # Apply the padding mask again.
            scores_chunk.masked_fill_(mask_chunk.view(B, 1, 1, -1), -torch.inf)
            
            # Calculate scaled, exponentiated scores using the SAFE max scores.
            # This exponentiation is now numerically stable due to float32 and max-score subtraction.
            exp_scores = torch.exp(scores_chunk - safe_max_scores)
            
            # Accumulate the weighted values (the numerator).
            # [B, h, L, N_chunk] @ [B, h, N_chunk, d_head] -> [B, h, L, d_head]
            # The accumulation happens safely in float32.
            weighted_values += exp_scores @ v_chunk
            
            # Accumulate the sum of exponentiated scores (the denominator/normalizer).
            # The accumulation happens safely in float32.
            normalizer += exp_scores.sum(dim=-1, keepdim=True)

        # --- STEP 4: NORMALIZE AND CAST BACK ---
        # Divide the accumulated weighted values by the accumulated normalizer.
        # The +1e-6 epsilon prevents division by zero for any queries that had no
        # valid keys to attend to (their normalizer will be 0).
        output_f32 = weighted_values / (normalizer + 1e-6)

        # Cast the final result back to the original input dtype.
        return output_f32.to(orig_dtype)
        
    def forward(self, indices: torch.Tensor, values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        batch_size = len(offsets) - 1
        if indices.numel() == 0:
            return torch.zeros(batch_size, self.config.d, device=indices.device, dtype=self.latents.dtype)

        tokens_flat = self.tokenizer(indices, values)
        lengths = (offsets[1:] - offsets[:-1]).tolist()
        tokens_padded = nn.utils.rnn.pad_sequence(list(torch.split(tokens_flat, lengths)), batch_first=True, padding_value=0.0)
        B, N_max, _ = tokens_padded.shape
        key_padding_mask = torch.arange(N_max, device=tokens_padded.device)[None, :] >= torch.tensor(lengths, device=tokens_padded.device)[:, None]

        if torch.rand(1) < 0.01: # Sample 1% of the time
            valid_keys_per_sample = (~key_padding_mask).sum(dim=1)
            print(
                f"\n[DEBUG ENCODER] Valid (unmasked) keys per sample in batch: "
                f"Min: {valid_keys_per_sample.min().item()}, "
                f"Mean: {valid_keys_per_sample.float().mean().item():.2f}, "
                f"Max: {valid_keys_per_sample.max().item()}"
            )
            if valid_keys_per_sample.min().item() == 0:
                print("!!! CRITICAL WARNING: A SAMPLE IN THIS BATCH HAS 0 VALID KEYS. THE ENCODER WILL OUTPUT A CONSTANT. !!!\n")
        
        queries, tokens_norm = self.cross_attn_norm_q(self.latents.unsqueeze(0).expand(B, -1, -1)), self.cross_attn_norm_kv(tokens_padded)
        q_proj, (k_proj, v_proj) = self.q_proj(queries), self.kv_proj(tokens_norm).chunk(2, dim=-1)
        h, d_head = self.config.heads_h, self.config.d // self.config.heads_h
        
        q, k, v = (t.view(B, -1, h, d_head).transpose(1, 2) for t in (q_proj, k_proj, v_proj))
        
        if N_max <= self.config.cross_attn_chunk_size and hasattr(F, 'scaled_dot_product_attention'):
            # Create a boolean mask where True means "mask out".
            # The shape (B, h, L, N_max) is explicit and avoids broadcasting ambiguity.
            # `key_padding_mask` is already boolean [B, N_max]
            attn_mask = key_padding_mask[:, None, None, :].expand(B, h, q.size(2), N_max)

            q32, k32, v32 = q.float(), k.float(), v.float()
            attn_output_h = F.scaled_dot_product_attention(q32, k32, v32, attn_mask=attn_mask).to(q.dtype)
        else:
            attn_output_h = self._chunked_cross_attention(q, k, v, key_padding_mask)
            
        attn_output = self.cross_attn_out_proj(attn_output_h.transpose(1, 2).reshape(B, self.config.latents_L, self.config.d))
        latents = queries + attn_output
        # Use gradient checkpointing for memory efficiency
        latents = torch.utils.checkpoint.checkpoint_sequential(
            list(self.latent_blocks_seq),
            len(self.latent_blocks_seq),
            latents,
            use_reentrant=False
        )

        processed_latents = self.final_norm(latents)

        return processed_latents.mean(dim=1)        

class MLPPredictor(nn.Module):
    """
    BYOL-style MLP predictor. Adds a LayerNorm on the final output
    to stabilize non-contrastive training.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        input_dim = config.d
        hidden_dim = config.d * config.predictor_expansion_factor
        output_dim = config.d

        layers = [nn.LayerNorm(input_dim)]

        if config.predictor_depth <= 1:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.LayerNorm(output_dim))
        else:
            # First layer: input -> hidden
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            # keep this BN if your per-GPU batch isn't tiny; otherwise consider SyncBN or swap to LN
            layers.append(nn.LayerNorm(hidden_dim))

            # Intermediate hidden layers
            for _ in range(config.predictor_depth - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())

            # Final layer: hidden -> output
            layers.append(nn.Linear(hidden_dim, output_dim))
            # NEW: BN on predictor output is key for BYOL-style setups
            layers.append(nn.LayerNorm(output_dim))  # MOST RECENT CHANGE: REMOVE LN ON OUTPUT

        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

class GenePerceiverJEPA(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.student_encoder = GenePerceiverEncoder(config)
        
        # This prevents the teacher from averaging a randomly initialized student at the start.
        self.teacher_encoder = EMA(
            self.student_encoder,
            beta=config.ema_start_decay,
            update_every=1,
            update_after_step=config.ema_warmup_steps
        )

        self.predictor = MLPPredictor(config)
        self.teacher_encoder.ema_model.eval()

    def update_teacher(self):
        self.teacher_encoder.update()

    def _random_block_masking(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        offsets: torch.Tensor,
        lengths: torch.Tensor,
    ):
        """
        JEPA masking with NO dependence on gene ids.
        Each sample: randomly permute positions, take ~mask_ratio as targets,
        split into K blocks; context = complement. No reusable global semantics.
        """
        device = indices.device
        B = int(lengths.numel())
        K = int(self.config.num_targets)
        min_ctx = int(self.config.min_context_genes)
        min_tgt = int(self.config.min_target_genes_per_block)
        mask_ratio = float(self.config.mask_ratio)
        required_min_targets = K * min_tgt

        ctx_idx_list, ctx_val_list, ctx_sizes = [], [], []
        tgt_idx_list, tgt_val_list, tgt_sizes = [], [], []
        pred_to_sample_idx = []

        kept_sample_id = 0
        for i in range(B):
            s = int(offsets[i].item()); e = int(offsets[i + 1].item())
            if e <= s: 
                continue

            ids = indices[s:e]
            vals = values[s:e]
            L = int(ids.numel())

            if L < (min_ctx + required_min_targets):
                continue

            total_target = int(L * mask_ratio)
            total_target = max(required_min_targets, total_target)
            total_target = min(total_target, L - min_ctx)

            # fresh permutation *per sample, per step*
            perm = torch.randperm(L, device=device)
            target_pos_all = perm[:total_target]

            # split targets into K blocks (balanced; each ≥ min_tgt)
            base = total_target // K
            rem = total_target % K
            block_sizes = [max(min_tgt, base + (1 if j < rem else 0)) for j in range(K)]
            # ensure total doesn't exceed total_target
            extra = max(0, sum(block_sizes) - total_target)
            j = 0
            while extra > 0:
                take = min(extra, block_sizes[j] - min_tgt)
                block_sizes[j] -= take
                extra -= take
                j = (j + 1) % K

            start = 0
            target_mask = torch.zeros(L, dtype=torch.bool, device=device)
            blocks_added_for_this_sample = 0
            for bsz in block_sizes:
                end = min(start + bsz, total_target)
                if end <= start:
                    continue
                take = target_pos_all[start:end]
                start = end
                target_mask[take] = True

                tgt_idx_list.append(ids[take])
                tgt_val_list.append(vals[take])
                tgt_sizes.append(int(take.numel()))
                pred_to_sample_idx.append(int(kept_sample_id))
                blocks_added_for_this_sample += 1

            ctx_pos = torch.nonzero(~target_mask, as_tuple=False).flatten()
            if ctx_pos.numel() < min_ctx:
                # rollback blocks for this sample
                for _ in range(blocks_added_for_this_sample):
                    tgt_sizes.pop()
                    tgt_idx_list.pop()
                    tgt_val_list.pop()
                    pred_to_sample_idx.pop()
                continue

            ctx_idx_list.append(ids[ctx_pos])
            ctx_val_list.append(vals[ctx_pos])
            ctx_sizes.append(int(ctx_pos.numel()))
            kept_sample_id += 1

        if kept_sample_id == 0:
            emptyL = torch.empty(0, device=device, dtype=torch.long)
            emptyF = torch.empty(0, device=device, dtype=values.dtype)
            return (
                {"indices": emptyL, "values": emptyF, "offsets": torch.tensor([0], device=device, dtype=torch.long)},
                {"indices": emptyL, "values": emptyF, "offsets": torch.tensor([0], device=device, dtype=torch.long)},
                torch.empty(0, device=device, dtype=torch.long),
            )

        def _stitch(idx_list, val_list, sizes):
            idx = torch.cat(idx_list)
            val = torch.cat(val_list)
            off = F.pad(torch.as_tensor(sizes, device=device, dtype=torch.long).cumsum(0), (1, 0))
            return {"indices": idx, "values": val, "offsets": off}

        context_inputs = _stitch(ctx_idx_list, ctx_val_list, ctx_sizes)
        target_inputs  = _stitch(tgt_idx_list, tgt_val_list, tgt_sizes)
        pred_to_sample_idx = torch.as_tensor(pred_to_sample_idx, device=device, dtype=torch.long)

        return context_inputs, target_inputs, pred_to_sample_idx

    def forward(self, indices: torch.Tensor, values: torch.Tensor, offsets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = indices.device
        if indices.numel() == 0:
            return (torch.empty(0, self.config.d, device=device),
                    torch.empty(0, self.config.d, device=device),
                    torch.empty(0, self.config.d, device=device))

        lengths = offsets[1:] - offsets[:-1]

        # A sample is valid only if it's long enough to provide min context AND required target blocks.
        min_required_genes = (
            self.config.min_context_genes
            + self.config.num_targets * self.config.min_target_genes_per_block
        )
        keep_mask = (lengths >= min_required_genes)

        if not keep_mask.any():
            return (torch.empty(0, self.config.d, device=device),
                    torch.empty(0, self.config.d, device=device),
                    torch.empty(0, self.config.d, device=device))

        if not keep_mask.all():
            retained_indices = torch.where(keep_mask)[0]
            indices_list, values_list = list(torch.split(indices, lengths.tolist())), list(torch.split(values, lengths.tolist()))
            filtered_indices_list = [indices_list[i] for i in retained_indices]
            filtered_values_list  = [values_list[i]  for i in retained_indices]
            indices = torch.cat(filtered_indices_list)
            values  = torch.cat(filtered_values_list)
            lengths = lengths[keep_mask]
            offsets = F.pad(torch.cumsum(lengths, dim=0), (1, 0))

        # === Use stochastic, id-independent masking (fixes leakage) ===
        context_inputs, target_inputs, pred_to_sample_idx = \
            self._random_block_masking(indices, values, offsets, lengths)

        if context_inputs["indices"].numel() == 0 or target_inputs["indices"].numel() == 0:
            return (torch.empty(0, self.config.d, device=device),
                    torch.empty(0, self.config.d, device=device),
                    torch.empty(0, self.config.d, device=device))

        # Student on context
        student_representations = self.student_encoder(**context_inputs)
        student_reps_expanded   = student_representations[pred_to_sample_idx]   # [num_blocks, d]

        predictor_input = student_reps_expanded

        predicted_representations = self.predictor(predictor_input)

        # Teacher on targets (stop-grad happens at loss site)
        with torch.no_grad():
            self.teacher_encoder.ema_model.eval()
            target_representations = self.teacher_encoder.ema_model(**target_inputs)

        # >>> DEBUG 6: teacher dispersion on target blocks
        try:
            t = target_representations.float()
            n = t.shape[0]
            if n > 0:
                # Subsample to <=512 for cost
                if n > 512:
                    idx = torch.randperm(n, device=t.device)[:512]
                    t = t.index_select(0, idx)
                norms = torch.linalg.norm(t, dim=1)
                std_norm = norms.std(unbiased=False)
                tN = F.normalize(t, dim=1, eps=1e-6)
                if tN.shape[0] > 1:
                    sims = tN @ tN.T
                    avg_cos = (sims.sum() - tN.shape[0]) / (tN.shape[0] * (tN.shape[0] - 1))
                else:
                    avg_cos = torch.tensor(0.0, device=t.device)
                # stash for lightning log
                if hasattr(self, "_debug_cache"):
                    self._debug_cache["teacher_avg_cos_targets"] = avg_cos.detach().cpu()
                    self._debug_cache["teacher_stdnorm_targets"] = std_norm.detach().cpu()
        except Exception:
            pass

        return predicted_representations, target_representations, student_representations

    @torch.no_grad()
    def get_embedding(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        offsets: torch.Tensor,
        use_teacher: bool = True,
    ) -> torch.Tensor:
        """
        Returns per-sample embeddings for the provided ragged inputs.

        By default, this method uses the EMA teacher to produce stable, high-quality
        embeddings for inference. Set use_teacher=False to instead use the student
        encoder, which may be preferable for certain diagnostics during training.
        """
        if use_teacher:
            # Ensure teacher is in eval mode and run without grads
            self.teacher_encoder.ema_model.eval()
            return self.teacher_encoder.ema_model(indices, values, offsets)
        else:
            self.student_encoder.eval()
            emb = self.student_encoder(indices, values, offsets)
            self.student_encoder.train()
            return emb
