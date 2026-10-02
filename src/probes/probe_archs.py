"""
Source: https://github.com/AlignmentResearch/obfuscation-atlas/blob/main/obfuscation_atlas/detectors/probe_archs.py

Probe architectures for transformer activation classification.

All probes output (batch, seq, nhead) logits where:
- Single-head probes (Linear, Nonlinear, Attention, Transformer): nhead=1
- Multi-head probes (GDM, MultiHeadLinear): nhead>1

Aggregation is handled separately by SequenceAggregator.
"""

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ==============================================================================
# Base Classes
# ==============================================================================


class Probe(nn.Module, ABC):
    """
    Base class for all probes.

    All probes output (batch, seq, nhead) logits.
    All probes accept padding_mask for uniform API.
    """

    def __init__(self, normalize_input: str = "none"):
        super().__init__()
        self.normalize_input = normalize_input
        self.register_buffer("input_scale", torch.tensor(1.0))
        # Platt scaling params (for calibration after aggregation)
        self.register_buffer("platt_A", torch.tensor(1.0))
        self.register_buffer("platt_B", torch.tensor(0.0))

    @property
    @abstractmethod
    def nhead(self) -> int:
        """Number of output heads."""
        pass

    def _maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply input normalization based on mode."""
        if self.normalize_input == "none":
            return x
        elif self.normalize_input == "l2":
            x_norm = torch.norm(x, dim=-1, keepdim=True)
            return x / (x_norm + 1e-8)
        elif self.normalize_input == "unit_norm":
            return x / self.input_scale
        else:
            raise ValueError(f"Unknown normalize_input mode: {self.normalize_input}")

    def set_input_scale(self, scale: float) -> None:
        """Set input normalization scale for unit_norm mode."""
        self.input_scale = torch.tensor(scale, dtype=self.input_scale.dtype, device=self.input_scale.device)

    def set_platt_params(self, A: float, B: float) -> None:
        """Set Platt scaling parameters for calibration.

        After calling this, predict() will return sigmoid(A * logit + B)
        instead of sigmoid(logit).

        Args:
            A: Scale parameter for logits.
            B: Shift parameter for logits.
        """
        self.platt_A = torch.tensor(A, dtype=self.platt_A.dtype, device=self.platt_A.device)
        self.platt_B = torch.tensor(B, dtype=self.platt_B.dtype, device=self.platt_B.device)

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input activations (batch, seq, d_model)
            padding_mask: Valid token mask (batch, seq), True = valid token
                         Position-wise probes ignore this.
                         Attention-based probes use this internally.

        Returns:
            Logits (batch, seq, nhead)
        """
        pass

    def forward_qv(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning Q and V for attention-based aggregation.

        Default implementation: Q = V (suitable for non-attention probes).
        Override in probes with learned Q projections (e.g., GDMProbe).

        Args:
            x: Input activations (batch, seq, d_model)
            padding_mask: Valid token mask (batch, seq), True = valid token

        Returns:
            Tuple of (Q, V), each (batch, seq, nhead)
        """
        v = self.forward(x, padding_mask)
        return v, v

    def copy_buffers_from(self, other: "Probe", strict: bool = False) -> None:
        """Copy buffers from another probe."""
        src_buffers = dict(other.named_buffers())
        dst_buffers = dict(self.named_buffers())

        if strict and src_buffers.keys() != dst_buffers.keys():
            raise ValueError(f"Buffer mismatch: {src_buffers.keys()} vs {dst_buffers.keys()}")

        for name, buffer in src_buffers.items():
            if name in dst_buffers:
                getattr(self, name).copy_(buffer)


class LinearProbe(Probe):
    def __init__(self, d_model: int, nhead: int = 1, normalize_input: str = "none"):
        super().__init__(normalize_input=normalize_input)
        self.d_model = d_model
        self._nhead = nhead
        self.linear = nn.Linear(d_model, nhead)

    @property
    def nhead(self) -> int:
        return self._nhead

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self._maybe_normalize(x)
        return self.linear(x)  # (batch, seq, nhead)

    def compute_orthogonality_loss(self) -> torch.Tensor:
        """Orthogonality regularization for multi-head probes."""
        if self._nhead <= 1:
            return torch.tensor(0.0, device=self.linear.weight.device)

        weight = self.linear.weight  # (nhead, d_model)
        normalized = weight / (weight.norm(dim=1, keepdim=True) + 1e-8)
        gram = torch.mm(normalized, normalized.t()).abs()
        identity = torch.eye(self._nhead, device=weight.device, dtype=weight.dtype)
        off_diag_sum = (gram - identity).abs().sum()
        num_pairs = self._nhead * (self._nhead - 1)
        return off_diag_sum / max(num_pairs, 1)

class PolytopeProbe(Probe):
    """Safety Polytope (SaP; Chen, As & Krause, ICML 2025, arXiv:2505.24445).

    Learns K half-spaces phi_k . f(h) - b_k <= 0 over a shared feature encoder.
    The encoder is ReLU(W h) when use_nonlinear=True and the identity otherwise.
    A representation is benign when it satisfies every constraint.

    forward() returns the maximum facet violation at each token, with shape
    (batch, seq, 1). Positive scores indicate harmful representations.
    Taking this maximum before token aggregation preserves token-specific
    violations; nonlinear encoding also prevents pooling activations first.

    Fitting and LoRA supervision use violations(), which returns the full
    (batch, seq, K) tensor. See polytope_fit_loss and
    utils.finetune_utils.compute_polytope_lora_loss.

    Constructor arguments are stored as same-named attributes for
    train.extract_model_config and model save/load support.
    """

    def __init__(
        self,
        d_model: int,
        num_facets: int = 16,
        feature_dim: int = 4096,
        use_nonlinear: bool = True,
        normalize_input: str = "none",
        max_temperature: float = 0.0,
    ):
        super().__init__(normalize_input=normalize_input)
        self.d_model = d_model
        self.num_facets = num_facets
        self.feature_dim = feature_dim
        self.use_nonlinear = use_nonlinear
        # 0.0 -> hard max over facets; >0 -> temperature-T log-sum-exp smooth max.
        self.max_temperature = max_temperature

        if use_nonlinear:
            enc = nn.Linear(d_model, feature_dim)
            nn.init.xavier_uniform_(enc.weight)
            nn.init.zeros_(enc.bias)
            self.feature_extractor = nn.Sequential(enc, nn.ReLU())
            phi_dim = feature_dim
        else:
            self.feature_extractor = nn.Identity()
            phi_dim = d_model
        self._phi_dim = phi_dim

        # Initialize facet weights and thresholds with standard normal samples, following SaP.
        self.phi = nn.Parameter(torch.randn(num_facets, phi_dim))
        self.threshold = nn.Parameter(torch.randn(num_facets))

    @property
    def nhead(self) -> int:
        # Deliberately 1, not num_facets: multihead aggregators SUM over heads,
        # which would turn the membership test into a sum of facet violations.
        return 1

    def violations(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ):
        """Signed per-facet violations, (batch, seq, K). v_k > 0 == facet k violated."""
        x = self._maybe_normalize(x)
        f = self.feature_extractor(x)
        v = f @ self.phi.to(f.dtype).t() - self.threshold.to(f.dtype)
        return (v, f) if return_features else v

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        v = self.violations(x, padding_mask=padding_mask)
        if self.max_temperature > 0:
            t = self.max_temperature
            s = t * torch.logsumexp(v / t, dim=-1)
        else:
            s = v.amax(dim=-1)
        return s.unsqueeze(-1)  # (batch, seq, 1)

    def set_retrain_scope(self, scope: str) -> None:
        """Control what a warm-started retrain is allowed to move.

        'constraints_only' freezes the concept encoder and refits only phi/b.
        That is the default for the continuously-updated regime: 200 steps on
        250 sequences cannot meaningfully update a 16.8M-parameter encoder, and
        holding W fixed keeps facet index k meaning the same concept across
        steps, which is what makes per-facet drift analysis possible.
        """
        if scope == "constraints_only":
            self.requires_grad_(True)
            self.feature_extractor.requires_grad_(False)
        elif scope == "all":
            self.requires_grad_(True)
        else:
            raise ValueError(f"unknown retrain_scope: {scope!r}")

    def facet_usage(self, v: torch.Tensor) -> torch.Tensor:
        """Histogram over argmax facets, for logging and interpretability."""
        return torch.bincount(
            v.reshape(-1, self.num_facets).argmax(dim=-1), minlength=self.num_facets
        ).float()

class QuadraticProbe(Probe):
    """Quadratic probe: x^T M x + w^T x + b"""

    def __init__(self, d_model: int, normalize_input: str = "none"):
        super().__init__(normalize_input=normalize_input)
        self.d_model = d_model
        self.M = nn.Parameter(torch.randn(d_model, d_model) / d_model**0.5)
        self.linear = nn.Linear(d_model, 1)

    @property
    def nhead(self) -> int:
        return 1

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self._maybe_normalize(x)
        batch_dims = x.shape[:-1]
        x_flat = x.view(-1, x.shape[-1])

        xM = torch.matmul(x_flat.unsqueeze(1), self.M)
        xMx = torch.matmul(xM, x_flat.unsqueeze(-1))
        quadratic_term = xMx.squeeze(-1).squeeze(-1).view(*batch_dims)

        linear_term = self.linear(x).squeeze(-1)

        return (quadratic_term + linear_term).unsqueeze(-1)  # (batch, seq, 1)


class NonlinearProbe(Probe):
    """Simple 2-layer MLP probe."""

    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        nhead: int = 1,
        dropout: float = 0.0,
        normalize_input: str = "none",
    ):
        super().__init__(normalize_input=normalize_input)
        self.d_model = d_model
        self.d_mlp = d_mlp
        self._nhead = nhead

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_mlp, nhead),
        )

    @property
    def nhead(self) -> int:
        return self._nhead

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self._maybe_normalize(x)
        return self.mlp(x)  # (batch, seq, nhead)


class AttentionProbe(Probe):
    """
    Self-attention probe. Uses padding_mask to ignore padding in attention.

    Optionally uses sliding window to limit attention context.
    """

    def __init__(
        self,
        d_model: int,
        d_proj: int,
        nhead: int = 8,
        sliding_window: int | None = None,
        max_length: int = 8192,
        use_checkpoint: bool = True,
        normalize_input: str = "none",
    ):
        super().__init__(normalize_input=normalize_input)
        self.d_model = d_model
        self.d_proj = d_proj
        self.num_heads = nhead
        self.sliding_window = sliding_window
        self.max_length = max_length
        self.use_checkpoint = use_checkpoint

        self.qkv_proj = nn.Linear(d_model, 3 * d_proj * nhead)
        self.out_proj = nn.Linear(d_proj * nhead, 1)

        # Pre-compute base causal/sliding mask if using sliding window
        if sliding_window is not None:
            base_mask = self._build_base_mask(max_length, sliding_window)
            self.register_buffer("base_mask", base_mask)
        else:
            self.register_buffer("base_mask", None)

    @property
    def nhead(self) -> int:
        return 1

    def _build_base_mask(self, seq_len: int, window_size: int) -> torch.Tensor:
        """
        Build causal sliding window mask.

        Position i can attend to positions [max(0, i-window+1), i].

        Returns:
            Boolean mask (seq, seq) where True = can attend
        """
        q_idx = torch.arange(seq_len).unsqueeze(1)
        kv_idx = torch.arange(seq_len).unsqueeze(0)
        causal = q_idx >= kv_idx
        windowed = (q_idx - kv_idx) < window_size
        return causal & windowed

    def _build_attn_mask(
        self,
        seq_len: int,
        padding_mask: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build attention mask combining causal/sliding window and padding.

        Args:
            seq_len: Sequence length
            padding_mask: (batch, seq) where True = valid token
            device: Target device
            dtype: Target dtype

        Returns:
            Attention mask for scaled_dot_product_attention
        """
        # Get base causal/sliding mask
        if self.base_mask is not None:
            # Use pre-computed sliding window mask
            base_mask = self.base_mask[:seq_len, :seq_len]  # (seq, seq)
        else:
            # Full causal mask
            base_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

        if padding_mask is None:
            # Just causal/sliding, no padding
            attn_mask = torch.where(base_mask, 0.0, float("-inf"))
            return attn_mask.to(dtype)

        # Combine with padding mask
        # base_mask: (seq, seq) -> (1, 1, seq, seq)
        base_mask = base_mask.unsqueeze(0).unsqueeze(0)

        # Key padding: can't attend TO padding positions
        # padding_mask: (batch, seq) where True = valid
        # -> (batch, 1, 1, seq)
        key_mask = padding_mask.unsqueeze(1).unsqueeze(2)

        # Combined: can attend if base allows AND key is valid
        combined = base_mask.to(device) & key_mask

        attn_mask = torch.where(combined, 0.0, float("-inf"))
        return attn_mask.to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert x.dim() == 3, "Input must be (batch, seq, d_model)"
        x = self._maybe_normalize(x)
        batch_size, seq_len, _ = x.shape

        attn_mask = self._build_attn_mask(seq_len, padding_mask, x.device, x.dtype)

        def compute_attention(x: torch.Tensor) -> torch.Tensor:
            qkv = self.qkv_proj(x)
            qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.d_proj)
            q, k, v = qkv.unbind(2)
            q = q.transpose(1, 2)  # (batch, num_heads, seq, d_proj)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=False,  # Causality is handled by attn_mask
            )
            return attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        if self.use_checkpoint and self.training:
            attn_output = checkpoint(compute_attention, x, use_reentrant=False)
        else:
            attn_output = compute_attention(x)

        return self.out_proj(attn_output)  # (batch, seq, 1)


class TransformerProbe(Probe):
    """
    Transformer encoder probe. Uses padding_mask for src_key_padding_mask.

    Uses full transformer encoder layers internally, outputs single-head scores.
    """

    def __init__(
        self,
        d_model: int,
        nlayer: int = 1,
        nhead: int = 8,
        d_mlp: int = 512,
        dropout: float = 0.0,
        activation: str = "relu",
        norm_first: bool = True,
        use_checkpoint: bool = True,
        normalize_input: str = "none",
    ):
        super().__init__(normalize_input=normalize_input)
        self.d_model = d_model
        self.nlayer = nlayer
        self.num_heads = nhead
        self.d_mlp = d_mlp
        self.use_checkpoint = use_checkpoint

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_mlp,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=nlayer,
            norm=nn.LayerNorm(d_model) if norm_first else None,
        )
        self.out_proj = nn.Linear(d_model, 1)

    @property
    def nhead(self) -> int:
        return 1

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert x.dim() == 3, "Input must be (batch, seq, d_model)"
        x = self._maybe_normalize(x)
        seq_len = x.size(1)

        # Causal mask for autoregressive attention
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device, dtype=x.dtype)

        # PyTorch TransformerEncoder expects src_key_padding_mask where True = IGNORE
        # Invert the valid-token mask for padding
        src_key_padding_mask = None
        if padding_mask is not None:
            src_key_padding_mask = ~padding_mask  # Invert: True = padding = ignore

        if self.use_checkpoint and self.training:
            out = checkpoint(
                self.transformer,
                x,
                causal_mask,
                src_key_padding_mask,
                True,  # is_causal
                use_reentrant=False,
            )
        else:
            out = self.transformer(
                x,
                mask=causal_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=True,
            )
        return self.out_proj(out)  # (batch, seq, 1)


# ==============================================================================
# Multi-Head Probes (nhead > 1)
# ==============================================================================


class GDMProbe(Probe):
    """
    Multi-head probe from "Building Production-Ready Probes For Gemini".

    Architecture:
    1. MLP transformation (no ReLU after final layer)
    2. Per-head Q and V projections
    3. Output: per-position, per-head V scores

    For attention-based aggregation (rolling_attention), use forward_qv().
    For multimax aggregation, only V scores are needed (forward()).

    Paper recommends:
    - Train with rolling_attention aggregation (smooth gradients)
    - Eval with multimax aggregation (robust to long contexts)
    """

    def __init__(
        self,
        d_model: int,
        d_proj: int = 100,
        nhead: int = 10,
        num_mlp_layers: int = 2,
        normalize_input: str = "none",
    ):
        super().__init__(normalize_input=normalize_input)
        self.d_model = d_model
        self.d_proj = d_proj
        self._nhead = nhead
        self.num_mlp_layers = num_mlp_layers

        # MLP transformation: Linear -> [ReLU -> Linear] * (num_layers - 1)
        # No ReLU after final layer (per paper Section 3.1.3)
        mlp_layers: list[nn.Module] = [nn.Linear(d_model, d_proj)]
        for _ in range(num_mlp_layers - 1):
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(d_proj, d_proj))
        self.mlp = nn.Sequential(*mlp_layers)

        # Per-head Q and V projections
        self.q_proj = nn.Linear(d_proj, nhead)
        self.v_proj = nn.Linear(d_proj, nhead)

    @property
    def nhead(self) -> int:
        return self._nhead

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Returns V scores: (batch, seq, nhead)."""
        x = self._maybe_normalize(x)
        y = self.mlp(x)
        return self.v_proj(y)

    def forward_qv(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (Q, V) for attention-based aggregation."""
        x = self._maybe_normalize(x)
        y = self.mlp(x)
        q = self.q_proj(y)
        v = self.v_proj(y)
        return q, v


# ==============================================================================
# Aggregation
# ==============================================================================


class SequenceAggregator:
    """
    Aggregates (batch, seq, nhead) logits to (batch,) logits.

    Methods:
    - mean: Mean over seq, sum over heads
    - max: Max over seq (after summing heads)
    - sum: Sum over seq and heads
    - last: Last valid position, sum over heads
    - multimax: Max per head over seq, then sum over heads (Equation 9 from GDM paper)
    - attention: Softmax-weighted sum over seq (Equation 8 from GDM paper)
    - rolling_attention: Sliding window attention + max over windows (Equation 10 from GDM paper)
    """

    def __init__(
        self,
        method: str = "mean",
        sliding_window: int | None = None,
    ):
        self.method = method
        self.sliding_window = sliding_window

    @property
    def needs_q(self) -> bool:
        """Whether this aggregation method requires Q scores."""
        return self.method in ["attention", "rolling_attention"]

    def __call__(
        self,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Aggregate sequence of logits.

        Args:
            v: Value logits (batch, seq, nhead)
            mask: Valid token mask (batch, seq), True = valid
            q: Query logits for attention methods (batch, seq, nhead)

        Returns:
            Aggregated logits (batch,)
        """
        if self.method == "mean":
            return self._mean(v, mask)
        elif self.method == "max":
            return self._max(v, mask)
        elif self.method == "sum":
            return self._sum(v, mask)
        elif self.method == "last":
            return self._last(v, mask)
        elif self.method == "multimax":
            return self._multimax(v, mask)
        elif self.method == "attention":
            return self._attention(q if q is not None else v, v, mask)
        elif self.method == "rolling_attention":
            return self._rolling_attention(q if q is not None else v, v, mask)
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")

    def _mean(self, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Mean over sequence, sum over heads."""
        v_summed = v.sum(dim=-1)  # (batch, seq)
        if mask is None:
            return v_summed.mean(dim=1)
        return (v_summed * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def _max(self, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Max over sequence (after summing heads)."""
        v_summed = v.sum(dim=-1)  # (batch, seq)
        if mask is not None:
            v_summed = v_summed.masked_fill(~mask, float("-inf"))
        return v_summed.max(dim=1).values

    def _sum(self, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Sum over sequence and heads."""
        v_summed = v.sum(dim=-1)  # (batch, seq)
        if mask is None:
            return v_summed.sum(dim=1)
        return (v_summed * mask.float()).sum(dim=1)

    def _last(self, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Last valid position, sum over heads."""
        v_summed = v.sum(dim=-1)  # (batch, seq)
        if mask is None:
            return v_summed[:, -1]
        idx = mask.long().cumsum(dim=1).argmax(dim=1)
        batch_idx = torch.arange(v_summed.size(0), device=v.device)
        return v_summed[batch_idx, idx]

    def _multimax(self, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Max per head over sequence, then sum over heads (Equation 9)."""
        if mask is not None:
            v = v.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        max_per_head = v.max(dim=1).values  # (batch, nhead)
        return max_per_head.sum(dim=-1)  # (batch,)

    def _attention(self, q: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Softmax-weighted sum over sequence (Equation 8)."""
        if mask is not None:
            q = q.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        weights = F.softmax(q, dim=1)

        if mask is not None:
            weights = weights.masked_fill(~mask.unsqueeze(-1), 0.0)

        # Weighted sum per head, then sum over heads
        out = (weights * v).sum(dim=1)  # (batch, nhead)
        return out.sum(dim=-1)  # (batch,)

    def _rolling_attention(self, q: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Sliding window attention + max over windows (Equation 10)."""
        if self.sliding_window is None:
            raise ValueError("sliding_window must be specified for rolling_attention")
        batch_size, seq_len, nhead = v.shape
        w = self.sliding_window

        if seq_len < w:
            return self._attention(q, v, mask)

        if mask is not None:
            q = q.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        weights = F.softmax(q, dim=1)

        if mask is not None:
            weights = weights.masked_fill(~mask.unsqueeze(-1), 0.0)

        # Unfold to get sliding windows: (batch, seq, nhead) -> (batch, num_windows, nhead, w)
        # num_windows = seq_len - w + 1
        w_windows = weights.unfold(dimension=1, size=w, step=1).permute(0, 1, 3, 2)  # (batch, num_windows, w, nhead)
        v_windows = v.unfold(dimension=1, size=w, step=1).permute(0, 1, 3, 2)  # (batch, num_windows, w, nhead)

        # Normalize weights within each window
        w_sum = w_windows.sum(dim=2, keepdim=True) + 1e-8  # (batch, num_windows, 1, nhead)
        w_norm = w_windows / w_sum

        # Weighted average within each window
        window_avg = (w_norm * v_windows).sum(dim=2)  # (batch, num_windows, nhead)

        # Max over windows, then sum over heads
        max_per_head = window_avg.max(dim=1).values  # (batch, nhead)
        return max_per_head.sum(dim=-1)  # (batch,)


# ==============================================================================
# Aggregated Probe Wrapper
# ==============================================================================


class AggregatedProbe(nn.Module):
    """
    Wraps a probe with an aggregator for sequence-level predictions.

    Pipeline:
        probe(x) → (batch, seq, nhead) logits
        aggregator(logits, mask) → (batch,) aggregated logits
        sigmoid(platt_A * logits + platt_B) → (batch,) probabilities

    Usage:
        # Training with rolling attention
        probe = GDMProbe(d_model=4096, nhead=10)
        train_wrapper = AggregatedProbe(probe, rolling_attention_aggregator())

        # Eval with multimax (same probe, different aggregation)
        eval_wrapper = AggregatedProbe(probe, multimax_aggregator())
    """

    def __init__(
        self,
        probe: Probe,
        aggregator: SequenceAggregator,
    ):
        super().__init__()
        self.probe = probe
        self.aggregator = aggregator

        # Platt scaling parameters (specific to probe + aggregation combo)
        self.register_buffer("platt_A", torch.tensor(1.0))
        self.register_buffer("platt_B", torch.tensor(0.0))

    @property
    def nhead(self) -> int:
        return self.probe.nhead

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Returns aggregated logits: (batch,).

        Args:
            x: Input activations (batch, seq, d_model)
            mask: Valid token mask (batch, seq), True = valid
        """
        if self.aggregator.needs_q:
            q, v = self.probe.forward_qv(x)
        else:
            q, v = None, self.probe(x)

        return self.aggregator(v, mask, q=q)

    def predict(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Returns calibrated probabilities: (batch,).

        Args:
            x: Input activations (batch, seq, d_model)
            mask: Valid token mask (batch, seq), True = valid
        """
        logits = self.forward(x, mask)
        return torch.sigmoid(self.platt_A * logits + self.platt_B)

    def set_platt_params(self, A: float, B: float) -> None:
        """Set Platt scaling parameters for calibration."""
        self.platt_A.fill_(A)
        self.platt_B.fill_(B)

    def train(self, mode: bool = True):
        super().train(mode)
        self.probe.train(mode)
        return self

    def eval(self):
        super().eval()
        self.probe.eval()
        return self


# ==============================================================================
# Convenience Constructors
# ==============================================================================


def mean_aggregator() -> SequenceAggregator:
    return SequenceAggregator(method="mean")


def max_aggregator() -> SequenceAggregator:
    return SequenceAggregator(method="max")


def sum_aggregator() -> SequenceAggregator:
    return SequenceAggregator(method="sum")


def last_aggregator() -> SequenceAggregator:
    return SequenceAggregator(method="last")


def multimax_aggregator() -> SequenceAggregator:
    return SequenceAggregator(method="multimax")


def attention_aggregator() -> SequenceAggregator:
    return SequenceAggregator(method="attention")


def rolling_attention_aggregator(window: int = 10) -> SequenceAggregator:
    return SequenceAggregator(method="rolling_attention", sliding_window=window)


# ==============================================================================
# Loss Computation
# ==============================================================================

# training.py


# ------------------------------------------------------------------------------
# Safety Polytope (SaP) fitting loss
# ------------------------------------------------------------------------------


def _facet_entropy(assign: torch.Tensor, num_facets: int) -> torch.Tensor:
    """Shannon entropy (bits) of the facet-assignment histogram.

    NOT differentiable w.r.t. the probe parameters -- it is built from integer
    argmax indices. SaP's loss subtracts this term, but since it carries no
    gradient it only shifts the reported scalar; the actual facet
    diversification comes from violation_entropy_assignment below.
    """
    counts = torch.bincount(assign, minlength=num_facets).float()
    p = counts / counts.sum().clamp(min=1e-10)
    return -(p * torch.log2(p + 1e-10)).sum()


def violation_entropy_assignment(
    v_unsafe: torch.Tensor,
    num_facets: int,
    valid_edges_threshold: float = 0.0,
    max_attempts: int = 100,
    entropy_threshold: float | None = None,
):
    """SaP Appendix A: entropy-based heuristic facet assignment.

    Start from argmax_k v, then repeatedly pick a random unsafe row and move it
    to a different facet whose violation still exceeds `valid_edges_threshold`,
    keeping the move only if it raises the assignment histogram's entropy. This
    stops the fit from routing every unsafe example through one facet.

    Args:
        v_unsafe: (N, K) violations for the unsafe rows only.
        num_facets: K.
        valid_edges_threshold: tau in the paper; candidate facets must exceed it.
        max_attempts: reassignment attempts before giving up.
        entropy_threshold: target entropy in bits; defaults to 0.5 * log2(K).

    Returns:
        (assigned_violation (N,), assigned_facet (N,), entropy scalar)
    """
    n, k = v_unsafe.shape
    if entropy_threshold is None:
        entropy_threshold = 0.5 * math.log2(max(k, 2))

    val, assign = v_unsafe.max(dim=1)
    val, assign = val.clone(), assign.clone()
    if n == 0:
        return val, assign, v_unsafe.new_zeros(())

    ent = _facet_entropy(assign, num_facets)
    for _ in range(max_attempts):
        if float(ent) >= entropy_threshold:
            break
        i = int(torch.randint(n, (1,)))
        sorted_v, sorted_i = torch.sort(v_unsafe[i].detach(), descending=True)
        valid = sorted_i[sorted_v > valid_edges_threshold]
        if valid.numel() <= 1:
            if k < 2:
                break
            cand = int(sorted_i[1])
        else:
            cand = int(valid[int(torch.randint(1, valid.numel(), (1,)))])
        trial = assign.clone()
        trial[i] = cand
        new_ent = _facet_entropy(trial, num_facets)
        if float(new_ent) > float(ent):
            assign = trial
            val = val.clone()
            val[i] = v_unsafe[i, cand]
            ent = new_ent

    # Re-gather so the returned violations carry gradient from the final assignment.
    val = v_unsafe.gather(1, assign.unsqueeze(1)).squeeze(1)
    return val, assign, ent


def polytope_fit_loss(
    probe: "PolytopeProbe",
    activations: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    margin: float = 1.0,
    unsafe_weight: float = 2.0,
    f_l1_weight: float = 1.0e-3,
    phi_l1_weight: float = 1.0e-4,
    entropy_mode: str = "sap_reassign",
    entropy_weight: float = 1.0,
    entropy_temp: float = 1.0,
    valid_edges_threshold: float = 0.0,
    max_attempts: int = 100,
    assignment_level: str = "token",
    safe_reduction: str = "sum",
    return_stats: bool = False,
):
    """Compute the SaP fitting objective with label 1 harmful and label 0 benign.

    Positive violations indicate harmful representations. Unlike the reference
    implementation's labels, benign examples select the safe branch at label 0.

    The benign term sums relu(margin + v_k) over facets by default;
    safe_reduction="mean" uses a facet mean instead. The harmful term is
    relu(margin - v_assigned), weighted by unsafe_weight. Both terms are
    averaged over their respective valid tokens. L1 penalties regularize
    encoder features and facet weights.

    assignment_level="token" assigns each harmful token to a facet.
    assignment_level="sequence" uses one facet per harmful sequence,
    selected from its maximum violations across valid tokens.

    Discrete reassignment entropy affects the reported loss without a gradient;
    entropy_mode="soft" provides a differentiable entropy term.
    """
    v, f = probe.violations(activations, padding_mask=mask, return_features=True)
    if v.ndim == 2:  # pre-aggregated activations: (batch, K)
        v, f = v.unsqueeze(1), f.unsqueeze(1)
    B, S, K = v.shape

    if mask is None:
        mask = torch.ones(B, S, dtype=torch.bool, device=v.device)
    mask = mask.to(v.device).bool()

    # Collapse (batch, seq) labels down to one label per sequence.
    if labels.ndim == 2:
        idx = mask.long().cumsum(dim=1).argmax(dim=1)
        labels = labels[torch.arange(B, device=labels.device), idx]
    y_seq = labels.float().reshape(B).to(v.device)  # 1 = harmful
    y_tok = y_seq[:, None].expand(B, S)

    tok = mask.reshape(-1)
    v_t = v.reshape(-1, K)[tok].float()
    f_t = f.reshape(-1, f.shape[-1])[tok].float()  # fp32: bf16 L1 over 4-16k dims loses precision
    y_t = y_tok.reshape(-1)[tok]
    harm_t = y_t > 0.5
    ben_t = ~harm_t

    zero = v_t.new_zeros(())

    # --- safe side (benign): every facet satisfied with margin -----------------
    # Sum the benign penalty over facets, then average over tokens.
    # Using a facet mean changes its weight relative to the single-facet harmful penalty.
    if bool(ben_t.any()):
        _sv = torch.relu(margin + v_t[ben_t])
        safe = (_sv.sum(dim=1) if safe_reduction == "sum" else _sv.mean(dim=1)).mean()
    else:
        safe = zero

    # --- unsafe side (harmful): the assigned facet violated with margin -------
    ent_val = zero
    if bool(harm_t.any()):
        if assignment_level == "sequence":
            harm_seq = y_seq > 0.5
            # Max over valid tokens gives each sequence its violation profile.
            v_seq = v.masked_fill(~mask[..., None], -1e4).amax(dim=1)[harm_seq].float()  # (Bh, K)
            if entropy_mode == "sap_reassign":
                _, assign_seq, ent_val = violation_entropy_assignment(
                    v_seq, K, valid_edges_threshold, max_attempts
                )
            else:
                assign_seq = v_seq.argmax(dim=1)
                ent_val = _facet_entropy(assign_seq, K)
            assign_full = torch.zeros(B, dtype=torch.long, device=v.device)
            assign_full[harm_seq] = assign_seq
            v_star = v.gather(2, assign_full[:, None, None].expand(B, S, 1)).squeeze(2)
            unsafe_v = v_star.reshape(-1)[tok][harm_t].float()
        elif assignment_level == "token":
            if entropy_mode == "sap_reassign":
                unsafe_v, assign_seq, ent_val = violation_entropy_assignment(
                    v_t[harm_t], K, valid_edges_threshold, max_attempts
                )
            else:
                unsafe_v, assign_seq = v_t[harm_t].max(dim=1)
                ent_val = _facet_entropy(assign_seq, K)
        else:
            raise ValueError(f"unknown assignment_level: {assignment_level!r}")
        unsafe = torch.relu(margin - unsafe_v).mean()
    else:
        unsafe = zero

    loss = safe + unsafe_weight * unsafe
    loss = loss + f_l1_weight * f_t.abs().sum(dim=1).mean()
    loss = loss + phi_l1_weight * probe.phi.float().abs().sum(dim=1).mean()

    # Discrete assignment entropy has no gradient; it changes the reported loss.
    # Facet diversification comes from randomized reassignment.
    if entropy_mode in ("sap_reassign", "none"):
        loss = loss - entropy_weight * ent_val.detach()

    # Soft assignment entropy provides a differentiable alternative.
    if entropy_mode == "soft" and bool(harm_t.any()):
        p = torch.softmax(v_t[harm_t] / entropy_temp, dim=1).mean(dim=0)
        loss = loss - entropy_weight * (-(p * (p + 1e-10).log2()).sum())

    if not return_stats:
        return loss

    with torch.no_grad():
        stats = {
            "polytope/safe_hinge": float(safe),
            "polytope/unsafe_hinge": float(unsafe),
            "polytope/facet_entropy_bits": float(ent_val),
            "polytope/max_entropy_bits": math.log2(max(K, 2)),
            "polytope/frac_harmful_outside": (
                float((v_t[harm_t].amax(1) > 0).float().mean()) if bool(harm_t.any()) else 0.0
            ),
            "polytope/frac_benign_outside": (
                float((v_t[ben_t].amax(1) > 0).float().mean()) if bool(ben_t.any()) else 0.0
            ),
            "polytope/active_facets": float((probe.phi.norm(dim=1) > 1e-3).sum()),
            "polytope/f_l1": float(f_t.abs().sum(1).mean()),
        }
    return loss, stats


def compute_loss(
    probe: Probe,
    activations: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None = None,
    aggregator: SequenceAggregator | None = None,
    loss_type: str = "bce",
    hinge_margin: float = 1.0,
    loss_kwargs: dict | None = None,
) -> torch.Tensor:
    """
    Compute probe-supervision loss.

    All probes output (batch, seq, nhead). This function handles:
    1. Token-level training: aggregator=None, loss computed per-token
    2. Sequence-level training: aggregator provided, loss computed per-sequence

    Args:
        probe: Probe instance (outputs (batch, seq, nhead))
        activations: Input activations (batch, seq, d_model)
        labels: Binary labels - (batch,) for sequence-level, (batch, seq) for token-level
        mask: Valid token mask (batch, seq), True = valid token
        aggregator: Optional aggregator for sequence-level training.
                   If provided, aggregates logits before computing loss.
        loss_type: 'bce' (default) or 'hinge'. Hinge uses
                   L = mean_i max(0, hinge_margin - y'_i * z_i) with y' = 2*y-1.
                   Saturates exactly when y'_i * z_i >= hinge_margin, removing
                   the unbounded "keep pushing past the boundary" gradient that
                   sigmoid-BCE has under FROZEN regimes (§5.4 of the report).
        hinge_margin: margin parameter for hinge loss. Default 1.0.
        loss_kwargs: extra keyword arguments for the chosen loss_type. Only
                   consumed by 'polytope' (see polytope_fit_loss).

    Returns:
        Scalar loss tensor
    """
    if loss_type == "polytope":
        # The polytope loss needs the raw per-facet, per-token violation tensor,
        # so it bypasses the aggregator entirely rather than consuming pooled
        # logits. `aggregator` is still accepted (and ignored) so callers do not
        # have to special-case this loss.
        return polytope_fit_loss(probe, activations, labels, mask=mask, **(loss_kwargs or {}))

    padding_mask = mask  # True indicates a valid token

    if aggregator is not None:
        # Sequence-level training: aggregate then compute loss
        if aggregator.needs_q:
            assert hasattr(probe, "forward_qv"), "Probe must have forward_qv method for sequence-level training"
            q, v = probe.forward_qv(activations, padding_mask=padding_mask)
        else:
            q = None
            v = probe(activations, padding_mask=padding_mask)

        # Aggregate: (batch, seq, nhead) -> (batch,)
        logits = aggregator(v, mask, q=q)

        # Ensure labels are (batch,) for sequence-level
        if labels.ndim == 2:
            if mask is not None:
                # Take label from last valid position
                idx = mask.long().cumsum(dim=1).argmax(dim=1)
                batch_idx = torch.arange(labels.size(0), device=labels.device)
                labels = labels[batch_idx, idx]
            else:
                labels = labels[:, -1]
    else:
        # Token-level training: loss per token
        logits = probe(activations, padding_mask=padding_mask)  # (batch, nhead)
        assert labels.ndim == 1, "Labels must be 1D for token-level training"
        assert logits.ndim == 2 and logits.shape[0] == labels.shape[0], (
            f"Probe must output (batch, nhead), got {logits.shape}"
        )
        logits = logits.sum(dim=-1)

        # Apply mask
        if mask is not None:
            logits = logits[mask]
            if labels.ndim == 2:
                labels = labels[mask]
            else:
                # Expand labels to match mask shape then apply
                labels = labels.unsqueeze(1).expand(-1, mask.size(1))[mask]
        if logits.numel() == 0:
            return torch.tensor(0.0, device=activations.device, requires_grad=True)

    labels = labels.float()
    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(logits, labels)
    elif loss_type == "hinge":
        # y' in {-1, +1}; hinge = max(0, margin - y' * z).
        y_pm = 2.0 * labels - 1.0
        return torch.clamp(hinge_margin - y_pm * logits, min=0.0).mean()
    else:
        raise ValueError(f"unknown loss_type: {loss_type!r}")


# ==============================================================================
# Utility Functions
# ==============================================================================


def is_multihead_probe(probe: Probe) -> bool:
    """Check if probe has multiple output heads."""
    return probe.nhead > 1