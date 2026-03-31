"""
TurboQuant end-to-end implementation for Hugging Face generation.

Paper:
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
arXiv:2504.19874

This implementation keeps the paper's two-stage key quantizer:
1. Qmse: random rotation + per-coordinate Lloyd-Max quantization
2. Qprod: 1-bit QJL residual sketch for unbiased inner-product estimation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MethodType
from typing import Optional

import torch
import torch.nn as nn
from scipy import integrate
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, DynamicLayer

# Qwen2 Imports
from transformers.models.qwen2.modeling_qwen2 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen2Attention,
    apply_rotary_pos_emb as qwen_apply_rotary_pos_emb,
    eager_attention_forward,
)

# Llama Imports
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
)


def beta_pdf(x: float, d: int) -> float:
    if abs(x) >= 1.0:
        return 0.0
    coeff = math.gamma(d / 2) / (math.sqrt(math.pi) * math.gamma((d - 1) / 2))
    return coeff * (1 - x * x) ** ((d - 3) / 2)


def gaussian_approx_pdf(x: float, d: int) -> float:
    sigma2 = 1.0 / d
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-(x * x) / (2 * sigma2))


def solve_lloyd_max(
    d: int,
    bits: int,
    use_exact: bool = False,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_levels = 2 ** bits
    pdf = (lambda x: beta_pdf(x, d)) if use_exact else (lambda x: gaussian_approx_pdf(x, d))
    sigma = 1.0 / math.sqrt(d)
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            numerator, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            denominator, _ = integrate.quad(pdf, a, b)
            new_centroids.append(numerator / denominator if denominator > 1e-15 else centroids[i])
        if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    return torch.tensor(centroids, dtype=torch.float32), torch.tensor(boundaries, dtype=torch.float32)


class LloydMaxCodebook:
    def __init__(self, d: int, bits: int, use_exact: bool = False):
        self.d = d
        self.bits = bits
        self.n_levels = 2 ** bits
        self.centroids, self.boundaries = solve_lloyd_max(d, bits, use_exact)


def generate_rotation_matrix(d: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(d, d, generator=gen))
    diag_sign = torch.sign(torch.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    return (q * diag_sign.unsqueeze(0)).to(device)


def generate_qjl_matrix(
    d: int,
    m: Optional[int] = None,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> torch.Tensor:
    if m is None:
        m = d
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    return torch.randn(m, d, generator=gen).to(device)


def _pack_unsigned(values: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 1 or bits > 8:
        raise ValueError(f"Unsupported bit-width: {bits}")

    flat = values.reshape(-1, values.shape[-1]).to(torch.int32)
    rows, width = flat.shape
    total_bits = width * bits
    num_bytes = (total_bits + 7) // 8
    packed = torch.zeros(rows, num_bytes, dtype=torch.int32, device=flat.device)
    mask = (1 << bits) - 1

    for i in range(width):
        value = flat[:, i] & mask
        bit_offset = i * bits
        byte_idx = bit_offset // 8
        bit_idx = bit_offset % 8

        packed[:, byte_idx] |= value << bit_idx
        overflow = bit_idx + bits - 8
        if overflow > 0 and byte_idx + 1 < num_bytes:
            packed[:, byte_idx + 1] |= value >> (bits - overflow)

    return packed.to(torch.uint8).reshape(*values.shape[:-1], num_bytes)


def _unpack_unsigned(packed: torch.Tensor, bits: int, original_dim: int) -> torch.Tensor:
    if bits < 1 or bits > 8:
        raise ValueError(f"Unsupported bit-width: {bits}")

    flat = packed.reshape(-1, packed.shape[-1]).to(torch.int32)
    rows = flat.shape[0]
    values = torch.zeros(rows, original_dim, dtype=torch.int32, device=flat.device)
    mask = (1 << bits) - 1

    for i in range(original_dim):
        bit_offset = i * bits
        byte_idx = bit_offset // 8
        bit_idx = bit_offset % 8

        value = flat[:, byte_idx] >> bit_idx
        overflow = bit_idx + bits - 8
        if overflow > 0 and byte_idx + 1 < flat.shape[1]:
            value |= flat[:, byte_idx + 1] << (bits - overflow)
        values[:, i] = value & mask

    return values.reshape(*packed.shape[:-1], original_dim)


def _pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    return _pack_unsigned(indices.long(), bits)


def _unpack_indices(packed: torch.Tensor, bits: int, original_dim: int) -> torch.Tensor:
    return _unpack_unsigned(packed, bits, original_dim)


def _pack_signs(signs: torch.Tensor) -> torch.Tensor:
    bits = (signs >= 0).to(torch.int32)
    return _pack_unsigned(bits, 1)


def _unpack_signs(packed: torch.Tensor, original_dim: int) -> torch.Tensor:
    unpacked = _unpack_unsigned(packed, 1, original_dim)
    return unpacked.to(torch.int8) * 2 - 1


class TurboQuantMSECompressor:
    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device
        self.Pi = generate_rotation_matrix(head_dim, seed=seed, device=device)
        self.centroids = LloydMaxCodebook(head_dim, bits).centroids.to(device)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        b, h, s, d = states.shape
        flat = states.reshape(-1, d).float()
        vec_norms = torch.norm(flat, dim=-1, keepdim=True)
        flat_norm = flat / (vec_norms + 1e-8)
        rotated = flat_norm @ self.Pi.T
        indices = (rotated.unsqueeze(-1) - self.centroids).abs().argmin(dim=-1).to(torch.uint8)

        return {
            "indices_packed": _pack_indices(indices.reshape(b, h, s, d), self.bits),
            "vec_norms": vec_norms.squeeze(-1).to(torch.float16).reshape(b, h, s),
            "shape": (b, h, s, d),
        }

    @torch.no_grad()
    def reconstruct(self, compressed: dict) -> torch.Tensor:
        _, _, _, d = compressed["shape"]
        indices = _unpack_indices(compressed["indices_packed"], self.bits, d).long()
        vec_norms = compressed["vec_norms"].float().unsqueeze(-1)
        return (self.centroids[indices] @ self.Pi) * vec_norms


class TurboQuantProdCompressor:
    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.device = device

        self.mse = TurboQuantMSECompressor(head_dim, self.mse_bits, seed=seed, device=device)
        self.S = generate_qjl_matrix(head_dim, head_dim, seed=seed + 10_000, device=device)
        self.qjl_dim = self.S.shape[0]

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        compressed = self.mse.compress(states)
        mse_recon = self.mse.reconstruct(compressed).reshape(-1, self.head_dim)

        flat = states.reshape(-1, self.head_dim).float()
        residual = flat - mse_recon
        residual_norm = torch.norm(residual, dim=-1)
        projected = residual @ self.S.T
        signs = torch.where(projected >= 0, 1, -1).to(torch.int8)

        b, h, s, _ = states.shape
        compressed["qjl_signs_packed"] = _pack_signs(signs.reshape(b, h, s, self.qjl_dim))
        compressed["residual_norm"] = residual_norm.to(torch.float16).reshape(b, h, s)
        return compressed

    @torch.no_grad()
    def reconstruct_mse(self, compressed: dict) -> torch.Tensor:
        return self.mse.reconstruct(compressed)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@dataclass
class TurboQuantCacheStats:
    compressed_bytes: int = 0
    fp16_visible_bytes: int = 0
    total_tokens: int = 0


class TurboQuantPaperCacheLayer(DynamicLayer):
    is_sliding = False

    def __init__(self, *, bits: int = 3, layer_idx: int = 0, seed: int = 42, head_dim: Optional[int] = None):
        super().__init__()
        self.bits = bits
        self.layer_idx = layer_idx
        self.seed = seed
        self.head_dim = head_dim
        self._key_compressor = None
        self._value_compressor = None
        self._compressed_keys = []
        self._compressed_values = []
        self._stats = TurboQuantCacheStats()

    def _get_compressors(self, device: torch.device, head_dim: int):
        if self._key_compressor is None:
            seed = self.seed + self.layer_idx * 1_000
            self._key_compressor = TurboQuantProdCompressor(head_dim, self.bits, seed=seed, device=str(device))
            self._value_compressor = TurboQuantMSECompressor(head_dim, self.bits, seed=seed + 137, device=str(device))
        return self._key_compressor, self._value_compressor

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: Optional[dict] = None):
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        head_dim = key_states.shape[-1] if self.head_dim is None else self.head_dim
        key_comp, value_comp = self._get_compressors(key_states.device, head_dim)

        compressed_k = key_comp.compress(key_states)
        compressed_v = value_comp.compress(value_states)
        self._compressed_keys.append(compressed_k)
        self._compressed_values.append(compressed_v)

        key_parts = [key_comp.reconstruct_mse(chunk).to(key_states.dtype) for chunk in self._compressed_keys]
        value_parts = [value_comp.reconstruct(chunk).to(value_states.dtype) for chunk in self._compressed_values]

        self.keys = torch.cat(key_parts, dim=-2)
        self.values = torch.cat(value_parts, dim=-2)

        self._update_stats()
        return self.keys, self.values

    def _update_stats(self) -> None:
        compressed_bytes = 0
        for ck in self._compressed_keys:
            compressed_bytes += ck["indices_packed"].numel()
            compressed_bytes += ck["vec_norms"].numel() * 2
            compressed_bytes += ck["qjl_signs_packed"].numel()
            compressed_bytes += ck["residual_norm"].numel() * 2
        for cv in self._compressed_values:
            compressed_bytes += cv["indices_packed"].numel()
            compressed_bytes += cv["vec_norms"].numel() * 2

        fp16_visible_bytes = 0 if self.keys.numel() == 0 else (self.keys.numel() + self.values.numel()) * 2
        self._stats = TurboQuantCacheStats(
            compressed_bytes=compressed_bytes,
            fp16_visible_bytes=fp16_visible_bytes,
            total_tokens=self.get_seq_length(),
        )

    def compute_attention_scores(
        self,
        query_states: torch.Tensor,
        *,
        num_key_value_groups: int,
        scaling: float,
    ) -> torch.Tensor:
        batch, num_heads, q_len, head_dim = query_states.shape
        num_kv_heads = num_heads // num_key_value_groups
        query_states_grouped = query_states.view(batch, num_kv_heads, num_key_value_groups, q_len, head_dim)

        parts = []
        for compressed_k in self._compressed_keys:
            k_mse = self._key_compressor.reconstruct_mse(compressed_k).float().unsqueeze(2)
            signs = _unpack_signs(compressed_k["qjl_signs_packed"], self._key_compressor.qjl_dim).float().unsqueeze(2)
            residual_norm = compressed_k["residual_norm"].float().unsqueeze(2).unsqueeze(-1)

            term1 = torch.matmul(query_states_grouped.float(), k_mse.transpose(-2, -1))
            q_projected = torch.matmul(query_states_grouped.float(), self._key_compressor.S.T)
            qjl_ip = torch.matmul(q_projected, signs.transpose(-2, -1))

            correction_scale = math.sqrt(math.pi / 2) / self._key_compressor.qjl_dim
            scores = term1 + correction_scale * qjl_ip * residual_norm.transpose(-2, -1)
            parts.append(scores.view(batch, num_heads, q_len, -1))

        return torch.cat(parts, dim=-1) * scaling

    @property
    def stats(self) -> TurboQuantCacheStats:
        return self._stats


class TurboQuantMSECacheLayer(DynamicLayer):
    is_sliding = False

    def __init__(self, *, bits: int = 4, layer_idx: int = 0, seed: int = 42, head_dim: Optional[int] = None):
        super().__init__()
        self.bits = bits
        self.layer_idx = layer_idx
        self.seed = seed
        self.head_dim = head_dim
        self._key_compressor = None
        self._value_compressor = None
        self._compressed_keys = []
        self._compressed_values = []
        self._stats = TurboQuantCacheStats()

    def _get_compressors(self, device: torch.device, head_dim: int):
        if self._key_compressor is None:
            seed = self.seed + self.layer_idx * 1_000
            self._key_compressor = TurboQuantMSECompressor(head_dim, self.bits, seed=seed, device=str(device))
            self._value_compressor = TurboQuantMSECompressor(head_dim, self.bits, seed=seed + 137, device=str(device))
        return self._key_compressor, self._value_compressor

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: Optional[dict] = None):
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        head_dim = key_states.shape[-1] if self.head_dim is None else self.head_dim
        key_comp, value_comp = self._get_compressors(key_states.device, head_dim)

        compressed_k = key_comp.compress(key_states)
        compressed_v = value_comp.compress(value_states)
        self._compressed_keys.append(compressed_k)
        self._compressed_values.append(compressed_v)

        key_parts = [key_comp.reconstruct(chunk).to(key_states.dtype) for chunk in self._compressed_keys]
        value_parts = [value_comp.reconstruct(chunk).to(value_states.dtype) for chunk in self._compressed_values]

        self.keys = torch.cat(key_parts, dim=-2)
        self.values = torch.cat(value_parts, dim=-2)

        self._update_stats()
        return self.keys, self.values

    def _update_stats(self) -> None:
        compressed_bytes = 0
        for ck in self._compressed_keys:
            compressed_bytes += ck["indices_packed"].numel()
            compressed_bytes += ck["vec_norms"].numel() * 2
        for cv in self._compressed_values:
            compressed_bytes += cv["indices_packed"].numel()
            compressed_bytes += cv["vec_norms"].numel() * 2

        fp16_visible_bytes = 0 if self.keys.numel() == 0 else (self.keys.numel() + self.values.numel()) * 2
        self._stats = TurboQuantCacheStats(
            compressed_bytes=compressed_bytes,
            fp16_visible_bytes=fp16_visible_bytes,
            total_tokens=self.get_seq_length(),
        )

    def compute_attention_scores(
        self,
        query_states: torch.Tensor,
        *,
        num_key_value_groups: int,
        scaling: float,
    ) -> torch.Tensor:
        batch, num_heads, q_len, head_dim = query_states.shape
        num_kv_heads = num_heads // num_key_value_groups
        query_states_grouped = query_states.view(batch, num_kv_heads, num_key_value_groups, q_len, head_dim)

        parts = []
        for compressed_k in self._compressed_keys:
            k_mse = self._key_compressor.reconstruct(compressed_k).float().unsqueeze(2)
            scores = torch.matmul(query_states_grouped.float(), k_mse.transpose(-2, -1))
            parts.append(scores.view(batch, num_heads, q_len, -1))

        return torch.cat(parts, dim=-1) * scaling

    @property
    def stats(self) -> TurboQuantCacheStats:
        return self._stats


class TurboQuantPaperCache(Cache):
    def __init__(self, *, n_layers: int, bits: int = 3, seed: int = 42):
        super().__init__(layers=[TurboQuantPaperCacheLayer(bits=bits, layer_idx=i, seed=seed) for i in range(n_layers)])

    @classmethod
    def from_model_config(cls, config, *, bits: int = 3, seed: int = 42):
        return cls(n_layers=config.get_text_config(decoder=True).num_hidden_layers, bits=bits, seed=seed)


class TurboQuantGenerationCache(Cache):
    def __init__(self, *, n_layers: int, bits: int = 4, seed: int = 42):
        super().__init__(layers=[TurboQuantMSECacheLayer(bits=bits, layer_idx=i, seed=seed) for i in range(n_layers)])

    @classmethod
    def from_model_config(cls, config, *, bits: int = 4, seed: int = 42):
        return cls(n_layers=config.get_text_config(decoder=True).num_hidden_layers, bits=bits, seed=seed)


def _patched_qwen2_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask,
    past_key_values: Cache | None = None,
    cache_position=None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = qwen_apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states,
            value_states,
            self.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )

    if isinstance(past_key_values, (TurboQuantPaperCache, TurboQuantGenerationCache)):
        layer = past_key_values.layers[self.layer_idx]
        attn_weights = layer.compute_attention_scores(
            query_states.float(),
            num_key_value_groups=self.num_key_value_groups,
            scaling=self.scaling,
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : attn_weights.shape[-1]]

        attn_weights = torch.clamp(attn_weights, min=-65_000.0, max=65_000.0)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        expanded_values = repeat_kv(value_states, self.num_key_value_groups)
        attn_output = torch.matmul(attn_weights, expanded_values)
        attn_output = attn_output.transpose(1, 2).contiguous()
    else:
        attention_interface = (
            eager_attention_forward
            if self.config._attn_implementation == "eager"
            else ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        )
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

    return self.o_proj(attn_output.reshape(*input_shape, -1).contiguous()), None


def _patched_llama_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.Tensor] = None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = llama_apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states,
            value_states,
            self.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )

    if isinstance(past_key_values, (TurboQuantPaperCache, TurboQuantGenerationCache)):
        layer = past_key_values.layers[self.layer_idx]
        
        # Handle Llama-specific scaling and GQA properties safely across HF versions
        scaling = getattr(self, "scaling", self.head_dim ** -0.5)
        num_heads = getattr(self, "num_heads", getattr(self.config, "num_attention_heads", 32))
        num_kv_heads = getattr(self, "num_key_value_heads", getattr(self.config, "num_key_value_heads", 32))
        kv_groups = getattr(self, "num_key_value_groups", num_heads // num_kv_heads)
        
        attn_weights = layer.compute_attention_scores(
            query_states.float(),
            num_key_value_groups=kv_groups,
            scaling=scaling,
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : attn_weights.shape[-1]]

        attn_weights = torch.clamp(attn_weights, min=-65_000.0, max=65_000.0)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        expanded_values = repeat_kv(value_states, kv_groups)
        attn_output = torch.matmul(attn_weights, expanded_values)
        attn_output = attn_output.transpose(1, 2).contiguous()
    else:
        # Standard fallback for the FP16 baseline pass
        import torch.nn.functional as F
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0 if not self.training else getattr(self, "attention_dropout", 0.0),
        )
        attn_output = attn_output.transpose(1, 2).contiguous()

    return self.o_proj(attn_output.reshape(*input_shape, -1).contiguous()), None


def patch_model_for_paper_generation(model) -> None:
    for module in model.modules():
        # Patch Qwen2
        if isinstance(module, Qwen2Attention):
            if not hasattr(module, "_turboquant_original_forward"):
                module._turboquant_original_forward = module.forward
            module.forward = MethodType(_patched_qwen2_forward, module)
        # Patch Llama
        elif isinstance(module, LlamaAttention):
            if not hasattr(module, "_turboquant_original_forward"):
                module._turboquant_original_forward = module.forward
            module.forward = MethodType(_patched_llama_forward, module)


def unpatch_model_for_paper_generation(model) -> None:
    for module in model.modules():
        if isinstance(module, (Qwen2Attention, LlamaAttention)) and hasattr(module, "_turboquant_original_forward"):
            module.forward = module._turboquant_original_forward

if __name__ == "__main__":
    MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
    print(f"Loading {MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda")
    
    if hasattr(model, "generation_config"):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    patch_model_for_paper_generation(model)
    print("TurboQuant Llama/Qwen2 patch applied.\n")

    # Few-Shot prompt to force a deterministic, exact-match answer
    prompt_text = (
        "Extract the requested information from the context exactly.\n\n"
        "Context: The developer built a remote agent controller called Control-PC-Terminal.\n"
        "Question: What is the name of the controller?\n"
        "Answer: Control-PC-Terminal\n\n"
        "Context: The team deployed the new SaaS platform known as AgentStackPro.\n"
        "Question: What is the name of the platform?\n"
        "Answer: AgentStackPro\n\n"
        "Context: The engineer, Shivam, released an observability stack today called LogPulse.\n"
        "Question: What is the name of the observability stack?\n"
        "Answer:"
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    attention_mask = torch.ones_like(inputs["input_ids"], device="cuda")
    
    # Store outputs to compare them automatically
    generated_outputs = {}

    print("--- Running Generations ---")
    
    # 1. FP16 Baseline
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        ans = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        generated_outputs["FP16 Baseline"] = ans
        print(f"FP16 Baseline: {ans}")

    # 2. TurboQuant 3-bit
    tq_cache = TurboQuantPaperCache.from_model_config(model.config, bits=3)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            past_key_values=tq_cache,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        ans = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        generated_outputs["TQ 3-bit"] = ans
        print(f"TQ 3-bit     : {ans}")

    # 3. TurboQuant 4-bit
    tq_cache_4 = TurboQuantPaperCache.from_model_config(model.config, bits=4)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            past_key_values=tq_cache_4,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        ans = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        generated_outputs["TQ 4-bit"] = ans
        print(f"TQ 4-bit     : {ans}")

    # 4. MSE-only 4-bit
    gen_cache_4 = TurboQuantGenerationCache.from_model_config(model.config, bits=4)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            past_key_values=gen_cache_4,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        ans = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        generated_outputs["MSE 4-bit"] = ans
        print(f"MSE 4-bit    : {ans}")

    # --- Automated Comparison ---
    print("\n========================================")
    print("          EVALUATION RESULTS            ")
    print("========================================")
    
    baseline = generated_outputs["FP16 Baseline"]
    
    for config, result in generated_outputs.items():
        if config == "FP16 Baseline":
            continue
            
        if result == baseline:
            status = "✅ EXACT MATCH"
        else:
            status = "❌ MISMATCH   "
            
        print(f"{config:<15} | {status} | Output: '{result}'")
        
    print("========================================\n")

    stats = tq_cache.layers[0].stats
    print(f"Cache Layer 0 Compression Check:")
    print(f"  FP16 memory required: {stats.fp16_visible_bytes / 1024:.2f} KB")
    print(f"  TQ 3-bit compressed : {stats.compressed_bytes / 1024:.2f} KB")