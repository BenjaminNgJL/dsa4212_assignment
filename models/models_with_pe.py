import jax.numpy as jnp
import flax.linen as nn
from flax.linen import attention as attn
import math

# ==========================================================
# Rotary Positional Encoding (RoPE) Helpers
# ==========================================================
def _rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rotary_pos_emb(q, k, sin, cos):
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    dim: int
    max_len: int

    def setup(self):
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.dim, 2) / self.dim))
        t = jnp.arange(self.max_len)
        freqs = jnp.einsum('i , j -> i j', t, inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        self.sin_cached = jnp.sin(emb)[None, :, None, :]
        self.cos_cached = jnp.cos(emb)[None, :, None, :]

    def __call__(self, seq_len):
        return self.sin_cached[:, :seq_len], self.cos_cached[:, :seq_len]

# ==========================================================
# Sinusoidal Positional Encoding
# ==========================================================
def sinusoidal_position_encoding(max_len, d_model):
    position = jnp.arange(max_len)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe = jnp.zeros((max_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe

# ==========================================================
# MLP Block
# ==========================================================
class MLP(nn.Module):
    d_model: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool):
        hidden_dim = int(self.d_model * self.mlp_ratio)
        x = nn.Dense(hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.d_model)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        return x

# ==========================================================
# Self-Attention Block (Rotary, Learned, or Sinusoidal)
# ==========================================================
class FlexibleSelfAttention(nn.Module):
    d_model: int
    n_heads: int
    dropout_rate: float = 0.1
    max_len: int = 512
    pos_encoding: str = "sinusoidal"  # options: rotary, learned, sinusoidal

    def setup(self):
        self.head_dim = self.d_model // self.n_heads
        if self.pos_encoding == "rotary":
            self.rotary_emb = RotaryEmbedding(dim=self.head_dim, max_len=self.max_len)
        ############# 

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):  # ← ADD THIS DECORATOR
        B, T, D = x.shape

        qkv = nn.Dense(3 * self.d_model, use_bias=False)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.n_heads, -1)
        k = k.reshape(B, T, self.n_heads, -1)
        v = v.reshape(B, T, self.n_heads, -1)

        if self.pos_encoding == "rotary":
            sin, cos = self.rotary_emb(T)
            q, k = apply_rotary_pos_emb(q, k, sin, cos)

        attn_weights = jnp.einsum('bthd,bThd->bhtT', q, k) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e10)

        attn_probs = nn.softmax(attn_weights, axis=-1)
        attn_probs = nn.Dropout(rate=self.dropout_rate)(attn_probs, deterministic=deterministic)

        out = jnp.einsum('bhtT,bThd->bthd', attn_probs, v)
        out = out.reshape(B, T, D)
        out = nn.Dense(self.d_model, use_bias=False)(out)
        out = nn.Dropout(rate=self.dropout_rate)(out, deterministic=deterministic)
        return out

# ==========================================================
# Decoder Block
# ==========================================================
class DecoderBlock(nn.Module):
    d_model: int
    n_heads: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.1
    max_len: int = 512
    pos_encoding: str = "rotary"

    @nn.compact
    def __call__(self, x, *, mask=None, deterministic=True):
        h = nn.LayerNorm()(x)
        h = FlexibleSelfAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
            max_len=self.max_len,
            pos_encoding=self.pos_encoding,
        )(h, mask=mask, deterministic=deterministic)
        x = x + h

        h = nn.LayerNorm()(x)
        h = MLP(
            d_model=self.d_model,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
        )(h, deterministic=deterministic)
        x = x + h
        return x

# ==========================================================
# Decoder-Only Transformer (Toggleable Positional Encoding)
# ==========================================================
class DecoderOnlyTransformer(nn.Module):
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    max_len: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.1
    pos_encoding: str = "sinusoidal"  # options: rotary, learned, sinusoidal

    def setup(self):
        self.tok_embed = nn.Embed(self.vocab_size, self.d_model)

        if self.pos_encoding == "learned":
            self.pos_embed = self.param(
                "pos_embed", nn.initializers.normal(stddev=0.02), (self.max_len, self.d_model)
            )
        elif self.pos_encoding == "sinusoidal":
            self.pos_embed = sinusoidal_position_encoding(self.max_len, self.d_model)

        self.blocks = [
            DecoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                max_len=self.max_len,
                pos_encoding=self.pos_encoding,
            ) for _ in range(self.n_layers)
        ]

        self.ln_final = nn.LayerNorm()
        self.to_logits = nn.Dense(self.vocab_size, use_bias=False)

    def __call__(self, idx, deterministic=True):
        B, T = idx.shape
        x = self.tok_embed(idx)

        if self.pos_encoding in ["learned", "sinusoidal"]:
            x = x + self.pos_embed[:T]

        causal_mask = attn.make_causal_mask(jnp.ones((B, T), dtype=bool))

        for block in self.blocks:
            x = block(x, mask=causal_mask, deterministic=deterministic)

        x = self.ln_final(x)
        logits = self.to_logits(x)
        return logits
