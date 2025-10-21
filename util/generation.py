# JITted token generator using jax.lax.scan - adds autoregressive sampling for the trained model
import jax
import jax.numpy as jnp

def sample_categorical(rng, logits, temperature=1.0):
    """Sample from categorical distribution given by logits with temperature."""
    # logits: (B, V)
    if temperature != 1.0:
        logits = logits / temperature
    return jax.random.categorical(rng, logits, axis=-1)

#@functools.partial(jax.jit, static_argnames=("model", "block_size", "temperature", "sample"))
def generate_tokens(model, params, rng, context, length, block_size=64, temperature=1.0, sample=True):
    """Generate `length` new tokens autoregressively starting from `context`.

    Args:
      model: Flax/linen model object with apply signature model.apply({'params': params}, tokens)
      params: model parameters pytree.
      rng: jax PRNGKey (e.g., jax.random.PRNGKey(0)).
      context: int32 array shape (B, S) where S <= block_size. If S < block_size, left-pad with zeros or use full context.
      length: number of tokens to generate (int).
      block_size: model context window (static).
      temperature: sampling temperature (static).
      sample: if True use sampling; if False use argmax (greedy).

    Returns:
      generated: int32 array shape (B, length) of generated token ids.
    """
    B = context.shape[0]
    S = context.shape[1]
    assert S <= block_size, "context length must be <= block_size"

    # initialize running context: if S < block_size, left-pad with zeros (or use a preferred pad token)
    if S < block_size:
        pad_len = block_size - S
        context = jnp.concatenate([jnp.zeros((B, pad_len), dtype=jnp.int32), context], axis=1)

    def _step(carry, _):
        rng, ctx = carry  # rng: PRNGKey, ctx: (B, block_size)
        # forward pass: get logits for all positions, take last position
        logits = model.apply({"params": params}, ctx)  # (B, block_size, V)
        last_logits = logits[:, -1, :]  # (B, V)
        rng, subkey = jax.random.split(rng)
        if sample:
            next_token = jax.random.categorical(subkey, last_logits / (temperature if temperature>0 else 1.0), axis=-1)
        else:
            next_token = jnp.argmax(last_logits, axis=-1)
        next_token = next_token.astype(jnp.int32)
        # append to context: drop first token, append new token at end
        next_token_col = next_token.reshape(B, 1)
        new_ctx = jnp.concatenate([ctx[:, 1:], next_token_col], axis=1)
        return (rng, new_ctx), next_token_col

    # run scan for `length` steps
    (rng_final, ctx_final), tokens = jax.lax.scan(_step, (rng, context), None, length=length)
    # tokens: shape (length, B, 1) -> reshape to (B, length)
    tokens = tokens.squeeze(-1).transpose(1, 0)
    return tokens

# example usage:
# generated = generate_tokens(model, params, rng, context, length=100, block_size=64, temperature=1.0, sample=True)
# print(generated)  # (B, 100) array of generated token ids