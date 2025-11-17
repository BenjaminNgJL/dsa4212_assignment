#load_ext autoreload
#autoreload 2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import jax
import jax.numpy as jnp
import optax
import time

# local imports
import models.models_with_pe as models
import util.generation as generation

# initialize the jax random key
key = jax.random.key(0)

# load the ./data/text8_train.txt and ./data/text8_test.txt files
with open("./data/text8_train.txt", "r") as f:
    train_text = f.read()
with open("./data/text8_test.txt", "r") as f:
    test_text = f.read()

# print the length of the training text and test text
print(f"Length of training text: {len(train_text):_} characters")
print(f"Length of test text: {len(test_text):_} characters")

# Build vocabulary (lowercase + space + a few punctuations)
char_set = list("abcdefghijklmnopqrstuvwxyz ")
char_to_int = {ch:i for i,ch in enumerate(char_set)}
int_to_char = {i:ch for ch,i in char_to_int.items()}

def encode(s):
    """Encode string to array of integers"""
    ids = [char_to_int[c] for c in s]
    return np.array(ids, dtype=np.uint8)

# encode the text
train_text_int = encode(train_text)
test_text_int = encode(test_text)

# sanity check: display a few random characters from the training text
T = 128
for _ in range(5):
    # choose random position in text
    N = np.random.randint(low=0, high=len(train_text)-T)
    print(train_text[N:N+T])
    print()

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from models.models_with_pe import DecoderOnlyTransformer

def create_train_state(rng, vocab_size, d_model, n_layers, n_heads, max_len,
                       lr=1e-4, pos_encoding="learned"):
    """Initializes model parameters and optimizer state."""
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_len=max_len,
        pos_encoding=pos_encoding
    )

    # Initialize model parameters
    variables = model.init(rng, jnp.ones((1, max_len), dtype=jnp.int32))
    params = variables["params"]

    return model, params

# vocab size
vocab_size= len(char_set)

# internal model dimensions
d_model=256

# number of attention heads
n_heads=8

# number of Transformer layers
n_layers=2

# maximum sequence length
max_len=128

# positional encoding type
# pos_encoding = "learned"  # (sinusoidal, learned, rotary)

model_learned, params_learned = create_train_state(key, vocab_size, d_model, n_layers, n_heads, max_len, pos_encoding="learned")
model_sinusoidal, params_sinusoidal = create_train_state(key, vocab_size, d_model, n_layers, n_heads, max_len, pos_encoding="sinusoidal")
model_rotary, params_rotary = create_train_state(key, vocab_size, d_model, n_layers, n_heads, max_len, pos_encoding="sinusoidal")

# compute the number of parameters
def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

print(f"Number of parameters for learned model: {count_params(params_learned):_}")
print(f"Number of parameters for sinusoidal model: {count_params(params_sinusoidal):_}")
print(f"Number of parameters for rotary model: {count_params(params_rotary):_}")

# sanity check: create a batch of data & run a forward pass
B, T = 4, 32
batch = jax.random.randint(
    key=key,
    shape=(B, T), minval=0, maxval=len(char_set))
logits_learned = model_learned.apply({"params": params_learned}, batch)
logits_sinusoidal = model_sinusoidal.apply({"params": params_sinusoidal}, batch)
logits_rotary = model_rotary.apply({"params": params_rotary}, batch)

print("batch shape:", batch.shape)  # (B, T)
print("logits learned shape:", logits_learned.shape)  # (B, T, vocab_size)
print("logits sinusoidal shape:", logits_sinusoidal.shape)  # (B, T, vocab_size)
print("logits rotary shape:", logits_rotary.shape)  # (B, T, vocab_size)

@jax.jit
def loss_and_metrics(logits, targets):
    """Compute cross-entropy loss and accuracy.

    Assumes `targets` contains only valid integer class ids in [0, V-1] (no -1 ignore tokens).

    Args:
      logits: (B, T, V) float array of unnormalized scores.
      targets: (B, T) integer array with ground-truth class ids.

    Returns:
      loss: scalar average cross-entropy over all positions.
      metrics: dict with keys "loss" and "acc" (both scalars).
    """
    # Flatten batch/time dims so optax works on shape (N, V) and (N,)
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_targets = targets.reshape(-1)

    # Per-position cross-entropy, then mean over all positions
    per_pos = optax.softmax_cross_entropy_with_integer_labels(flat_logits, flat_targets)
    loss = per_pos.mean()

    # prediction over all positions
    preds = jnp.argmax(logits, axis=-1)  # (B, T)
    
    # compute accuracy over only the last position
    is_match = preds == targets
    
    # Accuracy over all positions
    acc_all = jnp.mean(is_match.astype(jnp.float32))
    
    # Accuracy over only last position
    acc_last = jnp.mean(is_match.astype(jnp.float32)[:,-1])

    return loss, {"loss": loss, "acc": acc_all, "acc_last": acc_last}

def train_step(params, opt_state, x, y, tx, model):
    def loss_fn(params):
        logits = model.apply({"params": params}, x)
        loss, metrics = loss_and_metrics(logits, y)
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, metrics

train_step = jax.jit(train_step, static_argnames=("tx", "model"))

# create a batch from the training data
def get_batch(text_int, B, T):
    """Create a random batch of data from text_int.

    Args:
      text_int: 1D array of token ids.
      B: batch size (number of sequences).
      T: sequence length (number of tokens per sequence).

    Returns:
      x: (B, T) int array input tokens.
      y: (B, T) int array target tokens.
    """
    # choose random starting indices for each sequence in the batch
    ix = np.random.randint(0, len(text_int) - T, size=B)
    # inputs are text from i to i+T
    x = np.stack([text_int[i:i+T] for i in ix])
    # targets are text from i+1 to i+T+1
    y = np.stack([text_int[i+1:i+T+1] for i in ix])
    return jnp.array(x, dtype=jnp.int32), jnp.array(y, dtype=jnp.int32)

# define optax optimizer
learning_rate = 0.001
# Create Adam optimizer (Optax)
tx = optax.adam(learning_rate=learning_rate)
# Initialize optimizer state for current params
opt_state_learned = tx.init(params_learned)
opt_state_sinusoidal = tx.init(params_sinusoidal)
opt_state_rotary = tx.init(params_rotary)
print(f"Initialized optimizer: Adam lr={learning_rate}")

def train_model(model, params, opt_state, train_text_int, test_text_int,
                B, T, tx, model_name, max_time=180):
    """Train a given model for up to `max_time` seconds and return histories."""
    print(f"\nStarting training for {model_name} positional encoding model...")
    time_start = time.time()

    loss_history, time_history = [], []
    loss_test_history, time_test_history = [], []

    it = 0
    while True:
        # check elapsed time
        elapsed = time.time() - time_start
        if elapsed > max_time:
            print(f"Time limit reached ({elapsed:.1f}s). Stopping training.\n")
            break

        # get random training batch
        x, y = get_batch(train_text_int, B, T)
        params, opt_state, metrics = train_step(params, opt_state, x, y, tx, model)

        # record metrics
        loss = metrics["loss"]
        acc = metrics["acc"]
        acc_last = metrics["acc_last"]

        loss_history.append(loss)
        time_history.append(elapsed)

        # evaluate periodically (every 30 seconds for example)
        if it % 100 == 0 or (time.time() - time_start) % 30 < 1:
            B_test, T_test = 1024, 32
            test_input, test_target = get_batch(test_text_int, B_test, T_test)
            test_logits = model.apply({"params": params}, test_input)
            test_loss, test_metrics = loss_and_metrics(test_logits, test_target)
            test_acc = test_metrics["acc"]
            test_acc_last = test_metrics["acc_last"]

            loss_test_history.append(test_loss)
            time_test_history.append(time.time() - time_start)

            print(f"iteration {it:_}  time: {elapsed:.1f}s")
            print(f"    loss(train :: test): {loss:.4f} :: {test_loss:.4f}")
            print(f"    acc (train :: test): {100*acc:.1f}% :: {100*test_acc:.1f}%")
            print(f"    acc_last (train :: test): {100*acc_last:.1f}% :: {100*test_acc_last:.1f}%\n")

        it += 1

    return params, opt_state, loss_history, time_history, loss_test_history, time_test_history


# niter = 10000  # increase iterations for learning progress
B, T = 128, 32

# Train learned model
params_learned, opt_state_learned, loss_history_learned, time_history_learned, \
    loss_test_history_learned, time_test_history_learned = train_model(
        model_learned, params_learned, opt_state_learned,
        train_text_int, test_text_int, B, T, tx, "learned", max_time=180
    )


# Train sinusoidal model
params_sinusoidal, opt_state_sinusoidal, loss_history_sinusoidal, time_history_sinusoidal, \
    loss_test_history_sinusoidal, time_test_history_sinusoidal = train_model(
        model_sinusoidal, params_sinusoidal, opt_state_sinusoidal,
        train_text_int, test_text_int, B, T, tx, "sinusoidal", max_time=180
    )

# Train rotary model
params_rotary, opt_state_rotary, loss_history_rotary, time_history_rotary, \
    loss_test_history_rotary, time_test_history_rotary = train_model(
        model_rotary, params_rotary, opt_state_rotary,
        train_text_int, test_text_int, B, T, tx, "rotary", max_time=180
    )

# Plot training loss for comparison
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
plt.plot(time_history_learned, loss_history_learned, '-', label='learned', color="blue")
plt.plot(time_history_sinusoidal, loss_history_sinusoidal, '-', label='sinusoidal', color="green")
plt.plot(time_history_rotary, loss_history_rotary, '-', label='rotary', color="red")
plt.xlabel("Time (seconds)")
plt.ylabel("Loss")
plt.legend(loc='upper right')
plt.title("Training Loss History")
plt.grid(True)
plt.show()

B = 1
seed = 42
rng = jax.random.PRNGKey(seed)
prompt = "hello my fri"
# prompt_int = encode(prompt.lower())
prompt_int = jnp.array([ [char_to_int.get(c, len(char_set)) for c in prompt.lower()[:64]] ], dtype=jnp.int32)

gen_len = 1000
out_ids = generation.generate_tokens(model_learned, params_learned, rng, prompt_int, gen_len, block_size=64, 
                          temperature=0.7, sample=True)
print('generated ids shape:', out_ids.shape)
print('generated text for learned model:')
generated_text = ''.join(int_to_char.get(int(x), '?') for x in list(out_ids[0]))
# concatenate with prompt
print(prompt + generated_text)
#print(''.join(int_to_char.get(int(x), '?') for x in list(out_ids[0])))

gen_len = 1000
out_ids = generation.generate_tokens(model_sinusoidal, params_sinusoidal, rng, prompt_int, gen_len, block_size=64, 
                          temperature=0.7, sample=True)
print('generated ids shape:', out_ids.shape)
print('generated text for sinusoidal model:')
generated_text = ''.join(int_to_char.get(int(x), '?') for x in list(out_ids[0]))
# concatenate with prompt
print(prompt + generated_text)
#print(''.join(int_to_char.get(int(x), '?') for x in list(out_ids[0])))

gen_len = 1000
out_ids = generation.generate_tokens(model_rotary, params_rotary, rng, prompt_int, gen_len, block_size=64, 
                          temperature=0.7, sample=True)
print('generated ids shape:', out_ids.shape)
print('generated text for rotary model:')
generated_text = ''.join(int_to_char.get(int(x), '?') for x in list(out_ids[0]))
# concatenate with prompt
print(prompt + generated_text)
#print(''.join(int_to_char.get(int(x), '?') for x in list(out_ids[0])))