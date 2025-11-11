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
    return np.array(ids, dtype=np.uint8)  # use np.uint8 to save space

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
pos_encoding = "learned"  # (sinusoidal, learned, rotary)

# baseline
model_baseline, params_baseline = create_train_state(key, vocab_size, d_model, n_layers, n_heads, max_len, pos_encoding=pos_encoding)

# vocab scaling
model_vocab_x2, params_vocab_x2 = create_train_state(key, vocab_size*2, d_model, n_layers, n_heads, max_len, pos_encoding=pos_encoding)
model_vocab_x4, params_vocab_x4 = create_train_state(key, vocab_size*4, d_model, n_layers, n_heads, max_len, pos_encoding=pos_encoding)
model_vocab_x8, params_vocab_x8 = create_train_state(key, vocab_size*8, d_model, n_layers, n_heads, max_len, pos_encoding=pos_encoding)
model_vocab_x16, params_vocab_x16 = create_train_state(key, vocab_size*16, d_model, n_layers, n_heads, max_len, pos_encoding=pos_encoding)

# d_model scaling
model_dmodel_64, params_dmodel_64 = create_train_state(key, vocab_size, 64, n_layers, n_heads, max_len, pos_encoding=pos_encoding)
model_dmodel_128, params_dmodel_128 = create_train_state(key, vocab_size, 128, n_layers, n_heads, max_len, pos_encoding=pos_encoding)
model_dmodel_512, params_dmodel_512 = create_train_state(key, vocab_size, 512, n_layers, n_heads, max_len, pos_encoding=pos_encoding)
model_dmodel_1024, params_dmodel_1024 = create_train_state(key, vocab_size, 1024, n_layers, n_heads, max_len, pos_encoding=pos_encoding)

# n_heads scaling
model_nheads_2, params_nheads_2 = create_train_state(key, vocab_size, d_model, n_layers, 2, max_len, pos_encoding=pos_encoding)
model_nheads_4, params_nheads_4 = create_train_state(key, vocab_size, d_model, n_layers, 4, max_len, pos_encoding=pos_encoding)
model_nheads_16, params_nheads_16 = create_train_state(key, vocab_size, d_model, n_layers, 16, max_len, pos_encoding=pos_encoding)
model_nheads_32, params_nheads_32 = create_train_state(key, vocab_size, d_model, n_layers, 32, max_len, pos_encoding=pos_encoding)

# n_layers scaling
model_nlayers_1, params_nlayers_1 = create_train_state(key, vocab_size, d_model, 1, n_heads, max_len, pos_encoding=pos_encoding)
model_nlayers_4, params_nlayers_4 = create_train_state(key, vocab_size, d_model, 4, n_heads, max_len, pos_encoding=pos_encoding)
model_nlayers_8, params_nlayers_8 = create_train_state(key, vocab_size, d_model, 8, n_heads, max_len, pos_encoding=pos_encoding)
model_nlayers_16, params_nlayers_16 = create_train_state(key, vocab_size, d_model, 16, n_heads, max_len, pos_encoding=pos_encoding)

# compute the number of parameters
def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

# compute and display number of parameters for baseline model only
print(f"Number of parameters (baseline learned PE model): {count_params(params_baseline):_}")

# quick forward pass sanity check
B, T = 4, 32
batch = jax.random.randint(
    key=key,
    shape=(B, T), minval=0, maxval=len(char_set))
logits = model_baseline.apply({"params": params_baseline}, batch)
print("batch shape:", batch.shape)
print("logits shape:", logits.shape)  # (B, T, vocab_size)

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

learning_rate = 0.001
tx = optax.adam(learning_rate=learning_rate)
print(f"Initialized optimizer: Adam (lr={learning_rate})")

def train_model(model, params, opt_state, train_text_int, test_text_int,
                B, T, tx, model_name, max_time=180):
    """Train a given model for up to `max_time` seconds."""
    print(f"\nStarting training for {model_name} (max {max_time}s)...")
    time_start = time.time()
    loss_history, time_history = [], []
    loss_test_history, time_test_history = [], []
    it = 0

    while True:
        elapsed = time.time() - time_start
        if elapsed > max_time:
            print(f"⏰ Time limit reached ({elapsed:.1f}s). Stopping training.\n")
            break

        x, y = get_batch(train_text_int, B, T)
        params, opt_state, metrics = train_step(params, opt_state, x, y, tx, model)

        loss_history.append(metrics["loss"])
        time_history.append(elapsed)

        # evaluate every ~30s or 100 iters
        if it % 100 == 0 or (time.time() - time_start) % 30 < 1:
            B_test, T_test = 512, 32
            test_input, test_target = get_batch(test_text_int, B_test, T_test)
            test_logits = model.apply({"params": params}, test_input)
            test_loss, test_metrics = loss_and_metrics(test_logits, test_target)

            loss_test_history.append(test_loss)
            time_test_history.append(elapsed)

            print(f"iter {it:05d} | time {elapsed:.1f}s | "
                f"loss(train/test): {metrics['loss']:.4f} / {test_loss:.4f} | "
                f"acc(train/test): {100*metrics['acc']:.1f}% / {100*test_metrics['acc']:.1f}% | "
                f"acc_last(train/test): {100*metrics['acc_last']:.1f}% / {100*test_metrics['acc_last']:.1f}%")

        it += 1

    return params, opt_state, loss_history, time_history, loss_test_history, time_test_history

# ============================
# TRAINING SETTINGS
# ============================
B, T = 128, 32
max_time = 180  # seconds
results = {}

# ---------- VOCAB SIZE ----------
vocab_models = [
    ("vocab_x1", model_baseline, params_baseline),
    ("vocab_x2", model_vocab_x2, params_vocab_x2),
    ("vocab_x4", model_vocab_x4, params_vocab_x4),
    ("vocab_x8", model_vocab_x8, params_vocab_x8),
    ("vocab_x16", model_vocab_x16, params_vocab_x16),
]
results["vocab"] = []
for name, model, params in vocab_models:
    opt_state = tx.init(params)
    params, opt_state, loss_hist, time_hist, loss_test_hist, time_test_hist = train_model(
        model, params, opt_state, train_text_int, test_text_int,
        B, T, tx, name, max_time=max_time
    )
    results["vocab"].append({
        "name": name,
        "model": model,
        "params": params,
        "loss_history": loss_hist,
        "time_history": time_hist,
        "test_loss_history": loss_test_hist,
        "test_time_history": time_test_hist,
    })

# ---------- D_MODEL ----------
dmodel_models = [
    ("dmodel_64", model_dmodel_64, params_dmodel_64),
    ("dmodel_128", model_dmodel_128, params_dmodel_128),
    ("dmodel_256", model_baseline, params_baseline),
    ("dmodel_512", model_dmodel_512, params_dmodel_512),
    ("dmodel_1024", model_dmodel_1024, params_dmodel_1024),
]
results["d_model"] = []
for name, model, params in dmodel_models:
    opt_state = tx.init(params)
    params, opt_state, loss_hist, time_hist, loss_test_hist, time_test_hist = train_model(
        model, params, opt_state, train_text_int, test_text_int,
        B, T, tx, name, max_time=max_time
    )
    results["d_model"].append({
        "name": name,
        "model": model,
        "params": params,
        "loss_history": loss_hist,
        "time_history": time_hist,
        "test_loss_history": loss_test_hist,
        "test_time_history": time_test_hist,
    })

# ---------- N_HEADS ----------
nheads_models = [
    ("nheads_2", model_nheads_2, params_nheads_2),
    ("nheads_4", model_nheads_4, params_nheads_4),
    ("nheads_8", model_baseline, params_baseline),
    ("nheads_16", model_nheads_16, params_nheads_16),
    ("nheads_32", model_nheads_32, params_nheads_32),
]
results["n_heads"] = []
for name, model, params in nheads_models:
    opt_state = tx.init(params)
    params, opt_state, loss_hist, time_hist, loss_test_hist, time_test_hist = train_model(
        model, params, opt_state, train_text_int, test_text_int,
        B, T, tx, name, max_time=max_time
    )
    results["n_heads"].append({
        "name": name,
        "model": model,
        "params": params,
        "loss_history": loss_hist,
        "time_history": time_hist,
        "test_loss_history": loss_test_hist,
        "test_time_history": time_test_hist,
    })

# ---------- N_LAYERS ----------
nlayers_models = [
    ("nlayers_1", model_nlayers_1, params_nlayers_1),
    ("nlayers_2", model_baseline, params_baseline),
    ("nlayers_4", model_nlayers_4, params_nlayers_4),
    ("nlayers_8", model_nlayers_8, params_nlayers_8),
    ("nlayers_16", model_nlayers_16, params_nlayers_16),
]
results["n_layers"] = []
for name, model, params in nlayers_models:
    opt_state = tx.init(params)
    params, opt_state, loss_hist, time_hist, loss_test_hist, time_test_hist = train_model(
        model, params, opt_state, train_text_int, test_text_int,
        B, T, tx, name, max_time=max_time
    )
    results["n_layers"].append({
        "name": name,
        "model": model,
        "params": params,
        "loss_history": loss_hist,
        "time_history": time_hist,
        "test_loss_history": loss_test_hist,
        "test_time_history": time_test_hist,
    })

# ============================================================
# Plot training loss for comparison
# ============================================================
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

for feature, res_list in results.items():
    plt.figure(figsize=(8, 5))
    plt.title(f"Loss Over Time — {feature.replace('_', ' ').title()}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Loss")
    plt.grid(True)

    for r in res_list:
        plt.plot(r["time_history"], r["loss_history"], label=r["name"], alpha=0.8)

    plt.legend()
    file_path = f"plots/{feature.lower()}_loss_over_time.png"
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    print(f"📈 Saved plot: {file_path}")

# ============================================================
# STEP 5: TEXT GENERATION FOR EACH MODEL GROUP
# ============================================================
print("\n==============================")
print("🧠 Starting Text Generation Phase")
print("==============================")

B = 1
seed = 42
rng = jax.random.PRNGKey(seed)
prompt = "hello my fri"
prompt_int = jnp.array(
    [[char_to_int.get(c, len(char_set)) for c in prompt.lower()[:64]]],
    dtype=jnp.int32
)
gen_len = 300  # shorter for readability

# prepare a single output file
os.makedirs("generations", exist_ok=True)
master_path = "generations/all_generations.txt"

# clear old contents if file already exists
with open(master_path, "w") as f:
    f.write(f"🧠 Transformer Learned Positional Encoding — Text Generation Results\n")
    f.write(f"{'='*70}\n\n")

def generate_text_for_model(model, params, model_name, feature):
    """Generate text using the given trained model and parameters."""
    out_ids = generation.generate_tokens(
        model, params, rng, prompt_int,
        gen_len, block_size=64, temperature=0.7, sample=True
    )
    generated_text = ''.join(int_to_char.get(int(x), '?') for x in list(out_ids[0]))

    # append to master file
    with open(master_path, "a") as f:
        f.write(f"\n\n=== FEATURE: {feature.upper()} | MODEL: {model_name} ===\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Generated Text:\n")
        f.write(prompt + generated_text)
        f.write("\n" + "="*70 + "\n")

    # console preview
    print(f"\n📜 Generated text for {feature} → {model_name}:")
    print(prompt + generated_text[:300], "...\n")
    print(f"💾 Appended to: {master_path}")

# iterate over all groups and models
for feature, res_list in results.items():
    print(f"\n==============================")
    print(f"🔹 Text Generation — {feature.upper()}")
    print("==============================")
    for r in res_list:
        model_name = r["name"]
        model = r["model"]
        params = r["params"]

        generate_text_for_model(model, params, model_name, feature)