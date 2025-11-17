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
pos_encoding = "learned"  # (sinusoidal, learned, rotary)

# baseline
model_baseline, params_baseline = create_train_state(key, vocab_size, d_model, n_layers, n_heads, max_len, pos_encoding=pos_encoding)

# seq length variants
model_32, params_32 = create_train_state(key, vocab_size, d_model, n_layers, n_heads, 32, pos_encoding=pos_encoding)
model_64, params_64 = create_train_state(key, vocab_size, d_model, n_layers, n_heads, 64, pos_encoding=pos_encoding)
model_256, params_256 = create_train_state(key, vocab_size, d_model, n_layers, n_heads, 256, pos_encoding=pos_encoding)
model_512, params_512 = create_train_state(key, vocab_size, d_model, n_layers, n_heads, 512, pos_encoding=pos_encoding)

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
            print(f"Time limit reached ({elapsed:.1f}s). Stopping training.\n")
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

B, T = 128, 32
max_time = 180  # seconds per model
results = {}

# train each model in the sequence length variants
model_variants = [
    ("Baseline (learned PE, T=128)", model_baseline, params_baseline, 128),
    ("SeqLen 32", model_32, params_32, 32),
    ("SeqLen 64", model_64, params_64, 64),
    ("SeqLen 256", model_256, params_256, 256),
    ("SeqLen 512", model_512, params_512, 512),
]

feature = "sequence_length_variants"
results[feature] = []
for name, model, params, seq_len in model_variants:
    # initialize optimizer state
    opt_state = tx.init(params)

    # train the model
    trained_params, trained_opt_state, loss_history, time_history, loss_test_history, time_test_history = train_model(
        model, params, opt_state,
        train_text_int, test_text_int,
        B, seq_len, tx,
        model_name=name,
        max_time=max_time
    )

    # store results
    results[feature].append({
        "name": name,
        "model": model,
        "params": trained_params,
        "loss_history": loss_history,
        "time_history": time_history,
        "loss_test_history": loss_test_history,
        "time_test_history": time_test_history
    })

# plot training curves in one graph
import matplotlib.pyplot as plt
for feature, res_list in results.items():
    plt.figure(figsize=(10,6))
    for r in res_list:
        plt.plot(r["time_history"], r["loss_history"], label=r["name"])
    plt.xlabel("Time (s)")
    plt.ylabel("Training Loss")
    plt.title(f"Training Loss Curves — {feature.replace('_', ' ').title()}")
    plt.legend()
    plt.grid()
    plt.show()


# TEXT GENERATION FOR EACH MODEL GROUP
print("\n")
print("🧠 Starting Text Generation Phase")
print("\n")

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
    f.write(f"Transformer Learned Positional Encoding — Text Generation Results\n")
    f.write(f"{'='*70}\n\n")

def generate_text_for_model(model, params, model_name, feature):
    # block size should be half of max len used in training
    block_size = model.max_len // 2
    """Generate text using the given trained model and parameters."""
    out_ids = generation.generate_tokens(
        model, params, rng, prompt_int,
        gen_len, block_size=block_size, temperature=0.7, sample=True
    )
    generated_text = ''.join(int_to_char.get(int(x), '?') for x in list(out_ids[0]))

    # append to master file
    with open(master_path, "a") as f:
        f.write(f"\n\nFEATURE: {feature.upper()} | MODEL: {model_name}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Generated Text:\n")
        f.write(prompt + generated_text)
        f.write("\n" + "="*70 + "\n")

    # console preview
    print(f"\nGenerated text for {feature} → {model_name}:")
    print(prompt + generated_text[:300], "...\n")
    print(f"Appended to: {master_path}")

# iterate over all groups and models
for feature, res_list in results.items():
    print(f"\n")
    print(f"Text Generation — {feature.upper()}")
    print(f"\n")
    for r in res_list:
        model_name = r["name"]
        model = r["model"]
        params = r["params"]

        generate_text_for_model(model, params, model_name, feature)