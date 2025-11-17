#load_ext autoreload
#autoreload 2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import jax
import jax.numpy as jnp
import optax
import time
import matplotlib.pyplot as plt

# local imports
import models.models_with_pe as models
import util.generation as generation
from models.models_with_pe import DecoderOnlyTransformer

# Data Loading and Vocabulary Setup
key = jax.random.key(0)

with open("./data/text8_train.txt", "r") as f:
    train_text = f.read()
with open("./data/text8_test.txt", "r") as f:
    test_text = f.read()

print(f"Length of training text: {len(train_text):_} characters")
print(f"Length of test text: {len(test_text):_} characters")

char_set = list("abcdefghijklmnopqrstuvwxyz ")
char_to_int = {ch: i for i, ch in enumerate(char_set)}
int_to_char = {i: ch for ch, i in char_to_int.items()}

def encode(s):
    return np.array([char_to_int[c] for c in s], dtype=np.uint8)

train_text_int = encode(train_text)
test_text_int = encode(test_text)

# quick sanity sample
T = 128
for _ in range(3):
    N = np.random.randint(0, len(train_text) - T)
    print(train_text[N:N + T], "\n")

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
    variables = model.init(rng, jnp.ones((1, max_len), dtype=jnp.int32))
    params = variables["params"]
    return model, params

vocab_size = len(char_set)
max_len = 128
pos_encoding = "learned"
n_layers = 2

# create 6 models with different head dimensions
model_1, params_1 = create_train_state(key, vocab_size, 128, 2, 4, max_len, pos_encoding)   # head_dim=32
model_2, params_2 = create_train_state(key, vocab_size, 256, 2, 4, max_len, pos_encoding)   # head_dim=64
model_3, params_3 = create_train_state(key, vocab_size, 256, 2, 8, max_len, pos_encoding)   # baseline, head_dim=32
model_4, params_4 = create_train_state(key, vocab_size, 256, 2, 16, max_len, pos_encoding)  # head_dim=16
model_5, params_5 = create_train_state(key, vocab_size, 512, 2, 8, max_len, pos_encoding)   # head_dim=64
model_6, params_6 = create_train_state(key, vocab_size, 512, 2, 16, max_len, pos_encoding)  # head_dim=32

# count parameters
def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

print(f"Baseline model params: {count_params(params_3):_}")

# quick forward sanity check
B, T = 4, 32
batch = jax.random.randint(key=key, shape=(B, T), minval=0, maxval=len(char_set))
logits = model_3.apply({"params": params_3}, batch)
print("batch:", batch.shape, "logits:", logits.shape)

@jax.jit
def loss_and_metrics(logits, targets):
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_targets = targets.reshape(-1)
    per_pos = optax.softmax_cross_entropy_with_integer_labels(flat_logits, flat_targets)
    loss = per_pos.mean()

    preds = jnp.argmax(logits, axis=-1)
    is_match = preds == targets
    acc_all = jnp.mean(is_match.astype(jnp.float32))
    acc_last = jnp.mean(is_match.astype(jnp.float32)[:, -1])
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

def get_batch(text_int, B, T):
    ix = np.random.randint(0, len(text_int) - T, size=B)
    x = np.stack([text_int[i:i + T] for i in ix])
    y = np.stack([text_int[i + 1:i + T + 1] for i in ix])
    return jnp.array(x, dtype=jnp.int32), jnp.array(y, dtype=jnp.int32)

learning_rate = 0.001
tx = optax.adam(learning_rate=learning_rate)
print(f"Optimizer: Adam (lr={learning_rate})")

def train_model(model, params, opt_state, train_text_int, test_text_int,
                B, T, tx, model_name, max_time=180):
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

        if it % 100 == 0 or (time.time() - time_start) % 30 < 1:
            B_test, T_test = 512, 32
            test_input, test_target = get_batch(test_text_int, B_test, T_test)
            test_logits = model.apply({"params": params}, test_input)
            test_loss, test_metrics = loss_and_metrics(test_logits, test_target)
            loss_test_history.append(test_loss)
            time_test_history.append(elapsed)

            print(f"iter {it:05d} | {elapsed:6.1f}s | "
                  f"loss(train/test): {metrics['loss']:.4f}/{test_loss:.4f} | "
                  f"acc(train/test): {100*metrics['acc']:.1f}%/{100*test_metrics['acc']:.1f}% | "
                  f"acc_last(train/test): {100*metrics['acc_last']:.1f}%/{100*test_metrics['acc_last']:.1f}%")
        it += 1

    return params, opt_state, loss_history, time_history, loss_test_history, time_test_history

B, T = 128, 32
max_time = 180
results = []

models_to_train = [
    ("128d_4h", model_1, params_1),
    ("256d_4h", model_2, params_2),
    ("256d_8h_baseline", model_3, params_3),
    ("256d_16h", model_4, params_4),
    ("512d_8h", model_5, params_5),
    ("512d_16h", model_6, params_6),
]

for name, model, params in models_to_train:
    opt_state = tx.init(params)
    params, opt_state, loss_hist, time_hist, loss_test_hist, time_test_hist = train_model(
        model, params, opt_state, train_text_int, test_text_int,
        B, T, tx, name, max_time=max_time
    )
    results.append({
        "name": name,
        "model": model,
        "params": params,
        "loss_history": loss_hist,
        "time_history": time_hist,
        "test_loss_history": loss_test_hist,
        "test_time_history": time_test_hist,
    })

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(8, 5))
plt.title("Loss Over Time — Head Dimension Comparison")
plt.xlabel("Time (seconds)")
plt.ylabel("Loss")
plt.grid(True)
for r in results:
    plt.plot(r["time_history"], r["loss_history"], label=r["name"], alpha=0.8)
plt.legend()
plt.savefig("plots/head_dim_loss_over_time.png", bbox_inches="tight")
plt.close()
print("Saved plot: plots/head_dim_loss_over_time.png")

print("\n")
print("Starting Text Generation Phase")
print("\n")

B = 1
seed = 42
rng = jax.random.PRNGKey(seed)
prompt = "hello my fri"
prompt_int = jnp.array(
    [[char_to_int.get(c, len(char_set)) for c in prompt.lower()[:64]]],
    dtype=jnp.int32
)
gen_len = 300

os.makedirs("generations", exist_ok=True)
master_path = "generations/all_generations.txt"

with open(master_path, "w") as f:
    f.write("Transformer Head-Dimension Comparison — Text Generation Results\n")
    f.write("=" * 70 + "\n\n")

def generate_text_for_model(model, params, model_name):
    out_ids = generation.generate_tokens(
        model, params, rng, prompt_int,
        gen_len, block_size=64, temperature=0.7, sample=True
    )
    generated_text = ''.join(int_to_char.get(int(x), '?') for x in list(out_ids[0]))
    with open(master_path, "a") as f:
        f.write(f"\nMODEL: {model_name}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Generated Text:\n{prompt + generated_text}\n")
        f.write("=" * 70 + "\n")
    print(f"Generated text for {model_name}")
for r in results:
    generate_text_for_model(r["model"], r["params"], r["name"])

print(f"\nAll generations saved to: {master_path}")