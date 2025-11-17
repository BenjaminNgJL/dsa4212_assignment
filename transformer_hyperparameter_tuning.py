import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import jax
import jax.numpy as jnp
import optax
import time

#!pip install optuna
import optuna
from optax import adamw
import pickle
import matplotlib.pyplot as plt

# local imports
import models.models_with_dropout as models
import util.generation as generation

# initialize the jax random key
key = jax.random.key(0)

# load the ./data/text8_train.txt and ./data/text8_test.txt files
with open("/content/drive/MyDrive/DSA4212_Project/data/text8_train.txt", "r") as f:
    train_text = f.read()
with open("/content/drive/MyDrive/DSA4212_Project/data/text8_test.txt", "r") as f:
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

def create_train_state(rng, vocab_size=27, d_model=64, n_layers=6, n_heads=8, max_len=128):
    # create a basic Transformer model
    model = models.DecoderOnlyTransformer(vocab_size, d_model, n_layers, n_heads, max_len)
    # create a dummy input for initialization
    dummy = jnp.zeros((1, min(16, max_len)), dtype=jnp.int32)
    # pass the dummy input to the model to initialize the parameters
    params = model.init({"params": rng}, dummy)["params"]
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

model, params = create_train_state(key, vocab_size, d_model, n_layers, n_heads, max_len)

# compute the number of parameters
def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"Number of parameters: {count_params(params):_}")

# sanity check: create a batch of data & run a forward pass
B, T = 4, 32
batch = jax.random.randint(
    key=key,
    shape=(B, T), minval=0, maxval=len(char_set))
logits = model.apply({"params": params}, batch)

print("batch shape:", batch.shape)  # (B, T)
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

def train_step_with_dropout(params, opt_state, x, y, model, tx, rng):
    def loss_fn(params):
        logits = model.apply({"params": params}, x, deterministic=False, rngs={"dropout": rng})
        loss, metrics = loss_and_metrics(logits, y)
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, metrics

train_step_with_dropout_jit = jax.jit(train_step_with_dropout, static_argnames=("model", "tx"))

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

# TRAINING SETTINGS — STANDARD CONFIGURATION
B, T = 128, 32
max_time = 180  # seconds per model

def train_with_config(learning_rate, batch_size, weight_decay, dropout_rate, max_time=max_time):
    """Train model with given config for up to max_time seconds."""
    model = models.DecoderOnlyTransformer(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        max_len=max_len, dropout_rate=dropout_rate
    )

    # Initialize
    rng = jax.random.key(0)
    dummy = jnp.zeros((1, 16), dtype=jnp.int32)
    params = model.init({"params": rng}, dummy, deterministic=True)["params"]

    # Optimizer
    tx = adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    opt_state = tx.init(params)

    # Track loss over time
    time_history = []
    loss_history = []
    loss_test_history = []
    time_test_history = []
    time_start = time.time()
    dropout_rng = jax.random.key(42)
    it = 0

    while True:
        elapsed = time.time() - time_start
        if elapsed > max_time:
            break

        x, y = get_batch(train_text_int, batch_size, T)

        dropout_rng, step_rng = jax.random.split(dropout_rng)
        params, opt_state, metrics = train_step_with_dropout_jit(
            params, opt_state, x, y, model, tx, step_rng
        )

        loss_history.append(float(metrics['loss']))
        time_history.append(elapsed)

        # Evaluate every 100 iterations
        if it % 100 == 0:
            B_test, T_test = 512, T
            test_input, test_target = get_batch(test_text_int, B_test, T_test)
            test_logits = model.apply({"params": params}, test_input, deterministic=True)
            test_loss, test_metrics = loss_and_metrics(test_logits, test_target)

            loss_test_history.append(float(test_loss))
            time_test_history.append(elapsed)

        it += 1

    return time_history, loss_history, loss_test_history, time_test_history

def sweep_hyperparameter(param_name, param_values, base_lr=0.001, base_bs=128,
                         base_wd=0.01, base_dr=0.1, max_time=max_time):
    """Sweep a single hyperparameter while keeping others fixed."""
    results = {}

    for val in param_values:
        print(f"Testing {param_name}={val}...")

        if param_name == 'learning_rate':
            times, losses, test_losses, test_times = train_with_config(val, base_bs, base_wd, base_dr, max_time)
        elif param_name == 'batch_size':
            times, losses, test_losses, test_times = train_with_config(base_lr, val, base_wd, base_dr, max_time)
        elif param_name == 'weight_decay':
            times, losses, test_losses, test_times = train_with_config(base_lr, base_bs, val, base_dr, max_time)
        elif param_name == 'dropout_rate':
            times, losses, test_losses, test_times = train_with_config(base_lr, base_bs, base_wd, val, max_time)

        results[val] = {
            'times': times,
            'losses': losses,
            'test_times': test_times,
            'test_losses': test_losses
        }

    return results

def plot_sweep(results, param_name):
    """Plot training curves for hyperparameter sweep."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Training loss
    for val, data in results.items():
        ax1.plot(data['times'], data['losses'], label=f'{param_name}={val}', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Training Loss')
    ax1.set_title(f'Training Loss — Effect of {param_name}')
    ax1.legend()
    ax1.grid(True)

    # Test loss
    for val, data in results.items():
        ax2.plot(data['test_times'], data['test_losses'], label=f'{param_name}={val}', linewidth=2, marker='o', markersize=3)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Test Loss')
    ax2.set_title(f'Test Loss — Effect of {param_name}')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def objective(trial):
    """Optuna objective function."""
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    bs = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    wd = trial.suggest_float('weight_decay', 0.0, 0.1)
    dr = trial.suggest_float('dropout_rate', 0.0, 0.3)

    _, _, test_losses, _ = train_with_config(lr, bs, wd, dr, max_time=180)
    return test_losses[-1] if test_losses else float('inf')  # Return final test loss

sweeps = {
    'learning_rate': [0.0001, 0.0003, 0.001, 0.003, 0.01],
    'batch_size': [32, 64, 128, 256],
    'weight_decay': [0.0, 0.01, 0.05, 0.1],
    'dropout_rate': [0.0, 0.1, 0.2, 0.3]
}

all_results = {}
for param_name, param_values in sweeps.items():
    print(f"Sweeping {param_name}...")
    results = sweep_hyperparameter(param_name, param_values, max_time=max_time)
    all_results[param_name] = results
    plot_sweep(results, param_name)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("\nBest hyperparameters:")
print(study.best_params)
print(f"Best loss: {study.best_value:.4f}")

def train_final_model(learning_rate, batch_size, weight_decay, dropout_rate, max_time=600):
    """Train final model with best hyperparameters."""
    print(f"Training Final Model")
    print(f"Best hyperparameters:")
    print(f"  • Learning rate: {learning_rate}")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Weight decay: {weight_decay}")
    print(f"  • Dropout rate: {dropout_rate}")
    print(f"  • Max training time: {max_time}s")

    # Create model with best dropout
    model = models.DecoderOnlyTransformer(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        max_len=max_len, dropout_rate=dropout_rate
    )

    # Initialize
    rng = jax.random.key(0)
    dummy = jnp.zeros((1, 16), dtype=jnp.int32)
    params = model.init({"params": rng}, dummy, deterministic=True)["params"]

    # Optimizer with best hyperparameters
    tx = adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    opt_state = tx.init(params)

    # Training tracking
    T = 32
    loss_history = []
    time_history = []
    loss_test_history = []
    time_test_history = []
    time_start = time.time()
    dropout_rng = jax.random.key(42)
    it = 0

    while True:
        elapsed = time.time() - time_start
        if elapsed > max_time:
            print(f"⏰ Time limit reached ({elapsed:.1f}s). Stopping training.\n")
            break

        x, y = get_batch(train_text_int, batch_size, T)

        dropout_rng, step_rng = jax.random.split(dropout_rng)
        params, opt_state, metrics = train_step_with_dropout_jit(
            params, opt_state, x, y, model, tx, step_rng
        )

        loss_history.append(float(metrics['loss']))
        time_history.append(elapsed)

        # Evaluate every 100 iterations or roughly every 30 seconds
        if it % 100 == 0 or (elapsed % 30 < 0.5 and it > 0):
            B_test, T_test = 1024, T
            test_input, test_target = get_batch(test_text_int, B_test, T_test)
            test_logits = model.apply({"params": params}, test_input, deterministic=True)
            test_loss, test_metrics = loss_and_metrics(test_logits, test_target)

            loss_test_history.append(float(test_loss))
            time_test_history.append(elapsed)

            print(f"iter {it:05d} | time {elapsed:.1f}s | "
                  f"loss(train/test): {metrics['loss']:.4f} / {test_loss:.4f} | "
                  f"acc(train/test): {100*metrics['acc']:.1f}% / {100*test_metrics['acc']:.1f}% | "
                  f"acc_last(train/test): {100*metrics['acc_last']:.1f}% / {100*test_metrics['acc_last']:.1f}%")

        it += 1

    print("\nTraining complete!")
    return model, params, loss_history, time_history, loss_test_history, time_test_history

# Train with best hyperparameters from Optuna
best_lr = study.best_params['learning_rate']
best_bs = study.best_params['batch_size']
best_wd = study.best_params['weight_decay']
best_dr = study.best_params['dropout_rate']

model, params, loss_history, time_history, loss_test_history, time_test_history = train_final_model(
    best_lr, best_bs, best_wd, best_dr, max_time=600
)

# plot the loss history
import matplotlib.pyplot as plt
plt.plot(time_history, loss_history, '-', label='train', color="blue")
plt.plot(time_test_history, loss_test_history, '-', label='test', lw=2, color="red")
plt.xlabel("Time (seconds)")
plt.ylabel("Loss")
plt.legend(loc='upper right')
plt.title("Training Loss History")
plt.grid()

B = 1
seed = 42
rng = jax.random.PRNGKey(seed)
prompt = "hello my fri"
# prompt_int = encode(prompt.lower())
prompt_int = jnp.array([ [char_to_int.get(c, len(char_set)) for c in prompt.lower()[:64]] ], dtype=jnp.int32)

gen_len = 1000
out_ids = generation.generate_tokens(model, params, rng, prompt_int, gen_len, block_size=64,
                          temperature=0.7, sample=True)
print('generated ids shape:', out_ids.shape)
print('generated text:')
generated_text = ''.join(int_to_char.get(int(x), '?') for x in list(out_ids[0]))
# concatenate with prompt
print(prompt + generated_text)
#print(''.join(int_to_char.get(int(x), '?') for x in list(out_ids[0])))