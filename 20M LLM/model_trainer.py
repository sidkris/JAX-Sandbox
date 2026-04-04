import jax
import jax.numpy as jnp
import flax.nnx as nnx 
import grain.python as grain 
import tiktoken
import optax
from data_loader import load_data
from mini_llm import MiniLLM 

maxlen = 128
tokenizer = tiktoken.get_encoding("gpt2")
num_epochs = 3
text_dl, batches_per_epoch = load_data()
total_steps = num_epochs * batches_per_epoch

model = MiniLLM()

def loss_fn(model, batch):
    inputs, targets = batch
    logits = model(inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    return loss, logits


warmup_steps = max(1, total_steps // 10)
print(f"Total Training Steps : {total_steps:,}")
print(f"Warmup Steps : {warmup_steps:,}")

lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value = 0.0,
    peak_value = 3e-4,
    warmup_steps = warmup_steps,
    decay_steps = total_steps,
    end_value = 1e-5
)

optimizer = nnx.Optimizer(
    model, 
    optax.adamw(learning_rate = lr_schedule, weight_decay = 0.01)
)

metrics = nnx.MultiMetric(
    loss = nnx.metrics.Average("loss")
)