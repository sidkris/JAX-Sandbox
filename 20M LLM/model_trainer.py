import jax
import jax.numpy as jnp
import flax.nnx as nnx 
import grain.python as grain 
import tiktoken
import optax
from data_loader import load_data
from mini_llm import MiniLLM 
import orbax.checkpoint as orbax
import Path

maxlen = 128
tokenizer = tiktoken.get_encoding("gpt2")
num_epochs = 3
text_dl, batches_per_epoch = load_data()
total_steps = num_epochs * batches_per_epoch

model = MiniLLM(
    maxlen = 128,
    vocab_size = tokenizer.n_vocab,
    embed_dim = 192,
    num_heads = 6,
    feed_forward_dim = 512, 
    num_transformer_blocks = 6,
    rngs = nnx.Rngs(0)
)

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

optimizer = nnx.ModelAndOptimizer(
    model, 
    optax.adamw(learning_rate = lr_schedule, weight_decay = 0.01)
)

metrics = nnx.MultiMetric(
    loss = nnx.metrics.Average("loss")
)


@nnx.jit
def train_step(model, optimizer, metrics, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux = True)
    (loss, logits), grads = grad_fn(model, batch)

    metrics.update(loss = loss, logits = logits, labels = batch[1])
    optimizer.update(grads)


metrics_history = {"train_loss" : []}


prep_target_batch = jax.vmap(
    lambda tokens : jnp.concatenate((tokens[1:], jnp.array([0])))
)


for epoch in range(num_epochs):
    step = 0
    for batch in text_dl:
        input_batch = jnp.array(jnp.array(batch).T).astype(jnp.int32)
        target_batch = prep_target_batch(jnp.array(jnp.array(batch).T)).astype(jnp.int32)
        print(".", end = "")
        train_step(model, optimizer, metrics, (input_batch, target_batch))

        if (step + 1) % 2 == 0:
            for metric, value in metrics.compute().items():
                metrics_history[f"train_{metric}"].append(value)
            
            metrics.reset()

            current_lr = lr_schedule(step)
            print(f"\EPOCH : {epoch + 1} | STEP : {step + 1} | LOSS : {metrics_history} | LR : {current_lr}\n")
        step += 1


checkpoint_path = Path.cwd() / "small_checkpoint.orbax"

checkpointer = orbax.PyTreeCheckpointer()

checkpointer.save(checkpoint_path, nnx.state(model), force = True)
print(f"Model Saved as {checkpoint_path}")
