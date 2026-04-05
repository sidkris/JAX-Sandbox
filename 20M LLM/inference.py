import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import orbax.checkpoint as ocp
import tiktoken 
from pathlib import Path
from mini_llm import MiniLLM


tokenizer = tiktoken.get_encoding("gpt2")

# -----------------------------
# Load model
# -----------------------------
def load_model():
    rngs = nnx.Rngs(0)
    ckpt_path = (Path.cwd() / "small_checkpoint.orbax").resolve().as_posix()
    model = MiniLLM(
        maxlen = 128,
        vocab_size = tokenizer.n_vocab,
        embed_dim = 192,
        num_heads = 6,
        feed_forward_dim = 512, 
        num_transformer_blocks = 6,
        rngs = nnx.Rngs(0)
    )

    state = nnx.state(model)
    checkpointer = ocp.StandardCheckpointer()

    restored = checkpointer.restore(ckpt_path, item=state)
    nnx.update(model, restored)

    return model


# -----------------------------
# Generate
# -----------------------------
def generate(model, input_ids, max_new_tokens=50, seed=0):
    rng = jax.random.PRNGKey(seed)

    tokens = list(input_ids)

    for _ in range(max_new_tokens):
        x = jnp.array(tokens)[None, :]   # (1, T)

        logits = model(x)                # (1, T, vocab)
        next_token_logits = logits[0, -1]

        rng, sub = jax.random.split(rng)
        next_token = jax.random.categorical(sub, next_token_logits)

        tokens.append(int(next_token))

    return np.array(tokens)


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    
    model = load_model()

    prompt = input("Enter prompt: ")
    prompt = tokenizer.encode(prompt)
    prompt = np.array(prompt, dtype=np.int32)
    model_output = generate(model, prompt, max_new_tokens=100)
    print(model_output)