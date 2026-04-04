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

text_dl, batches_per_epoch = load_data()

model = MiniLLM()

def loss_fn(model, batch):
    inputs, targets = batch
    logits = model(inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    return loss, logits