import jax.numpy as jnp

def predict(params, inputs):
    for w, b in params:
        outputs = jnp.dot(inputs, w) + b
        outputs = jnp.tanh(outputs)
    return outputs

def loss(params, batch):
    inputs, targets = batch
    predictions = predict(params, inputs)
    return jnp.sum((predictions - targets) ** 2)