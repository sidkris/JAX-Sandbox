import jax.numpy as jnp
from jax import grad, jit, vmap

def predict(params, inputs):
    for w, b in params:
        outputs = jnp.dot(inputs, w) + b
        inputs = jnp.tanh(outputs)
    return inputs

def loss(params, batch):
    inputs, targets = batch
    predictions = predict(params, inputs)
    return jnp.mean((predictions - targets) ** 2)


gradient_function = jit(grad(loss))
perexample_grads = jit(vmap(grad(loss), in_axes=(None, 0)), in_shardings = ..., out_shardings= ...)