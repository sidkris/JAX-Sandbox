import jax 
import jax.numpy as jnp
import flax.nnx as nnx 
import matplotlib.pyplot as plt 


# Create Embeddings
class TokenAndPositionEmbedding(nnx.Module):
    
    def __init__(self, maxlen, vocab_size, embed_dim, *, rngs):
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs = rngs)
        self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs = rngs)

    def __call__(self, x):
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        return self.token_emb(x) + self.pos_emb(positions)
    

def causal_attention_mask(seq_len):
    return jnp.tril(jnp.ones((seq_len, seq_len)))


mask = causal_attention_mask(8)
plt.figure(figsize=(6, 5))
plt.imshow(mask, cmap = "Blues", interpolation = "nearest")
plt.xlabel("Key Position")
plt.ylabel("Query Position")
plt.title("Causal Attention Mask (White = Attend, Blue = Masked)")
plt.colorbar(label = "Attention Allowed")
plt.tight_layout()
plt.show()