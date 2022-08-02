import jax
import jax.numpy as jnp
import haiku as hk
from numpy import np
from typing import Callable, Optional, Tuple, List

class MixtureOfExperts(hk.Module):
    def __init__(self, num_experts, k, name=None):
        super().__init__(name=name)
        self._num_experts = num_experts
        self._k = k

    def route(self, x:jnp.ndarray):
        logits = hk.Linear(self._num_experts)(x)
        top_k_vals, top_k_idxs = jax.lax.top_k(logits, self._k)
        top_k_gates = jax.nn.softmax(top_k_vals, axis=-1)
        return top_k_gates, top_k_idxs

    def map(self, top_k_idxs:jnp.ndarray, x:jnp.ndarray)\
            -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        # total_counts, embedding_size = x.shape
        expert_idxs = jnp.arange(self._num_experts, dtype=np.int32)
        top_k_mask = expert_idxs[:, None, None] == top_k_idxs[None, :, :]
        sharded_x = [None] * self._num_experts
        idxs = [None] * self._num_experts
        for i in range(self._num_experts):
            idxs[i] = source_idx, slot_idx = jnp.where(top_k_mask[i])
            sharded_x[i] = x[source_idx]
        return idxs, sharded_x

    def apply_experts(self, sharded_y, experts):
        return jax.tree_util.tree_map(lambda f,x: f(x), experts, sharded_y)

    def gather(self, in_shape, idxs, sharded_y):
        out_buf = jnp.empty((in_shape, self._k, sharded_y[0].shape[-1]),
                            dtype=sharded_y[0].dtype)
        for i in range(self._num_experts):
            out_buf = out_buf.at[idxs[i]].set(sharded_y[i])
        return out_buf

    def reduce(self, in_shape, idxs, sharded_y, top_k_gates):
        out_buf = jnp.empty((in_shape, sharded_y[0].shape[-1]), dtype=sharded_y[0].dtype)
        for i in range(self._num_experts):
            source_idx, slot_idx = idxs[i]
            out_buf = out_buf.at[source_idx].add(
                top_k_gates[source_idx, slot_idx][:, None] * sharded_y[i])
        return out_buf

if __name__ == "__main__":
    def moe(x):
        num_experts = 4
        experts = [hk.Linear(7) for i in range(num_experts)] 
        moe = MixtureOfExperts(num_experts, k=2)
        top_k_gates, top_k_idxs = moe.route(x)
        idxs, sharded_x = moe.map(top_k_idxs, x)
        sharded_outs = moe.apply_experts(sharded_x, experts)
        return moe.reduce(x.shape[0], idxs, sharded_outs, top_k_gates)

    forward_moe = hk.transform(moe)
