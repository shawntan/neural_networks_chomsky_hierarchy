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

    def route(self, x: jnp.ndarray):
        logits = hk.Linear(self._num_experts)(x)
        top_k_vals, top_k_idxs = jax.lax.top_k(logits, self._k)
        top_k_gates = jax.nn.softmax(top_k_vals, axis=-1)
        return top_k_gates, top_k_idxs

    def distribute_map(self, x: jnp.ndarray, top_k_idxs: jnp.ndarray, experts: List[Callable]) -> jnp.ndarray:
        def expert_map_i(x_i, top_k_idxs_i):
            def expert_map_ij(top_k_idxs_ij):
                if hk.running_init():
                    for f in experts:
                        out = f(x_i)
                else:
                    out = hk.switch(top_k_idxs_ij, experts, x_i)
                return out
            return hk.vmap(expert_map_ij, split_rng=(not hk.running_init()))(top_k_idxs_i)
        return hk.vmap(expert_map_i, split_rng=(not hk.running_init()))(x, top_k_idxs)

    def map(self, x: jnp.ndarray, top_k_idxs: jnp.ndarray, experts: List[Callable]) -> jnp.ndarray:
        def expert_map_i(x_i, top_k_idxs_i):
            def expert_map_ij(x_ij, top_k_idxs_ij):
                if hk.running_init():
                    for f in experts:
                        out = f(x_ij)
                else:
                    out = hk.switch(top_k_idxs_ij, experts, x_ij)
                return out
            return hk.vmap(expert_map_ij, split_rng=(not hk.running_init()))(x_i, top_k_idxs_i)
        return hk.vmap(expert_map_i, split_rng=(not hk.running_init()))(x, top_k_idxs)

    def weighted_reduce(self, x, top_k_gates):
        return jnp.einsum('...kd,...k->...d', x, top_k_gates)

if __name__ == "__main__":

    def moe(x):
        num_experts = 4
        experts_query = [hk.Linear(7) for i in range(num_experts)]
        moe = MixtureOfExperts(num_experts, k=2)
        top_k_gates, top_k_idxs = moe.route(x)
        q = moe.distribute_map(x, top_k_idxs, experts_query)
        mid_out = moe.map(q, top_k_idxs, [hk.Linear(8) for i in range(num_experts)] )
        out = moe.weighted_reduce(mid_out, top_k_gates)
        return out

    forward_moe = hk.transform(moe)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2000, 20))
    # print(x)
    rng_key = jax.random.PRNGKey(42)
    params = forward_moe.init(rng=rng_key, x=x)
    fn_apply = jax.jit(forward_moe.apply)
    out_buf = fn_apply(params=params, x=x, rng=rng_key)
    print(out_buf.shape)
    out_buf = fn_apply(params=params, x=x, rng=rng_key)
    print(out_buf.shape)