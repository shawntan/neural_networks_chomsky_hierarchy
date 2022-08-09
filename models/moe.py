import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from typing import Callable, Optional, Tuple, List

class MixtureOfExperts(hk.Module):
    def __init__(self, num_experts, k, capacity, name=None):
        super().__init__(name=name)
        self._num_experts = num_experts
        self._k = k
        self._expert_idxs = jnp.arange(self._num_experts, dtype=np.int32)
        self._capacity = capacity

    def route(self, x:jnp.ndarray):
        logits = hk.Linear(self._num_experts)(x)
        top_k_vals, top_k_idxs = jax.lax.top_k(logits, self._k)
        top_k_gates = jax.nn.softmax(top_k_vals, axis=-1)
        # buffer_idxs
        # I x K x E: for each item, which of the top k is going to which expert,
        #            value is the i-th item of each expert
        expert_mask = top_k_idxs.transpose((1, 0)).flatten()[:, None] == self._expert_idxs
        # K * I x E
        buffer_idxs = jnp.cumsum(expert_mask, axis=0) * expert_mask - 1
        # K * I x E
        buffer_idxs = buffer_idxs.reshape(self._k, -1, self._num_experts)
        # K x I x E

        # dispatch_idxs : E x I
        dispatch_idxs = buffer_idxs.transpose((2, 0, 1)).max(axis=1)
        # combine_idxs : I x K
        combine_idxs = buffer_idxs.max(axis=-1).transpose(1, 0)
        return top_k_gates, top_k_idxs, dispatch_idxs, combine_idxs

    def dispatch(self, dispatch_idxs:jnp.ndarray, x:jnp.ndarray) -> jnp.ndarray:
        def expert_array(exp_buf_idxs):
            zeros = jnp.zeros((self._capacity + 1, x.shape[-1]))
            return zeros.at[exp_buf_idxs].set(x)[:-1]
        data = jax.vmap(expert_array)(dispatch_idxs)
        return data

    def combine(self, expert_idxs, buffer_idxs, data):
        # expert_idxs: I x K
        # buffer_idxs: I x K
        # data: E x C x D
        data = jnp.pad(data, ((0, 0), (0, 1), (0, 0)), 'constant')
        return data.at[expert_idxs, buffer_idxs].get()

    def wrap(self, cls, *args, **kwargs):
        functions = [cls(*args, **kwargs, name='expert_%d' % i)
                     for i in range(self._num_experts)]
        # nested transforms for all 8 + extract inits + applies
        init_applies = jax.tree_util.tree_map(lambda fun: hk.transform(fun), functions)  # nested transform
        inits = [x.init for x in init_applies]
        apply = init_applies[0].apply 

        def moe_fun(x):
            # get lifted parameters for each expert
            params = [hk.lift(init, name="inner")(hk.next_rng_key(), x) for init in inits]
            param_name = next(iter(params[0].keys()))
            just_params = [next(iter(p.values())) for p in params]
            stacked_params = jax.tree_util.tree_map(lambda *arrays: jnp.stack(arrays), *just_params)
            def apply_fn(params, x):
                return apply(params, hk.next_rng_key(), x)

            return jax.vmap(apply_fn)({param_name: stacked_params}, x)

        return moe_fun

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
