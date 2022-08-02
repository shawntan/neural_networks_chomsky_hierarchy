# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Transformer model."""

import enum
from typing import Callable, Optional, Tuple
from transformer import PositionalEncodings, layer_norm
import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
import jax
import numpy as np
from moe import MixtureOfExperts


class MultiHeadAttention(hk.Module):
    """Multi-headed attention mechanism.

    As described in the vanilla Transformer paper:
      "Attention is all you need" https://arxiv.org/abs/1706.03762
    """

    def __init__(
            self,
            num_experts: int, top_k: int,
            key_size: int,
            model_size: int,
            # TODO(romanring, tycai): migrate to a more generic `w_init` initializer.
            w_init_scale: float,
            value_size: Optional[int] = None,
            name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_experts = num_experts
        self.top_k = top_k
        self.key_size = key_size
        self.value_size = key_size
        self.model_size = model_size
        self.w_init = hk.initializers.VarianceScaling(w_init_scale)

    def __call__(
            self,
            query: jnp.ndarray,
            key: jnp.ndarray,
            value: jnp.ndarray,
            mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Compute (optionally masked) MHA with queries, keys & values."""
        key_heads = self._linear_projection(self.key_size, "key")(key)
        value_heads = self._linear_projection(self.value_size, "value")(value)
        query_experts = [self._linear_projection(self.key_size, "query_%d" % i)
                         for i in range(self.num_experts)]
        moe = MixtureOfExperts(num_experts=self.num_experts, k=self.top_k,
                               name="query_moe")

        query_flat = jnp.reshape(query, (-1, query.shape[-1]))
        top_k_gates, top_k_idxs = moe.route(query_flat)
        idxs, sharded_query = moe.map(top_k_idxs, query_flat)
        sharded_query = moe.apply_experts(sharded_query, query_experts)
        query_heads_flat = moe.gather(query_flat.shape[0], idxs, sharded_query)
        query_heads = jnp.reshape(query_heads_flat, (*query.shape[:-1], self.top_k, -1))

        attn_logits = jnp.einsum("...thd,...Td->...htT", query_heads, key_heads)
        sqrt_key_size = np.sqrt(self.key_size).astype(key.dtype)
        attn_logits = attn_logits / sqrt_key_size
        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(f"Mask dimensionality {mask.ndim} must match logits "
                                 f"{attn_logits.ndim}.")
            attn_logits = jnp.where(mask, attn_logits, -1e30)

        attn_weights = jax.nn.softmax(attn_logits)
        # Concatenate attention matrix of all heads into a single vector.
        # attn_vec = jnp.reshape(attn, (*query.shape[:-1], -1))

        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        attn_experts = [hk.Linear(self.model_size, w_init=self.w_init,
                                  name="attn_expert_%d" % i)
                        for i in range(self.num_experts)]
        attn_flat = jnp.reshape(attn, (-1, attn.shape[-1]))
        idxs, sharded_attn = moe.map(top_k_idxs, attn_flat)
        sharded_out = moe.apply_experts(sharded_attn, attn_experts)
        out_flat = moe.reduce(attn.shape[0], idxs, sharded_out, top_k_gates)
        out = jnp.reshape(out_flat, (*query.shape[:-1], -1))
        return out

    @hk.transparent
    def _linear_projection(self, head_size: int, name: Optional[str] = None):
        return hk.Linear(head_size, w_init=self.w_init, name=name)


class Transformer(hk.Module):
    """Transformer tower."""

    def __init__(
            self,
            embedding_dim: int = 128,
            num_layers: int = 2,
            num_heads: int = 8,
            hiddens_per_head: Optional[int] = None,
            dropout_prob: float = 0.1,
            emb_init_scale: float = 0.02,
            use_embeddings: bool = True,
            attention_window: Optional[int] = None,
            positional_encodings: PositionalEncodings = PositionalEncodings.SIN_COS,
            name: Optional[str] = None):
        """Initializes the transformer.

        Args:
          vocab_size: The number of tokens to consider (length of the one_hot).
          embedding_dim: The dimension of the first embedding.
          num_layers: The number of multi-head attention layers.
          num_heads: The number of heads per layer.
          hiddens_per_head: The number of hidden neurons per head. If None, equal to
            the embedding dimension divided by the number of heads.
          dropout_prob: The probability of dropout during training.
          emb_init_scale: Params initializer scale for the embeddings.
          use_embeddings: Whether to use embeddings rather than raw inputs.
          attention_window: Size of the attention sliding window. See
            MultiHeadSelfAttention.
          positional_encodings: Which positional encodings to use. Default is the
            same as in the seminal transformer paper, ie sin and cos values.
          name: The name of the module.
        """
        super().__init__(name=name)
        self._emb_dim = embedding_dim
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout_prob = dropout_prob
        self._emb_init_scale = emb_init_scale
        self._attention_window = attention_window
        self._positional_encodings = positional_encodings
        self._use_embeddings = use_embeddings
        self._hiddens_per_head = hiddens_per_head
        if hiddens_per_head is None:
            self._hiddens_per_head = embedding_dim // num_heads

    def __call__(self, x: jnp.ndarray, is_training: bool = True):
        """Returns the transformer tower output, shape [B, T, E]."""
        initializer = hk.initializers.VarianceScaling(2 / self._num_layers)
        if self._use_embeddings:
            embs_init = hk.initializers.TruncatedNormal(stddev=self._emb_init_scale)
            embeddings = hk.Linear(
                self._emb_dim, with_bias=False, w_init=embs_init)(x)
        else:
            embeddings = x

        # First the attention block.
        attn_block = MultiHeadAttention(
            num_experts=self._num_heads * self._num_layers,
            top_k=self._num_heads,
            key_size=self._hiddens_per_head,
            model_size=self._emb_dim,
            w_init_scale=2 / self._num_layers
        )
        # Then the dense block.
        dense_block = hk.Sequential([
            hk.Linear(2 * self._emb_dim, w_init=initializer),
            jnn.gelu,
            hk.Linear(self._emb_dim, w_init=initializer),
        ])

        h = embeddings
        for _ in range(self._num_layers):
            h_norm = layer_norm(h)
            h_attn = attn_block(h_norm, h_norm, h_norm)
            h_attn = hk.dropout(hk.next_rng_key(), self._dropout_prob, h_attn)
            h = h + h_attn
            h_norm = layer_norm(h)
            h_dense = dense_block(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), self._dropout_prob, h_dense)
            h = h + h_dense
        return layer_norm(h)


def make_transformer(
        num_layers: int,
        output_size: int,
        return_all_outputs: bool = False,
        embedding_dim: int = 128,
        attention_window: Optional[int] = None,
        positional_encodings: PositionalEncodings = PositionalEncodings.SIN_COS,
        dropout_prob: float = 0.1,
        is_training: bool = True) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Returns a transformer model."""

    def transformer(x):
        output = Transformer(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            positional_encodings=positional_encodings,
            dropout_prob=dropout_prob,
            attention_window=attention_window)(
            x, is_training=is_training)
        if not return_all_outputs:
            output = output[:, -1, :]
        return hk.Linear(output_size)(output)

    return transformer
