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

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
import jax
import numpy as np

from .transformer import \
        compute_attention_with_relative_encodings, \
        sin_cos_positional_encodings
from .transformer import PositionalEncodings

_INF_LOGITS = 10000


def layer_norm():
    """Applies a unique LayerNorm to x with default settings."""
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln


class MultiHeadAttention(hk.Module):
    """Multi-headed attention mechanism.

    As described in the vanilla Transformer paper:
      "Attention is all you need" https://arxiv.org/abs/1706.03762
    """
    def __init__(self,
                 num_heads: int,
                 key_size: int,
                 w_init_scale: float,
                 dropout: float = 0.,
                 value_size: Optional[int] = None,
                 model_size: Optional[int] = None,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads
        self.w_init = hk.initializers.VarianceScaling(w_init_scale)
        self.dropout = dropout

    def __call__(self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray,
                 mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Compute (optionally masked) MHA with queries, keys & values."""
        query_heads = self._linear_projection(query, self.key_size, "query")
        key_heads = self._linear_projection(key, self.key_size, "key")
        value_heads = self._linear_projection(value, self.value_size, "value")

        # attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        attn_logits = \
            compute_attention_with_relative_encodings(query_heads, key_heads, self.dropout)

        sqrt_key_size = np.sqrt(self.key_size).astype(key.dtype)
        attn_logits = attn_logits / sqrt_key_size

        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} must match logits "
                    f"{attn_logits.ndim}."
                )
            attn_logits = jnp.where(mask, attn_logits, -1e30)

        attn_weights = jax.nn.softmax(attn_logits)
        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        # Concatenate attention matrix of all heads into a single vector.
        attn_vec = jnp.reshape(attn, (*query.shape[:-1], -1))
        return hk.Linear(self.model_size, w_init=self.w_init)(attn_vec)

    @hk.transparent
    def _linear_projection(
            self,
            x: jnp.ndarray,
            head_size: int,
            name: Optional[str] = None
    ) -> jnp.ndarray:
        y = hk.Linear(self.num_heads * head_size, w_init=self.w_init, name=name)(x)
        return y.reshape((*x.shape[:-1], self.num_heads, head_size))


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
        embs_init = hk.initializers.TruncatedNormal(stddev=self._emb_init_scale)
        embeddings = hk.Linear(self._emb_dim, with_bias=False, w_init=embs_init)(x)
        batch_size, sequence_length , embedding_size = embeddings.shape
        # embeddings += sin_cos_positional_encodings(sequence_length, embedding_size)

        ln1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        # First the attention block.
        attn_block = MultiHeadAttention(
            num_heads=self._num_heads,
            key_size=self._hiddens_per_head,
            model_size=self._emb_dim,
            w_init_scale=2 / self._num_layers,
            dropout=self._dropout_prob
        )
        ln2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        # Then the dense block.
        dense_block = hk.Sequential([
            hk.Linear(2 * self._emb_dim, w_init=initializer),
            jnn.gelu,
            hk.Linear(self._emb_dim, w_init=initializer),
        ])


        _f_halt = hk.Linear(2, w_init=initializer)
        def f_halt(h):
            z = _f_halt(h)
            return jnn.log_softmax(z, axis=-1)

        def update_halting_log_probs(log_g, log_acc_no_halt, log_acc_halt):
            log_acc_no_halt = log_acc_no_halt + log_g[..., 0]
            log_halt = log_acc_no_halt + log_g[..., 1]
            log_acc_halt = jnp.logaddexp(log_acc_halt, log_halt)
            return log_acc_no_halt, log_acc_halt, log_halt

        def last_update(log_g, log_acc_no_halt, log_acc_halt):
            log_acc_no_halt = log_acc_no_halt - 64.
            log_halt = log_acc_no_halt
            log_acc_halt = jnp.logaddexp(log_acc_halt, log_halt)
            return log_acc_no_halt, log_acc_halt, log_halt



        def trnsfrm_block(i, state):
            (h, log_acc_no_halt, log_acc_halt, h_out) = state

            halted = jnp.exp(log_acc_halt)[..., None]
            prev_h = h

            h_norm = ln1(h)
            h_attn = attn_block(h_norm, h_norm, h_norm)
            h_attn = hk.dropout(hk.next_rng_key(), self._dropout_prob, h_attn)
            h = h + h_attn
            h_norm = ln2(h)
            h_dense = dense_block(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), self._dropout_prob, h_dense)
            curr_h = h + h_dense

            log_acc_no_halt, log_acc_halt, log_halt = jax.lax.cond(
                i + 1 == curr_h.shape[1],
                last_update,
                update_halting_log_probs,
                f_halt(curr_h), log_acc_no_halt, log_acc_halt
            )

            h =  halted * prev_h + (1 - halted) * curr_h
            p_halt = jnp.exp(log_halt)[..., None]
            h_out = h_out + p_halt * curr_h
            # print(jnp.exp(log_acc_halt[0]), log_acc_halt.shape)
            return h, log_acc_no_halt, log_acc_halt, h_out


        h = embeddings
        h_out = jnp.zeros_like(h)
        log_acc_no_halt = jnp.zeros_like(h[..., 0])
        log_acc_halt = jnp.full_like(h[..., 0], -64.)
        if hk.running_init():
            h, log_acc_no_halt, log_acc_halt, h_out = \
                trnsfrm_block(0, (h, log_acc_no_halt, log_acc_halt, h_out))
        else:
            if True:
                h, log_acc_no_halt, log_acc_halt, h_out = \
                    jax.lax.fori_loop(
                        0, x.shape[1], trnsfrm_block,
                        (h, log_acc_no_halt, log_acc_halt, h_out)
                    )
            else:
                for i in range(x.shape[1]):
                    h, log_acc_no_halt, log_acc_halt, h_out = \
                        trnsfrm_block(
                            i, (h, log_acc_no_halt, log_acc_halt, h_out))
        return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h_out)


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
