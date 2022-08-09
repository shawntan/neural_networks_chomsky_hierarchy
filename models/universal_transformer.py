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

_INF_LOGITS = 10000

def layer_norm():
    """Applies a unique LayerNorm to x with default settings."""
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln

class PositionalEncodings(enum.Enum):
    NONE = 0
    SIN_COS = 1
    ALIBI = 2
    RELATIVE = 3


def sin_cos_positional_encodings(sequence_length: int,
                                 embedding_size: int,
                                 with_negative: bool = False,
                                 max_time: float = 10000.0) -> jnp.ndarray:
    """Generates positional encodings for the input.

    Args:
      sequence_length: The length of the output sequence.
      embedding_size: The size of the embedding to consider. Must be even.
      with_negative: Whether to also compute values before 0 (useful for
        shifting).
      max_time: (default 10000) Constant used to scale position by in the
        encodings.
    Returns:
      A tensor of size [seq_len, emb_size].

    Raises:
      ValueError if embedding_size is odd.
    """
    if embedding_size % 2 == 1:
        raise ValueError(
            'Embedding sizes must be even if using positional encodings.')

    # Generate a sequence of positions and frequencies.
    if not with_negative:
        pos = jnp.arange(0, sequence_length, dtype=jnp.float32)
    else:
        pos = jnp.arange(-sequence_length + 1, sequence_length, dtype=jnp.float32)
    freqs = jnp.arange(0, embedding_size, 2, dtype=jnp.float32)
    inverse_freqs = 1.0 / (max_time ** (freqs / embedding_size))

    # We combine [seq_len] and [emb_size / 2] to [seq_len, emb_size / 2].
    pos_emb = jnp.einsum('i,j->ij', pos, inverse_freqs)
    return jnp.concatenate([jnp.sin(pos_emb), jnp.cos(pos_emb)], -1)


def _fixed_encodings_to_relative(encodings: jnp.ndarray) -> jnp.ndarray:
    """Returns a matrix of shifted encodings.

    If the input is [[-2], [-1], [0], [1], [2]], the output will be
      [[[0], [1], [2]]
       [[-1], [0], [1]]
       [[-2], [-1], [0]]]

    Args:
      encodings: A tensor of encodings, of shape (length, encoding_size).

    Returns:
      A tensor of shifted encodings, of shape
      (length//2+1, length//2+1, encoding_size).
    Raises:
      ValueError if encodings is not in dimension 2.
    """
    if encodings.ndim != 2:
        raise ValueError('`logits` needs to be an array of dimension 2.')
    sequence_length, num_hiddens = encodings.shape
    if sequence_length == 1:
        return jnp.expand_dims(encodings, axis=0)
    sequence_length = sequence_length // 2 + 1
    index_matrix = jnp.sum(
        jnp.stack([
            k * jnp.eye(sequence_length, sequence_length, k=k, dtype=jnp.int32)
            for k in range(1, sequence_length)
        ]),
        axis=0)
    index_matrix = index_matrix - jnp.transpose(index_matrix)
    index_matrix += sequence_length - 1
    shifted = jnp.take(
        encodings, jnp.reshape(index_matrix, (sequence_length ** 2,)), axis=0)
    return jnp.reshape(shifted, (sequence_length, sequence_length, num_hiddens))


def compute_attention_with_relative_encodings(queries: jnp.ndarray,
                                              keys: jnp.ndarray) -> jnp.ndarray:
    """Returns attention with relative positional encodings.

    This code strictly follows what is described in the TransformerXL paper.
    https://arxiv.org/pdf/1901.02860.pdf

    Args:
      queries: The queries used for attention. Shape (b, t, h, d).
      keys: The keys used for attention. Shape (b, t, h, d').

    Returns:
      The attention logits. Shape (b, h, t, t).
    """
    sequence_length, num_heads, num_hiddens = queries.shape[-3:]

    #  First compute the content logits.
    content_bias = hk.get_parameter(
        name='relpos_contentbias',
        shape=[num_heads, num_hiddens],
        init=hk.initializers.RandomNormal(stddev=0.02))
    content_logits = jnp.einsum('bthd,bThd->bhtT', queries + content_bias, keys)

    #  Then compute the relative part.
    relative_bias = hk.get_parameter(
        name='relpos_relativebias',
        shape=[num_heads, num_hiddens],
        init=hk.initializers.RandomNormal(stddev=0.02))
    sin_cos = sin_cos_positional_encodings(
        sequence_length, num_hiddens, with_negative=True)
    shifted_sin_cos = _fixed_encodings_to_relative(sin_cos)
    relative_keys = hk.Linear(num_hiddens, name='k_params')(shifted_sin_cos)
    relative_logits = jnp.einsum('bthd,Ttd->bhtT', queries + relative_bias,
                                 relative_keys)  # No need to broadcast batch.
    return content_logits + relative_logits


def compute_alibi_encodings_biases(
        attention_shape: Tuple[int, int, int, int]) -> jnp.ndarray:
    """Returns the biases following the ALiBi method.

    This code strictly follows what is described in the ALiBi paper.
    https://arxiv.org/pdf/2108.12409.pdf

    Args:
      attention_shape: The attention logits shape. Shape (b, h, t, t).

    Returns:
      The alibi biases, same shape as the input logits shape.
    """
    batch_size, num_heads, sequence_length, _ = attention_shape

    base_coeff = 2 ** (-8 / num_heads)
    # Coeffs tensor of shape (h, 1, 1).
    coeffs = jnp.array([base_coeff ** i for i in range(1, num_heads + 1)])
    coeffs = jnp.expand_dims(coeffs, -1)
    coeffs = jnp.expand_dims(coeffs, -1)

    # Biases tensor of shape (h, t, t).
    #  The upper part of the matrix is not zero like in the paper because we
    # don't use causal attention.
    if sequence_length == 1:
        biases = jnp.zeros((1, 1))
    else:
        biases = jnp.sum(
            jnp.stack([
                k * jnp.eye(sequence_length, sequence_length, k=k)
                for k in range(1, sequence_length)
            ]),
            axis=0)
        biases -= jnp.transpose(biases)
        biases = jnp.stack([biases] * num_heads, axis=0)

    #  Multiply the biases with the coeffs, and batch the resulting tensor.
    biases = coeffs * biases
    return jnp.stack([biases] * batch_size, axis=0)


def compute_sliding_window_mask(sequence_length: int,
                                attention_window: int) -> jnp.ndarray:
    """Returns a k-diagonal mask for a sliding window.

    Args:
      sequence_length: The length of the sequence, which will determine the shape
        of the output.
      attention_window: The size of the sliding window.

    Returns:
      A symmetric matrix of shape (sequence_length, sequence_length),
      attention_window-diagonal, with ones on the diagonal and on all the
      upper/lower diagonals up to attention_window // 2.

    Raises:
      ValueError if attention_window is <= 0.
    """
    if attention_window <= 0:
        raise ValueError(
            f'The attention window should be > 0. Got {attention_window}.')

    if attention_window == 1:
        return jnp.eye(sequence_length, sequence_length)

    attention_mask = jnp.sum(
        jnp.stack([
            jnp.eye(sequence_length, sequence_length, k=k, dtype=jnp.int32)
            for k in range(1, attention_window // 2 + 1)
        ]),
        axis=0)
    attention_mask += jnp.transpose(attention_mask)
    attention_mask += jnp.eye(sequence_length, sequence_length)
    return attention_mask

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
   
        ln1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        # First the attention block.
        attn_block = hk.MultiHeadAttention(
            num_heads=self._num_heads,
            key_size=self._hiddens_per_head,
            model_size=self._emb_dim,
            w_init_scale=2 / self._num_layers
        )
        ln2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        # Then the dense block.
        dense_block = hk.Sequential([
            hk.Linear(2 * self._emb_dim, w_init=initializer),
            jnn.gelu,
            hk.Linear(self._emb_dim, w_init=initializer),
        ])

        h = embeddings
        def trnsfrm_block(i, h):
            h_norm = ln1(h)
            h_attn = attn_block(h_norm, h_norm, h_norm)
            h_attn = hk.dropout(hk.next_rng_key(), self._dropout_prob, h_attn)
            h = h + h_attn
            h_norm = ln2(h)
            h_dense = dense_block(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), self._dropout_prob, h_dense)
            h = h + h_dense
            return h

        if hk.running_init():
            h = trnsfrm_block(0, h)
        else:
            # for _ in range(x.shape[1] ):
            #     h = trnsfrm_block(h)
            h = jax.lax.fori_loop(0, x.shape[1], trnsfrm_block, h)

        return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)


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
