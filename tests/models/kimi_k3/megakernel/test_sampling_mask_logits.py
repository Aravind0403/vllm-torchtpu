# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for sampling mask_logits unpacking, grammar masking, and sharding."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel.program import TP_SIZE, VOCAB
from vllm_torchtpu.models.kimi_k3.megakernel.sampling import (
    AXIS,
    apply_grammar_mask,
    mask_logits,
)

_NEG_INF = -1e30
_BITS_PER_WORD = 32
_INT32_BOUND = 2**31
VOCAB_WORDS = VOCAB // _BITS_PER_WORD


def test_mask_logits_unpacking_and_sharding() -> None:
    """Unpack bitmask shards per TP rank and set disallowed logits to -1e30."""
    width = VOCAB // TP_SIZE
    rows = 3
    rng = np.random.default_rng(0)
    mask = rng.integers(
        -_INT32_BOUND,
        _INT32_BOUND,
        (rows, VOCAB_WORDS),
        dtype=np.int32,
    )
    logits = rng.normal(size=(rows, width)).astype(np.float32)

    masked = jax.jit(mask_logits)
    for rank in (0, 1, 3, TP_SIZE - 1):
        out = np.asarray(masked(jnp.asarray(logits), jnp.asarray(mask), rank))
        words = mask[
            :,
            rank * width // _BITS_PER_WORD : (rank + 1) * width // _BITS_PER_WORD,
        ]
        bit_shifts = np.arange(_BITS_PER_WORD, dtype=np.int32)
        allowed = ((words[:, :, None] >> bit_shifts) & 1) != 0
        allowed = allowed.reshape(rows, width)
        assert bool(((out == _NEG_INF) == ~allowed).all())
        np.testing.assert_array_equal(out[allowed], logits[allowed])


def test_mask_logits_all_ones_is_identity_and_finite_when_rank_empty() -> None:
    """Preserve finite logits unchanged when every bitmask word is all-ones."""
    width = VOCAB // TP_SIZE
    rows = 2
    mask = np.full((rows, VOCAB_WORDS), -1, dtype=np.int32)
    logits = np.ones((rows, width), dtype=np.float32)

    masked = jax.jit(mask_logits)
    out = np.asarray(masked(jnp.asarray(logits), jnp.asarray(mask), 0))
    np.testing.assert_array_equal(out, logits)


def test_apply_grammar_mask_sharded_round_trip() -> None:
    """Verify apply_grammar_mask under shard_map matches rank-0 mask_logits."""
    width = VOCAB // TP_SIZE
    rows = 2
    rng = np.random.default_rng(7)
    mask = rng.integers(
        -_INT32_BOUND,
        _INT32_BOUND,
        (rows, VOCAB_WORDS),
        dtype=np.int32,
    )
    logits = rng.normal(size=(rows, width)).astype(np.float32)
    mesh = Mesh(np.array([jax.devices()[0]]), (AXIS,))
    sharded_fn = jax.shard_map(
        apply_grammar_mask,
        mesh=mesh,
        in_specs=(P(AXIS), P()),
        out_specs=P(AXIS),
    )
    out = np.asarray(sharded_fn(jnp.asarray(logits), jnp.asarray(mask)))
    expected = np.asarray(mask_logits(jnp.asarray(logits), jnp.asarray(mask), 0))
    np.testing.assert_array_equal(out, expected)
