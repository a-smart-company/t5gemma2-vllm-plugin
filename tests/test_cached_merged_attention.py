from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from t5gemma2_vllm_plugin.kernels.flash_t5gemma2_attention import (
    flash_t5gemma2_attention,
    flash_t5gemma2_cached_single_attention,
    flash_t5gemma2_paged_merged_attention,
)
from t5gemma2_vllm_plugin.merged_cross_attention import (
    _build_cross_slot_mapping,
    _split_kv_cache,
)


@pytest.mark.parametrize("layout", ["five_dimensional", "packed_four_dimensional"])
def test_split_kv_cache_layouts_preserve_values_and_strides(layout: str) -> None:
    # kv_heads=2 makes shape-based guesses particularly error-prone; ndim is
    # the discriminator between the two documented backend layouts.
    num_blocks, num_kv_heads, page_size, head_size = 3, 2, 4, 5
    if layout == "five_dimensional":
        cache = torch.empty(
            num_blocks, 2, page_size, num_kv_heads, head_size
        )
        expected_key = torch.arange(
            cache[:, 0].numel(), dtype=cache.dtype
        ).reshape_as(cache[:, 0])
        expected_value = expected_key + 10_000
        cache[:, 0].copy_(expected_key)
        cache[:, 1].copy_(expected_value)
        expected_strides = (cache[:, 0].stride(), cache[:, 1].stride())
    else:
        cache = torch.empty(
            num_blocks, num_kv_heads, page_size, 2 * head_size
        )
        expected_key = torch.arange(
            num_blocks * page_size * num_kv_heads * head_size, dtype=cache.dtype
        )
        expected_key = expected_key.reshape(num_blocks, page_size, num_kv_heads, head_size)
        expected_value = expected_key + 10_000
        cache.transpose(1, 2).copy_(torch.cat([expected_key, expected_value], dim=-1))
        expected_strides = (cache.transpose(1, 2).stride(),) * 2

    key_cache, value_cache = _split_kv_cache(
        cache, num_kv_heads=num_kv_heads, head_size=head_size
    )
    torch.testing.assert_close(key_cache, expected_key)
    torch.testing.assert_close(value_cache, expected_value)
    assert key_cache.shape == (num_blocks, page_size, num_kv_heads, head_size)
    assert value_cache.shape == key_cache.shape
    assert (key_cache.stride(), value_cache.stride()) == expected_strides


@pytest.mark.parametrize(
    "cache",
    [
        torch.empty(3, 2, 4, 10),  # wrong kv-head dimension
        torch.empty(3, 2, 4, 9),  # packed head size is not 2 * head_size
        torch.empty(3, 3, 4, 2, 5),  # wrong K/V dimension
        torch.empty(3, 2, 4, 2, 6),  # wrong head size
        torch.empty(3, 2, 4, 2, 5, 1),  # unsupported rank
    ],
)
def test_split_kv_cache_rejects_malformed_shapes(cache: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="unsupported cache dimensions"):
        _split_kv_cache(cache, num_kv_heads=1, head_size=5)


def test_flash_attention_long_sliding_window_has_no_masked_block_nans() -> None:
    torch.manual_seed(3)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    seq_len = 320
    head_dim = 64
    window = 128
    scale = head_dim**-0.5
    q = torch.randn(1, 1, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    positions = torch.arange(seq_len, device=device)
    distance = positions[:, None] - positions[None, :]
    left = (window + 1) // 2
    right = (window // 2) + 1
    mask = ((distance >= 0) & (distance < left)) | (
        (distance < 0) & (-distance < right)
    )
    reference = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask[None, None],
        scale=scale,
    )
    actual = flash_t5gemma2_attention(
        q,
        k,
        v,
        is_causal=False,
        sliding_window=window,
        sm_scale=scale,
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, atol=3e-2, rtol=3e-2)


def test_cross_slot_mapping_selects_only_fresh_encoder_requests() -> None:
    positions = torch.tensor([5, 0, 1], dtype=torch.long)
    query_start_loc = torch.tensor([0, 1, 3], dtype=torch.int32)
    encoder_seq_lens = torch.tensor([10, 20], dtype=torch.int32)
    block_table = torch.tensor(
        [[1, 2, -1], [7, 8, -1]], dtype=torch.int32
    )

    actual = _build_cross_slot_mapping(
        positions,
        query_start_loc,
        encoder_seq_lens,
        block_table,
        block_size=16,
        num_cross_tokens=20,
    )

    expected = torch.cat(
        [torch.arange(112, 128), torch.arange(128, 132)]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("position_values", [list(range(6)), [5]])
def test_cached_single_attention_matches_padded_reference(
    position_values: list[int],
) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_query_heads = 4
    num_kv_heads = 1
    head_dim = 256
    capacity = 32
    cross_tokens = 7

    self_key = torch.randn(
        capacity, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    self_value = torch.randn_like(self_key)
    cross_key = torch.randn_like(self_key)
    cross_value = torch.randn_like(self_key)
    positions = torch.tensor(position_values, device=device, dtype=torch.long)
    query = torch.randn(
        len(position_values),
        num_query_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    self_tokens = max(position_values) + 1
    scale = head_dim**-0.5

    reference = flash_t5gemma2_attention(
        query.unsqueeze(0).transpose(1, 2),
        torch.cat([self_key[:self_tokens], cross_key[:cross_tokens]])
        .unsqueeze(0)
        .transpose(1, 2),
        torch.cat([self_value[:self_tokens], cross_value[:cross_tokens]])
        .unsqueeze(0)
        .transpose(1, 2),
        key_mask=torch.ones(
            1,
            self_tokens + cross_tokens,
            device=device,
            dtype=torch.int32,
        ),
        q_start_pos=torch.tensor(
            [position_values[0]], device=device, dtype=torch.int32
        ),
        is_causal=True,
        self_len=self_tokens,
        sm_scale=scale,
    ).transpose(1, 2).squeeze(0)

    actual = flash_t5gemma2_cached_single_attention(
        query,
        self_key,
        self_value,
        cross_key,
        cross_value,
        positions,
        torch.tensor([cross_tokens], device=device, dtype=torch.int32),
        sm_scale=scale,
    )

    torch.testing.assert_close(actual, reference, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_paged_attention_matches_variable_length_batch() -> None:
    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_query_heads = 4
    num_kv_heads = 1
    head_dim = 256
    page_size = 16
    self_lengths = [20, 7, 18]
    cross_lengths = [18, 5, 17]
    query_lengths = [3, 1, 2]
    num_seqs = len(self_lengths)

    query_parts = [
        torch.randn(
            length,
            num_query_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        for length in query_lengths
    ]
    position_parts = [
        torch.arange(self_len - query_len, self_len, device=device)
        for self_len, query_len in zip(self_lengths, query_lengths, strict=True)
    ]
    query = torch.cat(query_parts)
    positions = torch.cat(position_parts)
    query_start_loc = torch.tensor(
        [0, 3, 4, 6], device=device, dtype=torch.int32
    )

    num_pages = 16
    cache_shape = (num_pages, page_size, num_kv_heads, head_dim)
    self_key_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
    self_value_cache = torch.zeros_like(self_key_cache)
    cross_key_cache = torch.zeros_like(self_key_cache)
    cross_value_cache = torch.zeros_like(self_key_cache)
    self_block_table = torch.full(
        (num_seqs, 4), -1, device=device, dtype=torch.int32
    )
    cross_block_table = torch.full_like(self_block_table, -1)
    references = []
    scale = head_dim**-0.5

    for seq_idx in range(num_seqs):
        self_key = torch.randn(
            self_lengths[seq_idx],
            num_kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self_value = torch.randn_like(self_key)
        cross_key = torch.randn(
            cross_lengths[seq_idx],
            num_kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        cross_value = torch.randn_like(cross_key)

        for logical_page, start in enumerate(
            range(0, self_lengths[seq_idx], page_size)
        ):
            physical_page = seq_idx * 2 + logical_page
            self_block_table[seq_idx, logical_page] = physical_page
            count = min(page_size, self_lengths[seq_idx] - start)
            self_key_cache[physical_page, :count] = self_key[start : start + count]
            self_value_cache[physical_page, :count] = self_value[
                start : start + count
            ]

        for logical_page, start in enumerate(
            range(0, cross_lengths[seq_idx], page_size)
        ):
            physical_page = 8 + seq_idx * 2 + logical_page
            cross_block_table[seq_idx, logical_page] = physical_page
            count = min(page_size, cross_lengths[seq_idx] - start)
            cross_key_cache[physical_page, :count] = cross_key[start : start + count]
            cross_value_cache[physical_page, :count] = cross_value[
                start : start + count
            ]

        reference = flash_t5gemma2_attention(
            query_parts[seq_idx].unsqueeze(0).transpose(1, 2),
            torch.cat([self_key, cross_key]).unsqueeze(0).transpose(1, 2),
            torch.cat([self_value, cross_value]).unsqueeze(0).transpose(1, 2),
            key_mask=torch.ones(
                1,
                self_lengths[seq_idx] + cross_lengths[seq_idx],
                device=device,
                dtype=torch.int32,
            ),
            q_start_pos=position_parts[seq_idx][:1].to(torch.int32),
            is_causal=True,
            self_len=self_lengths[seq_idx],
            sm_scale=scale,
        ).transpose(1, 2).squeeze(0)
        references.append(reference)

    actual = flash_t5gemma2_paged_merged_attention(
        query,
        self_key_cache,
        self_value_cache,
        cross_key_cache,
        cross_value_cache,
        positions,
        query_start_loc,
        torch.tensor(self_lengths, device=device, dtype=torch.int32),
        torch.tensor(cross_lengths, device=device, dtype=torch.int32),
        self_block_table,
        cross_block_table,
        max_query_len=max(query_lengths),
        sm_scale=scale,
    )

    torch.testing.assert_close(
        actual, torch.cat(references), atol=3e-2, rtol=3e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_paged_attention_matches_long_cross_prompt() -> None:
    """Cover encoder KV spanning more than one Triton BLOCK_N."""
    torch.manual_seed(2)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_query_heads = 4
    num_kv_heads = 1
    head_dim = 256
    page_size = 16
    self_len = 1
    cross_len = 44
    scale = 256**-0.5

    query = torch.randn(
        1, num_query_heads, head_dim, device=device, dtype=dtype
    )
    self_key = torch.randn(
        self_len, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    self_value = torch.randn_like(self_key)
    cross_key = torch.randn(
        cross_len, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    cross_value = torch.randn_like(cross_key)

    cache_shape = (12, page_size, num_kv_heads, head_dim)
    self_key_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
    self_value_cache = torch.zeros_like(self_key_cache)
    cross_key_cache = torch.zeros_like(self_key_cache)
    cross_value_cache = torch.zeros_like(self_key_cache)
    self_key_cache[1, :self_len] = self_key
    self_value_cache[1, :self_len] = self_value
    cross_pages = [7, 8, 9]
    for logical_page, physical_page in enumerate(cross_pages):
        start = logical_page * page_size
        count = min(page_size, cross_len - start)
        cross_key_cache[physical_page, :count] = cross_key[start : start + count]
        cross_value_cache[physical_page, :count] = cross_value[start : start + count]

    reference = flash_t5gemma2_attention(
        query.unsqueeze(0).transpose(1, 2),
        torch.cat([self_key, cross_key]).unsqueeze(0).transpose(1, 2),
        torch.cat([self_value, cross_value]).unsqueeze(0).transpose(1, 2),
        key_mask=torch.ones(
            1, self_len + cross_len, device=device, dtype=torch.int32
        ),
        q_start_pos=torch.zeros(1, device=device, dtype=torch.int32),
        is_causal=True,
        self_len=self_len,
        sm_scale=scale,
    ).transpose(1, 2).squeeze(0)

    actual = flash_t5gemma2_paged_merged_attention(
        query,
        self_key_cache,
        self_value_cache,
        cross_key_cache,
        cross_value_cache,
        torch.zeros(1, device=device, dtype=torch.long),
        torch.tensor([0, 1], device=device, dtype=torch.int32),
        torch.tensor([self_len], device=device, dtype=torch.int32),
        torch.tensor([cross_len], device=device, dtype=torch.int32),
        torch.tensor([[1, -1, -1, -1]], device=device, dtype=torch.int32),
        torch.tensor([[*cross_pages, -1]], device=device, dtype=torch.int32),
        max_query_len=1,
        sm_scale=scale,
    )

    torch.testing.assert_close(actual, reference, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_paged_attention_masks_initial_self_blocks_with_cross_attention() -> None:
    torch.manual_seed(2)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_query_heads = 2
    num_kv_heads = 1
    head_dim = 64
    page_size = 16
    sliding_window = 512
    block_n = 32
    position = sliding_window + block_n + 1
    self_len = position + 1
    cross_len = 19
    scale = head_dim**-0.5

    query = torch.randn(
        1, num_query_heads, head_dim, device=device, dtype=dtype
    )
    self_key = torch.randn(
        self_len, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    self_value = torch.randn_like(self_key)
    cross_key = torch.randn(
        cross_len, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    cross_value = torch.randn_like(cross_key)

    num_self_pages = (self_len + page_size - 1) // page_size
    num_cross_pages = (cross_len + page_size - 1) // page_size
    cache_shape = (
        num_self_pages + num_cross_pages,
        page_size,
        num_kv_heads,
        head_dim,
    )
    self_key_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
    self_value_cache = torch.zeros_like(self_key_cache)
    cross_key_cache = torch.zeros_like(self_key_cache)
    cross_value_cache = torch.zeros_like(self_key_cache)
    self_block_table = torch.arange(
        num_self_pages, device=device, dtype=torch.int32
    )[None]
    cross_block_table = (
        torch.arange(num_cross_pages, device=device, dtype=torch.int32)
        + num_self_pages
    )[None]

    for page in range(num_self_pages):
        start = page * page_size
        count = min(page_size, self_len - start)
        self_key_cache[page, :count] = self_key[start : start + count]
        self_value_cache[page, :count] = self_value[start : start + count]
    for page in range(num_cross_pages):
        start = page * page_size
        count = min(page_size, cross_len - start)
        physical_page = num_self_pages + page
        cross_key_cache[physical_page, :count] = cross_key[start : start + count]
        cross_value_cache[physical_page, :count] = cross_value[start : start + count]

    positions = torch.tensor([position], device=device, dtype=torch.int32)
    actual = flash_t5gemma2_paged_merged_attention(
        query,
        self_key_cache,
        self_value_cache,
        cross_key_cache,
        cross_value_cache,
        positions,
        torch.tensor([0, 1], device=device, dtype=torch.int32),
        torch.tensor([self_len], device=device, dtype=torch.int32),
        torch.tensor([cross_len], device=device, dtype=torch.int32),
        self_block_table,
        cross_block_table,
        max_query_len=1,
        sliding_window=sliding_window,
        sm_scale=scale,
    )

    reference = flash_t5gemma2_attention(
        query.unsqueeze(0).transpose(1, 2),
        torch.cat([self_key, cross_key]).unsqueeze(0).transpose(1, 2),
        torch.cat([self_value, cross_value]).unsqueeze(0).transpose(1, 2),
        key_mask=torch.ones(
            1, self_len + cross_len, device=device, dtype=torch.int32
        ),
        q_start_pos=positions,
        is_causal=True,
        self_len=self_len,
        sliding_window=sliding_window,
        sm_scale=scale,
    ).transpose(1, 2).squeeze(0)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, atol=3e-2, rtol=3e-2)
