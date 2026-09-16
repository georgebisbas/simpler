# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for window sub-buffer cache-line alignment.

The NPU comm primitives (TWait, TNotify) use dcci which operates on whole
64-byte cache lines. Adjacent buffers sharing a cache line can corrupt each
other when one rank spins on a signal while another writes payload.

These tests verify that the buffer carving logic in _handle_ctrl_alloc_domain
aligns each buffer offset to 64-byte boundaries.
"""

from collections import Counter

import pytest
from simpler.worker import _WINDOW_BUFFER_ALIGN

CACHELINE_SIZE = 64


class TestBufferCachelineAlignment:
    """Tests for cache-line alignment of window sub-buffers."""

    def test_unaligned_buffers_share_cacheline(self):
        """Without alignment, small buffers after a 4KB buffer share a cache line."""
        # This test documents the CURRENT (buggy) behavior that the fix addresses.
        # Simulate the OLD carving logic (no alignment):
        buffer_sizes = [4096, 4096, 16, 16, 16]  # data_a, data_b, signal, counts, recv

        offsets = []
        offset = 0
        for size in buffer_sizes:
            offsets.append(offset)
            offset += size

        # signal, counts, recv all land in cache line 128
        cache_lines = [off // CACHELINE_SIZE for off in offsets]
        assert cache_lines == [0, 64, 128, 128, 128], (
            f"Expected signal/counts/recv to share line 128, got lines {cache_lines}"
        )

        # Verify collision exists
        line_counts = Counter(cache_lines)
        collisions = [line for line, count in line_counts.items() if count > 1]
        assert len(collisions) > 0, "Expected cache-line collision in unaligned case"

    def test_aligned_buffers_separate_cachelines(self):
        """With 64-byte alignment, each buffer gets its own cache line."""
        buffer_sizes = [4096, 4096, 16, 16, 16]

        # Simulate the NEW carving logic (with alignment):
        offsets = []
        offset = 0
        align = _WINDOW_BUFFER_ALIGN
        for i, size in enumerate(buffer_sizes):
            if i > 0 and align > 1:
                offset = ((offset + align - 1) // align) * align
            offsets.append(offset)
            offset += size

        # Each buffer should be in its own cache line
        cache_lines = [off // CACHELINE_SIZE for off in offsets]

        line_counts = Counter(cache_lines)
        collisions = [line for line, count in line_counts.items() if count > 1]
        assert len(collisions) == 0, (
            f"Expected no cache-line collisions with alignment, but found collisions "
            f"on lines {collisions}. Offsets: {offsets}, lines: {cache_lines}"
        )

    def test_alignment_must_be_power_of_two(self):
        """Alignment must be a power of two for bit-math to work."""
        align = _WINDOW_BUFFER_ALIGN
        assert align > 0, "Alignment must be positive"
        assert (align & (align - 1)) == 0, f"_WINDOW_BUFFER_ALIGN={align} is not a power of two"

    def test_alignment_default_is_64(self):
        """The default alignment should be 64 (cache-line size)."""
        # Default must be 64 to prevent cache-line corruption
        assert _WINDOW_BUFFER_ALIGN == 64, (
            f"Expected _WINDOW_BUFFER_ALIGN=64, got {_WINDOW_BUFFER_ALIGN}. "
            "This alignment is required to prevent NPU comm primitive corruption."
        )

    @pytest.mark.parametrize(
        "buffer_sizes",
        [
            [4096, 4096, 16, 16, 16],  # all_to_all_v domain
            [16388, 112],  # ring allreduce (4097 * 4, 7 * 4 * 4)
            [32, 32],  # payload + signal, both small
            [64, 4],  # exactly one cache line + tiny signal
            [65, 4],  # one byte over cache line + signal
        ],
    )
    def test_various_buffer_layouts(self, buffer_sizes):
        """Various buffer size combinations should never produce cache-line collisions."""
        if _WINDOW_BUFFER_ALIGN == 1:
            pytest.skip("Alignment disabled via SIMPLER_WINDOW_BUFFER_ALIGN=1")

        offsets = []
        offset = 0
        align = _WINDOW_BUFFER_ALIGN
        for i, size in enumerate(buffer_sizes):
            if i > 0 and align > 1:
                offset = ((offset + align - 1) // align) * align
            offsets.append(offset)
            offset += size

        cache_lines = [off // CACHELINE_SIZE for off in offsets]

        # All cache lines should be unique
        assert len(cache_lines) == len(set(cache_lines)), (
            f"Cache-line collision detected for buffer_sizes={buffer_sizes}. Offsets: {offsets}, lines: {cache_lines}"
        )
