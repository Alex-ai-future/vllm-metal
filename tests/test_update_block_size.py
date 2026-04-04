# SPDX-License-Identifier: Apache-2.0
"""Test Metal platform's update_block_size_for_backend() for hybrid models.

This test verifies that the page size alignment logic works correctly.
NOTE: In actual vLLM execution, update_block_size_for_backend() is called
after load_model(), which may be too late for KV cache initialization.
See: https://github.com/vllm-project/vllm/issues/xxxxx
"""

import pytest

from vllm.config import CacheConfig, ModelConfig, VllmConfig


@pytest.fixture
def default_vllm_config():
    """Create a default VllmConfig for testing."""
    model_config = ModelConfig(
        model="Qwen/Qwen3.5-0.8B",
        tokenizer="Qwen/Qwen3.5-0.8B",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="auto",
        seed=0,
        max_model_len=2048,
        enforce_eager=True,
    )

    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
    )

    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
    )


def test_update_block_size_for_backend_hybrid(default_vllm_config):
    """Test that update_block_size_for_backend() aligns block_size for hybrid models.
    
    This test verifies the method works in isolation.
    TODO: Fix timing issue where method is called after KV cache initialization.
    """
    from vllm.platforms import current_platform

    # Store original block_size
    original_block_size = default_vllm_config.cache_config.block_size

    # Call the method
    current_platform.update_block_size_for_backend(default_vllm_config)

    # For hybrid models, block_size should be increased to ensure
    # attention page_size >= mamba page_size
    new_block_size = default_vllm_config.cache_config.block_size

    # Block size should be >= original and aligned to 32
    assert new_block_size >= original_block_size, (
        f"block_size should increase from {original_block_size}, got {new_block_size}"
    )
    assert new_block_size % 32 == 0, f"block_size {new_block_size} not aligned to 32"

    # If mamba_cache_mode is "align", mamba_block_size should match block_size
    if default_vllm_config.cache_config.mamba_cache_mode == "align":
        assert default_vllm_config.cache_config.mamba_block_size == new_block_size


def test_update_block_size_for_backend_page_size_alignment(default_vllm_config):
    """Test that page sizes are properly aligned after update_block_size_for_backend()."""
    from vllm.platforms import current_platform
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    # Call the method
    current_platform.update_block_size_for_backend(default_vllm_config)

    cache_config = default_vllm_config.cache_config
    model_config = default_vllm_config.model_config

    # Compute attention page size
    attn_page_size = FullAttentionSpec(
        block_size=cache_config.block_size,
        num_kv_heads=model_config.get_num_kv_heads(
            default_vllm_config.parallel_config
        ),
        head_size=model_config.get_head_size(),
        dtype=model_config.dtype,
    ).page_size_bytes

    # Get mamba page size (may be padded)
    mamba_page_size = cache_config.mamba_page_size_padded

    # If padding is applied, attention page size should equal padded mamba page size
    if mamba_page_size is not None:
        assert attn_page_size == mamba_page_size, (
            f"Padded mamba page size {mamba_page_size} "
            f"!= attention page size {attn_page_size}"
        )


def test_update_block_size_for_backend_timing_issue():
    """Document the timing issue: update_block_size_for_backend() called too late.
    
    In vLLM's uniproc_executor._init_executor():
    1. load_model() is called first
    2. update_block_size_for_backend() is called after
    3. But KV cache initialization may happen during load_model()
    
    This test documents the expected behavior once the timing is fixed.
    """
    from vllm.platforms import current_platform
    
    model_config = ModelConfig(
        model="Qwen/Qwen3.5-0.8B",
        tokenizer="Qwen/Qwen3.5-0.8B",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="auto",
        seed=0,
        max_model_len=65534,
    )

    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
    )
    
    # Verify is_hybrid is detected
    assert getattr(model_config, 'is_hybrid', False) == True
    
    # Call the method
    current_platform.update_block_size_for_backend(vllm_config)
    
    # After calling, block_size should be updated
    assert cache_config.block_size >= 544, (
        f"block_size should be >= 544, got {cache_config.block_size}. "
        "This indicates update_block_size_for_backend() was not called or called too late."
    )
    assert cache_config.mamba_page_size_padded is not None, (
        "mamba_page_size_padded should be set"
    )
