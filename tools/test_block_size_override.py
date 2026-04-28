#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: ``--block-size`` (== ``LLM(block_size=...)``) overrides
upstream's preferred block_size end-to-end.

Background:
- After the vLLM 0.20 bump, ``Platform.update_block_size_for_backend`` Phase 1
  picks ``MetalBackend.get_preferred_block_size()`` (32 from ``MultipleOf(32)``)
  unless ``cache_config.user_specified_block_size`` is True.
- ``--block-size N`` (CLI) and ``LLM(block_size=N)`` (Python API) both set that
  flag, so the caller's value should survive Phase 1, Phase 2 (non-hybrid skips
  it), and our wrapper's post-super pass-through.

This script loads a small real model with two different block_size values and
checks that the engine's ``cache_config.block_size`` matches the requested
value. Each iteration runs in a fresh subprocess so MLX state and the EngineCore
process tree are reset cleanly between runs.

Usage:
    python tools/test_block_size_override.py
"""

from __future__ import annotations

import os
import subprocess
import sys

MODEL = "Qwen/Qwen3-0.6B"
TEST_BLOCK_SIZES = (8, 16)


def _run_one(block_size: int) -> int:
    """Subprocess entry point: load LLM, print result, exit 0 on match."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        block_size=block_size,
        max_model_len=512,
        max_num_batched_tokens=64,
        enforce_eager=True,
    )
    actual = llm.llm_engine.vllm_config.cache_config.block_size
    out = llm.generate(["Hello"], SamplingParams(max_tokens=3, temperature=0))
    text = out[0].outputs[0].text
    status = "PASS" if actual == block_size else "FAIL"
    print(
        f"[block_size_override] requested={block_size} actual={actual} "
        f"text={text!r} [{status}]",
        flush=True,
    )
    return 0 if actual == block_size else 1


def main() -> int:
    if len(sys.argv) == 2:
        return _run_one(int(sys.argv[1]))

    env = os.environ.copy()
    env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    env.setdefault("VLLM_METAL_USE_PAGED_ATTENTION", "1")
    env.setdefault("VLLM_METAL_MEMORY_FRACTION", "0.8")

    failures: list[int] = []
    for bs in TEST_BLOCK_SIZES:
        print(f"\n=== Loading {MODEL} with block_size={bs} ===", flush=True)
        proc = subprocess.run(
            [sys.executable, __file__, str(bs)],
            env=env,
        )
        if proc.returncode != 0:
            failures.append(bs)

    print()
    if failures:
        print(f"FAIL: block_size override did not stick for: {failures}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
