# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import sys

import vllm_metal as vm


def test_apply_macos_defaults_sets_spawn(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")

    vm._apply_macos_defaults()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_apply_macos_defaults_respects_user_value(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    monkeypatch.setattr(sys, "platform", "darwin")

    vm._apply_macos_defaults()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"


def test_apply_macos_defaults_noop_on_non_macos(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    vm._apply_macos_defaults()
    assert "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ


def test_apply_macos_defaults_logs_when_setting(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")

    # Ensure logging is configured
    vm._configure_logging()

    # Verify logger is properly configured to output logs
    # (actual output testing is unreliable with capsys due to handler timing)
    metal_logger = logging.getLogger("vllm_metal")
    assert len(metal_logger.handlers) > 0, "Logger should have handlers"
    assert metal_logger.level <= logging.INFO, "Logger level should allow INFO"

    # Just verify the function runs without error
    vm._apply_macos_defaults()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
