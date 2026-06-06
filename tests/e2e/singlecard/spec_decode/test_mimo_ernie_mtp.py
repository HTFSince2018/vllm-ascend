#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#
"""Compare the outputs of a non-speculative LLM and a speculative LLM
using mimo_mtp and ernie_mtp methods. Outputs should be semantically
consistent when using the same seed and temperature."""

from __future__ import annotations

import logging
import os
import re

import pytest
from vllm import SamplingParams
from vllm.logger import logger as vllm_logger

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MIMO_MODEL = os.getenv("MIMO_MODEL_PATH", "/data/models/MiMo-7B-RL")
EXAMPLE_PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


def _get_sampling_params():
    return SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=False)


class _MetricsCapture(logging.Handler):
    """Capture SpecDecoding metrics lines from the vLLM logger."""

    def __init__(self):
        super().__init__()
        self.captured: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        if "SpecDecoding metrics" in msg:
            self.captured.append(msg)

    def get_metrics(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {}
        for line in self.captured:
            m = re.search(r"Mean acceptance length:\s*([\d.]+)", line)
            if m:
                result["mean_acceptance_length"] = float(m.group(1))
            m = re.search(r"Accepted:\s*(\d+)\s*tokens?", line)
            if m:
                result["accepted_tokens"] = int(m.group(1))
            m = re.search(r"Draft acceptance rate:\s*([\d.]+)%", line)
            if m:
                result["draft_acceptance_rate_pct"] = float(m.group(1))
        return result


@pytest.mark.parametrize("method", ["mimo_mtp", "ernie_mtp"])
@pytest.mark.parametrize("num_speculative_tokens", [1])
def test_mimo_ernie_mtp_correctness(method: str, num_speculative_tokens: int):
    """Compare outputs of a reference (non-speculative) run and a
    speculative run with the given method.

    Both runs use the same seed and temperature, so the outputs should
    match for at least 66% of the prompts.
    """
    spec_config = {
        "method": method,
        "num_speculative_tokens": num_speculative_tokens,
    }

    handler = _MetricsCapture()
    handler.setFormatter(logging.Formatter("%(message)s"))
    vllm_logger.addHandler(handler)

    with VllmRunner(
        MIMO_MODEL,
        tensor_parallel_size=1,
        max_model_len=4096,
        seed=42,
        speculative_config=spec_config,
    ) as spec_llm:
        spec_outputs = spec_llm.generate(EXAMPLE_PROMPTS, _get_sampling_params())

    vllm_logger.removeHandler(handler)
    metrics = handler.get_metrics()

    with VllmRunner(
        MIMO_MODEL,
        tensor_parallel_size=1,
        max_model_len=4096,
        seed=42,
    ) as ref_llm:
        ref_outputs = ref_llm.generate(EXAMPLE_PROMPTS, _get_sampling_params())

    matches = 0
    misses = 0
    for ref_output, spec_output in zip(ref_outputs, spec_outputs):
        ref_token_ids = ref_output[0][0]
        spec_token_ids = spec_output[0][0]
        if ref_token_ids == spec_token_ids[: len(ref_token_ids)]:
            matches += 1
        else:
            misses += 1
            print(f"  ref_output: {ref_output[1][0]}")
            print(f"  spec_output: {spec_output[1][0]}")

    total = len(ref_outputs)
    threshold = 0.66
    acceptance_rate = matches / total if total > 0 else 0.0

    print(f"\n=== SpecDecode Results: method={method}, num_speculative_tokens={num_speculative_tokens} ===")
    print(f"  Prompts matched:   {matches}/{total} ({acceptance_rate:.1%})")
    print(f"  Prompts diverged:  {misses}/{total}")
    if metrics.get("mean_acceptance_length") is not None:
        print(f"  Mean acceptance length:     {metrics['mean_acceptance_length']:.2f}")
    if metrics.get("accepted_tokens") is not None:
        print(f"  Accepted tokens:            {metrics['accepted_tokens']}")
    if metrics.get("draft_acceptance_rate_pct") is not None:
        print(f"  Draft acceptance rate:      {metrics['draft_acceptance_rate_pct']:.1f}%")
    print(f"  Token-level acceptance threshold: {threshold:.0%}")
    print(f"  Result: {'PASS' if acceptance_rate > threshold else 'FAIL'}\n")

    assert acceptance_rate > threshold, (
        f"method={method}, matches={matches}/{total}, "
        "indicating speculative decoding changed the outputs."
    )
    cleanup_dist_env_and_memory()
    del spec_llm
    del ref_llm
