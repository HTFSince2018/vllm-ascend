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
# Adapted from tests/e2e/singlecard/spec_decode/test_mtp_eagle_correctness.py
#
"""Compare the outputs of a speculative LLM (using mimo_mtp or ernie_mtp)
and a non-speculative reference LLM using the same seed and temperature.

Both runs use the same prompts, seed (42), and greedy sampling (temperature=0).
The speculative outputs should match the non-speculative outputs for at least
66% of the prompts, confirming that the speculative method does not alter
the model's output distribution."""

from __future__ import annotations

import os

import pytest
from vllm import SamplingParams

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

    with VllmRunner(
        MIMO_MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        max_num_seqs=256,
        max_model_len=4096,
        seed=42,
        speculative_config=spec_config,
    ) as spec_llm:
        spec_outputs = spec_llm.generate(EXAMPLE_PROMPTS, _get_sampling_params())

    with VllmRunner(
        MIMO_MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        max_num_seqs=256,
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
            print(f"ref_output: {ref_output[1][0]}")
            print(f"spec_output: {spec_output[1][0]}")

    # Heuristic: expect at least 66% of the prompts to match exactly
    # Upon failure, inspect the outputs to check for inaccuracy.
    threshold = 0.66
    assert matches > int(threshold * len(ref_outputs)), (
        f"method={method}, matches={matches}/{len(ref_outputs)}, "
        "indicating speculative decoding changed the outputs."
    )
    cleanup_dist_env_and_memory()
    del spec_llm
    del ref_llm
