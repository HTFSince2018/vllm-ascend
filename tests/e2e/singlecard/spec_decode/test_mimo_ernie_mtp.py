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
"""E2E tests for mimo_mtp and ernie_mtp speculative decoding on Ascend NPU.

Test coverage:
  - test_mimo_ernie_mtp_dummy_load :  Verify model loads and architecture resolves
  - test_mimo_ernie_mtp_correctness:  Semantic consistency (spec vs reference)
  - test_mimo_ernie_mtp_multi_request: Continuous multi-request stability
  - test_mimo_ernie_mtp_openai_api   : OpenAI-compatible API serving
"""

from __future__ import annotations

import os

import openai
import pytest
from vllm import SamplingParams

from tests.e2e.conftest import RemoteOpenAIServer, VllmRunner, cleanup_dist_env_and_memory

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


def _make_server_args(method: str) -> list[str]:
    return [
        "--speculative-config",
        f'{{"method": "{method}", "num_speculative_tokens": 1}}',
        "--trust-remote-code",
        "--max-model-len",
        "4096",
        "--seed",
        "42",
        "--gpu-memory-utilization",
        "0.7",
        "--tensor-parallel-size",
        "1",
    ]


def test_mimo_ernie_mtp_dummy_load():
    """Verify the MiMo model loads and its architecture resolves correctly."""
    with VllmRunner(
        MIMO_MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        max_num_seqs=256,
        max_model_len=4096,
        seed=42,
        speculative_config={
            "method": "mimo_mtp",
            "num_speculative_tokens": 1,
        },
    ) as llm:
        assert llm is not None
        cleanup_dist_env_and_memory()
        del llm


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


@pytest.mark.parametrize("method", ["mimo_mtp", "ernie_mtp"])
def test_mimo_ernie_mtp_multi_request(method: str):
    """Verify the speculative model handles multiple consecutive
    generate requests without crashing."""
    spec_config = {
        "method": method,
        "num_speculative_tokens": 1,
    }

    with VllmRunner(
        MIMO_MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        max_num_seqs=256,
        max_model_len=4096,
        seed=42,
        speculative_config=spec_config,
    ) as llm:
        for i in range(3):
            outputs = llm.generate(EXAMPLE_PROMPTS, _get_sampling_params())
            assert len(outputs) == len(EXAMPLE_PROMPTS), (
                f"Request {i + 1}: expected {len(EXAMPLE_PROMPTS)} outputs, "
                f"got {len(outputs)}"
            )
            print(f"  Multi-request {i + 1}/3: {len(outputs)} outputs generated")
        cleanup_dist_env_and_memory()
        del llm


@pytest.mark.parametrize("method", ["mimo_mtp", "ernie_mtp"])
def test_mimo_ernie_mtp_openai_api(method: str):
    """Verify that the model serves OpenAI-compatible API requests
    with the speculative method enabled.

    This test starts a vllm serve subprocess, waits for it to be ready,
    sends a completion request, and validates the JSON response.
    """
    server_args = _make_server_args(method)
    with RemoteOpenAIServer(
        MIMO_MODEL,
        vllm_serve_args=server_args,
        max_wait_seconds=600,
    ) as server:
        client = server.get_client()
        response = client.completions.create(
            model=MIMO_MODEL,
            prompt=EXAMPLE_PROMPTS[0],
            max_tokens=50,
            temperature=0.0,
        )
        assert len(response.choices) > 0, "No choices returned"
        assert response.choices[0].text is not None, "Empty response text"
        assert len(response.choices[0].text.strip()) > 0, "Response text is whitespace only"
