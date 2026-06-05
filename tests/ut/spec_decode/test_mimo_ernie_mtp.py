#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Tests for mimo_mtp and ernie_mtp method support."""

from typing import get_args
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CacheConfig, VllmConfig
from vllm.config.speculative import MTPModelTypes

import vllm_ascend.spec_decode.eagle_proposer as eagle_proposer
from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer


class TestMimoErnieMethodRecognition:
    """Test that mimo_mtp and ernie_mtp are recognized as valid MTP methods."""

    def test_mimo_mtp_in_mtp_model_types(self):
        """mimo_mtp should be listed as a valid MTP model type."""
        assert "mimo_mtp" in get_args(MTPModelTypes)

    def test_ernie_mtp_in_mtp_model_types(self):
        """ernie_mtp should be listed as a valid MTP model type."""
        assert "ernie_mtp" in get_args(MTPModelTypes)


class TestMimoErnieMethodRouting:
    """Test that mimo_mtp and ernie_mtp route to the correct proposer."""

    def _make_vllm_config(self, method: str):
        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.speculative_config = MagicMock()
        vllm_config.cache_config = MagicMock(spec=CacheConfig)
        vllm_config.cache_config.block_size = 16
        vllm_config.scheduler_config = MagicMock()
        vllm_config.scheduler_config.max_num_batched_tokens = 1024
        vllm_config.scheduler_config.max_num_seqs = 32
        vllm_config.model_config = MagicMock()
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.max_model_len = 2048
        vllm_config.model_config.uses_mrope = False
        vllm_config.model_config.uses_xdrope_dim = 0
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.data_parallel_rank = 0
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.enable_expert_parallel = False
        vllm_config.compilation_config = MagicMock()
        vllm_config.speculative_config.draft_tensor_parallel_size = 1
        vllm_config.speculative_config.num_speculative_tokens = 1
        vllm_config.speculative_config.method = method
        return vllm_config

    def test_mimo_mtp_routes_to_eagle_proposer(self):
        """get_spec_decode_method('mimo_mtp') should return AscendEagleProposer."""
        vllm_config = self._make_vllm_config("mimo_mtp")
        device = torch.device("cpu")
        runner = MagicMock()
        proposer = get_spec_decode_method("mimo_mtp", vllm_config, device, runner)
        assert isinstance(proposer, AscendEagleProposer)

    def test_ernie_mtp_routes_to_eagle_proposer(self):
        """get_spec_decode_method('ernie_mtp') should return AscendEagleProposer."""
        vllm_config = self._make_vllm_config("ernie_mtp")
        device = torch.device("cpu")
        runner = MagicMock()
        proposer = get_spec_decode_method("ernie_mtp", vllm_config, device, runner)
        assert isinstance(proposer, AscendEagleProposer)


class TestMimoErnieInEagleProposer:
    """Test that eagle_proposer handles mimo_mtp and ernie_mtp in method checks."""

    def _make_proposer(self, method: str):
        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.speculative_config = MagicMock()
        vllm_config.speculative_config.method = method
        vllm_config.cache_config = MagicMock(spec=CacheConfig)
        vllm_config.cache_config.block_size = 16
        vllm_config.scheduler_config = MagicMock()
        vllm_config.scheduler_config.max_num_batched_tokens = 1024
        vllm_config.scheduler_config.max_num_seqs = 32
        vllm_config.model_config = MagicMock()
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.max_model_len = 2048
        vllm_config.model_config.uses_mrope = False
        vllm_config.model_config.uses_xdrope_dim = 0
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.data_parallel_rank = 0
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.enable_expert_parallel = False
        vllm_config.compilation_config = MagicMock()
        device = torch.device("cpu")
        runner = MagicMock()
        return AscendEagleProposer(vllm_config, device, runner)

    def test_mimo_mtp_method_property(self):
        """Proposer.method should preserve 'mimo_mtp'."""
        proposer = self._make_proposer("mimo_mtp")
        assert proposer.method == "mimo_mtp"

    def test_ernie_mtp_method_property(self):
        """Proposer.method should preserve 'ernie_mtp'."""
        proposer = self._make_proposer("ernie_mtp")
        assert proposer.method == "ernie_mtp"

    def test_mimo_mtp_in_use_draft_model_check(self):
        """Proposer should not identify mimo_mtp as a draft_model method."""
        proposer = self._make_proposer("mimo_mtp")
        assert not proposer.uses_draft_model()

    def test_ernie_mtp_in_use_draft_model_check(self):
        """Proposer should not identify ernie_mtp as a draft_model method."""
        proposer = self._make_proposer("ernie_mtp")
        assert not proposer.uses_draft_model()
