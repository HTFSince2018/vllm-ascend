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
from unittest.mock import MagicMock

import torch
from vllm.config import VllmConfig
from vllm.config.speculative import MTPModelTypes

from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.utils import speculative_enable_dispatch_gmm_combine_decode


class TestMimoErnieMethodRecognition:
    """mimo_mtp and ernie_mtp must be recognised as valid MTP methods."""

    def test_mimo_mtp_in_mtp_model_types(self):
        assert "mimo_mtp" in get_args(MTPModelTypes)

    def test_ernie_mtp_in_mtp_model_types(self):
        assert "ernie_mtp" in get_args(MTPModelTypes)


class TestMimoErnieMethodNormalization:
    """Deprecated MTP aliases are normalised to mtp upstream."""

    def test_mimo_mtp_normalized_to_mtp(self):
        assert "mimo_mtp" in get_args(MTPModelTypes)
        assert "mtp" in get_args(MTPModelTypes)

    def test_ernie_mtp_normalized_to_mtp(self):
        assert "ernie_mtp" in get_args(MTPModelTypes)
        assert "mtp" in get_args(MTPModelTypes)

    def test_mtp_is_in_mtp_model_types(self):
        assert "mtp" in get_args(MTPModelTypes)


class TestMimoErnieMethodRouting:
    """get_spec_decode_method returns AscendEagleProposer for both methods."""

    def test_mimo_mtp_routes_to_eagle_proposer(self):
        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.speculative_config = MagicMock()
        vllm_config.speculative_config.method = "mimo_mtp"
        vllm_config.speculative_config.num_speculative_tokens = 1
        vllm_config.speculative_config.draft_tensor_parallel_size = 1
        vllm_config.cache_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.scheduler_config = MagicMock()
        vllm_config.scheduler_config.max_num_batched_tokens = 1024
        vllm_config.scheduler_config.max_num_seqs = 32
        vllm_config.model_config = MagicMock()
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.max_model_len = 2048
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.data_parallel_rank = 0
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.compilation_config = MagicMock()
        proposer = get_spec_decode_method("mimo_mtp", vllm_config, torch.device("cpu"), MagicMock())
        assert isinstance(proposer, AscendEagleProposer)

    def test_ernie_mtp_routes_to_eagle_proposer(self):
        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.speculative_config = MagicMock()
        vllm_config.speculative_config.method = "ernie_mtp"
        vllm_config.speculative_config.num_speculative_tokens = 1
        vllm_config.speculative_config.draft_tensor_parallel_size = 1
        vllm_config.cache_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.scheduler_config = MagicMock()
        vllm_config.scheduler_config.max_num_batched_tokens = 1024
        vllm_config.scheduler_config.max_num_seqs = 32
        vllm_config.model_config = MagicMock()
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.max_model_len = 2048
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.data_parallel_rank = 0
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.compilation_config = MagicMock()
        proposer = get_spec_decode_method("ernie_mtp", vllm_config, torch.device("cpu"), MagicMock())
        assert isinstance(proposer, AscendEagleProposer)


class TestMimoErnieMethodDetection:
    """speculative_enable_dispatch_gmm_combine_decode detects MTP methods."""

    def test_mimo_mtp_detected_as_mtp_method(self):
        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.speculative_config = MagicMock()
        vllm_config.speculative_config.method = "mimo_mtp"
        vllm_config.model_config = MagicMock()
        vllm_config.model_config.hf_text_config = MagicMock()
        vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
        assert speculative_enable_dispatch_gmm_combine_decode(vllm_config) is False

    def test_ernie_mtp_detected_as_mtp_method(self):
        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.speculative_config = MagicMock()
        vllm_config.speculative_config.method = "ernie_mtp"
        vllm_config.model_config = MagicMock()
        vllm_config.model_config.hf_text_config = MagicMock()
        vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
        assert speculative_enable_dispatch_gmm_combine_decode(vllm_config) is False


class TestMimoErniePatchLoading:
    """The patch_mimo_ernie_mtp module must be importable."""

    def test_patch_module_imports(self):
        import vllm_ascend.patch.platform.patch_mimo_ernie_mtp  # noqa: F401
