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
"""Tests for mimo_mtp and ernie_mtp method support.

These tests verify:

1. **MTP method recognition** — ``mimo_mtp`` and ``ernie_mtp`` are listed in
   the upstream vLLM ``MTPModelTypes`` literal type and will be normalised to
   ``mtp`` by ``SpeculativeConfig.__post_init__``.

2. **Method routing** — ``get_spec_decode_method`` returns an
   ``AscendEagleProposer`` instance for both method names.

3. **MTP method detection** — The utility function
   ``speculative_enable_dispatch_gmm_combine_decode`` correctly classifies
   these methods as MTP (not EAGLE) methods.

4. **Patch module** — The ``patch_mimo_ernie_mtp.py`` module loads without
   errors when the vllm-ascend package is up to date.
"""

from typing import get_args
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import CacheConfig, VllmConfig
from vllm.config.speculative import MTPModelTypes

from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.utils import speculative_enable_dispatch_gmm_combine_decode


# ---------------------------------------------------------------------------
#  1.  MTP method recognition
# ---------------------------------------------------------------------------


class TestMimoErnieMethodRecognition:
    """Both ``mimo_mtp`` and ``ernie_mtp`` must be recognised as valid MTP
    method names by the upstream vLLM framework."""

    def test_mimo_mtp_in_mtp_model_types(self):
        assert "mimo_mtp" in get_args(MTPModelTypes)

    def test_ernie_mtp_in_mtp_model_types(self):
        assert "ernie_mtp" in get_args(MTPModelTypes)


# ---------------------------------------------------------------------------
#  2.  Method normalisation
# ---------------------------------------------------------------------------


class TestMimoErnieMethodNormalization:
    """Deprecated MTP aliases are normalised to ``mtp`` upstream — confirm
    all relevant strings are in the type set so normalisation will fire."""

    def test_mimo_mtp_normalized_to_mtp(self):
        assert "mimo_mtp" in get_args(MTPModelTypes)
        assert "mtp" in get_args(MTPModelTypes)

    def test_ernie_mtp_normalized_to_mtp(self):
        assert "ernie_mtp" in get_args(MTPModelTypes)
        assert "mtp" in get_args(MTPModelTypes)

    def test_mtp_is_in_mtp_model_types(self):
        assert "mtp" in get_args(MTPModelTypes)


# ---------------------------------------------------------------------------
#  3.  Method routing
# ---------------------------------------------------------------------------


class TestMimoErnieMethodRouting:
    """``get_spec_decode_method`` must return an ``AscendEagleProposer``
    when called with either ``mimo_mtp`` or ``ernie_mtp``."""

    _COMMON_CONFIG = {
        "cache_config.block_size": 16,
        "scheduler_config.max_num_batched_tokens": 1024,
        "scheduler_config.max_num_seqs": 32,
        "model_config.dtype": torch.float16,
        "model_config.max_model_len": 2048,
        "parallel_config.tensor_parallel_size": 1,
        "parallel_config.data_parallel_rank": 0,
        "parallel_config.data_parallel_size": 1,
        "speculative_config.draft_tensor_parallel_size": 1,
        "speculative_config.num_speculative_tokens": 1,
    }

    def _make_vllm_config(self, method: str):
        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.speculative_config = MagicMock()
        vllm_config.cache_config = MagicMock(spec=CacheConfig)
        vllm_config.scheduler_config = MagicMock()
        vllm_config.model_config = MagicMock()
        vllm_config.parallel_config = MagicMock()
        vllm_config.compilation_config = MagicMock()
        for attr, val in self._COMMON_CONFIG.items():
            obj, _, field = attr.partition(".")
            getattr(vllm_config, obj).__setattr__(field, val)
        vllm_config.speculative_config.method = method
        return vllm_config

    def test_mimo_mtp_routes_to_eagle_proposer(self):
        vllm_config = self._make_vllm_config("mimo_mtp")
        proposer = get_spec_decode_method("mimo_mtp", vllm_config, torch.device("cpu"), MagicMock())
        assert isinstance(proposer, AscendEagleProposer)

    def test_ernie_mtp_routes_to_eagle_proposer(self):
        vllm_config = self._make_vllm_config("ernie_mtp")
        proposer = get_spec_decode_method("ernie_mtp", vllm_config, torch.device("cpu"), MagicMock())
        assert isinstance(proposer, AscendEagleProposer)


# ---------------------------------------------------------------------------
#  4.  MTP method detection in utility functions
# ---------------------------------------------------------------------------


class TestMimoErnieMethodDetection:
    """``speculative_enable_dispatch_gmm_combine_decode`` must return ``False``
    for MTP methods (they are *not* EAGLE methods)."""

    def _make_vllm_config(self, method):
        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.speculative_config = MagicMock()
        vllm_config.speculative_config.method = method
        vllm_config.model_config = MagicMock()
        vllm_config.model_config.hf_text_config = MagicMock()
        vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
        return vllm_config

    def test_mimo_mtp_detected_as_mtp_method(self):
        result = speculative_enable_dispatch_gmm_combine_decode(self._make_vllm_config("mimo_mtp"))
        assert result is False

    def test_ernie_mtp_detected_as_mtp_method(self):
        result = speculative_enable_dispatch_gmm_combine_decode(self._make_vllm_config("ernie_mtp"))
        assert result is False


# ---------------------------------------------------------------------------
#  5.  Patch module import
# ---------------------------------------------------------------------------


class TestMimoErniePatchLoading:
    """The ``patch_mimo_ernie_mtp.py`` module must be importable after the
    package is built / installed."""

    def test_patch_module_imports(self):
        import vllm_ascend.patch.platform.patch_mimo_ernie_mtp  # noqa: F401
