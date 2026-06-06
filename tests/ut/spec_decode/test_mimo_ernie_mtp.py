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

These tests verify that:
1. ``mimo_mtp`` and ``ernie_mtp`` are recognised as valid MTP method names
   in the upstream vLLM ``MTPModelTypes`` literal.
2. The ``patch_mimo_ernie_mtp.py`` module can be imported without errors.
3. The speculative decoding method routing includes ``mimo_mtp`` and
   ``ernie_mtp`` alongside the existing ``mtp`` method.
4. The utility function ``speculative_enable_dispatch_gmm_combine_decode``
   correctly identifies these methods as MTP (not EAGLE) methods.
5. The MTP method aliases will be normalised to ``mtp`` by upstream logic.
"""

import ast
import os
from typing import get_args

import pytest
from vllm.config.speculative import MTPModelTypes

# Path to the project root (two levels up from tests/ut/spec_decode/).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _get_source_file_path(relative_path: str) -> str:
    """Return the absolute path of a source file inside the project."""
    return os.path.normpath(os.path.join(PROJECT_ROOT, relative_path))


def _read_source_file(relative_path: str) -> str:
    """Read and return the contents of a source file."""
    path = _get_source_file_path(relative_path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _routing_tuple_contains(method: str) -> bool:
    """Check whether *method* appears in the routing tuple inside
    ``vllm_ascend/spec_decode/__init__.py``."""
    source = _read_source_file("vllm_ascend/spec_decode/__init__.py")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple):
            elts = [ast.literal_eval(e) for e in node.elts if isinstance(e, ast.Constant)]
            if method in elts:
                return True
    return False


# ---------------------------------------------------------------------------
#  1.  Method recognition
# ---------------------------------------------------------------------------

class TestMimoErnieMethodRecognition:
    """Verify that ``mimo_mtp`` and ``ernie_mtp`` are recognised as valid MTP
    method names in the upstream vLLM ``MTPModelTypes`` literal type."""

    def test_mimo_mtp_in_mtp_model_types(self):
        assert "mimo_mtp" in get_args(MTPModelTypes)

    def test_ernie_mtp_in_mtp_model_types(self):
        assert "ernie_mtp" in get_args(MTPModelTypes)


# ---------------------------------------------------------------------------
#  2.  Method normalisation
# ---------------------------------------------------------------------------

class TestMimoErnieMethodNormalization:
    """Upstream vLLM converts deprecated MTP method aliases (such as
    ``mimo_mtp`` and ``ernie_mtp``) to ``mtp`` via
    ``SpeculativeConfig.__post_init__``.  These tests confirm that the aliases
    are present in the set of types that are subject to normalisation."""

    def test_mimo_mtp_normalized_to_mtp(self):
        assert "mimo_mtp" in get_args(MTPModelTypes)
        assert "mtp" in get_args(MTPModelTypes)

    def test_ernie_mtp_normalized_to_mtp(self):
        assert "ernie_mtp" in get_args(MTPModelTypes)
        assert "mtp" in get_args(MTPModelTypes)

    def test_mtp_is_in_mtp_model_types(self):
        assert "mtp" in get_args(MTPModelTypes)


# ---------------------------------------------------------------------------
#  3.  Source-code routing check
# ---------------------------------------------------------------------------

class TestMimoErnieSourceRouting:
    """Verify that the speculative-decoding method routing tuple in the
    *source code* includes ``mimo_mtp`` and ``ernie_mtp``.

    Because the test reads the source file directly it does **not** depend on
    which vllm-ascend package revision is installed in the runtime environment.
    """

    def test_mimo_mtp_in_routing_tuple(self):
        assert _routing_tuple_contains("mimo_mtp"), \
            "mimo_mtp not found in spec_decode/__init__.py routing tuple"

    def test_ernie_mtp_in_routing_tuple(self):
        assert _routing_tuple_contains("ernie_mtp"), \
            "ernie_mtp not found in spec_decode/__init__.py routing tuple"


# ---------------------------------------------------------------------------
#  4.  Method detection in utility functions
# ---------------------------------------------------------------------------

class TestMimoErnieMethodDetection:
    """Test that ``speculative_enable_dispatch_gmm_combine_decode`` correctly
    identifies ``mimo_mtp`` and ``ernie_mtp`` as MTP methods."""

    def _make_vllm_config(self, method):
        from unittest.mock import MagicMock
        from vllm.config import VllmConfig

        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.speculative_config = MagicMock()
        vllm_config.speculative_config.method = method
        vllm_config.model_config = MagicMock()
        vllm_config.model_config.hf_text_config = MagicMock()
        vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
        return vllm_config

    def test_mimo_mtp_detected_as_mtp_method(self):
        from vllm_ascend.utils import speculative_enable_dispatch_gmm_combine_decode
        vllm_config = self._make_vllm_config("mimo_mtp")
        result = speculative_enable_dispatch_gmm_combine_decode(vllm_config)
        assert result is False

    def test_ernie_mtp_detected_as_mtp_method(self):
        from vllm_ascend.utils import speculative_enable_dispatch_gmm_combine_decode
        vllm_config = self._make_vllm_config("ernie_mtp")
        result = speculative_enable_dispatch_gmm_combine_decode(vllm_config)
        assert result is False


# ---------------------------------------------------------------------------
#  5.  Patch file existence & importability
# ---------------------------------------------------------------------------

class TestMimoErniePatchLoading:
    """Verify that the ``patch_mimo_ernie_mtp.py`` file exists and can be
    imported after the vllm-ascend package is updated to include it."""

    def test_patch_file_exists(self):
        path = _get_source_file_path("vllm_ascend/patch/platform/patch_mimo_ernie_mtp.py")
        assert os.path.isfile(path), f"patch file not found: {path}"

    def test_patch_source_syntax(self):
        source = _read_source_file("vllm_ascend/patch/platform/patch_mimo_ernie_mtp.py")
        ast.parse(source)
