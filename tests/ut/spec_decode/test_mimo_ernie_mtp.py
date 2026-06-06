"""Tests for mimo_mtp and ernie_mtp method support."""

from contextlib import contextmanager
from typing import get_args
from unittest.mock import MagicMock, patch

import torch
from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config

from vllm.config.speculative import MTPModelTypes

from vllm_ascend.ascend_config import init_ascend_config, clear_ascend_config
from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.utils import speculative_enable_dispatch_gmm_combine_decode


@contextmanager
def _setup_ascend_env(vllm_config):
    init_ascend_config(vllm_config)
    set_current_vllm_config(vllm_config)
    try:
        yield
    finally:
        set_current_vllm_config(None)
        clear_ascend_config()


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


def _make_base_vllm_config(method: str) -> tuple:
    hf_config = MagicMock()
    hf_config.architectures = ["MiMoForCausalLM"]
    hf_config.model_type = "mimo"
    hf_config.num_hidden_layers = 36

    model_config = MagicMock()
    model_config.dtype = torch.float16
    model_config.max_model_len = 2048
    model_config.hf_config = hf_config
    model_config.hf_text_config = hf_config
    model_config.uses_xdrope_dim = 0
    model_config.uses_mrope = False
    model_config.registry = MagicMock()
    model_config.registry.resolve_model_cls.return_value = (MagicMock(), "test")

    draft_hf_config = MagicMock()
    draft_hf_config.model_type = "mimo"
    draft_hf_config.architectures = ["MiMoMTPModel"]
    draft_hf_config.num_hidden_layers = 1

    draft_model_config = MagicMock()
    draft_model_config.hf_config = draft_hf_config
    draft_model_config.hf_text_config = draft_hf_config
    draft_model_config.get_hidden_size.return_value = 1024
    draft_model_config.get_inputs_embeds_size.return_value = 1024
    draft_model_config.uses_xdrope_dim = 0
    draft_model_config.uses_mrope = False

    speculative_config = MagicMock()
    speculative_config.method = method
    speculative_config.num_speculative_tokens = 1
    speculative_config.parallel_drafting = False
    speculative_config.draft_model_config = draft_model_config
    speculative_config.disable_padded_drafter_batch = False
    speculative_config.draft_tensor_parallel_size = 1
    speculative_config.use_local_argmax_reduction = False
    speculative_config.speculative_token_tree = "[(0,)]"
    speculative_config.uses_draft_model.return_value = False

    parallel_config = MagicMock()
    parallel_config.tensor_parallel_size = 1
    parallel_config.data_parallel_rank = 0
    parallel_config.data_parallel_size = 1
    parallel_config.prefill_context_parallel_size = 1
    parallel_config.enable_expert_parallel = False

    cache_config = MagicMock(spec=CacheConfig)
    cache_config.block_size = 16

    scheduler_config = MagicMock()
    scheduler_config.max_num_batched_tokens = 1024
    scheduler_config.max_num_seqs = 32

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.speculative_config = speculative_config
    vllm_config.cache_config = cache_config
    vllm_config.scheduler_config = scheduler_config
    vllm_config.model_config = model_config
    vllm_config.parallel_config = parallel_config
    vllm_config.compilation_config = MagicMock()
    vllm_config.additional_config = None

    runner = MagicMock()
    runner.pin_memory = False

    return vllm_config, torch.device("cpu"), runner


class TestMimoErnieMethodRouting:
    """get_spec_decode_method returns AscendEagleProposer for both methods."""

    def test_mimo_mtp_routes_to_eagle_proposer(self):
        vllm_config, device, runner = _make_base_vllm_config("mimo_mtp")
        with _setup_ascend_env(vllm_config):
            proposer = get_spec_decode_method("mimo_mtp", vllm_config, device, runner)
        assert isinstance(proposer, AscendEagleProposer)

    def test_ernie_mtp_routes_to_eagle_proposer(self):
        vllm_config, device, runner = _make_base_vllm_config("ernie_mtp")
        with _setup_ascend_env(vllm_config):
            proposer = get_spec_decode_method("ernie_mtp", vllm_config, device, runner)
        assert isinstance(proposer, AscendEagleProposer)


@patch("vllm_ascend.spec_decode.eagle_proposer.shared_expert_dp_enabled", return_value=False)
@patch("vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False)
@patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
class TestMimoErnieInEagleProposer:
    """EagleProposer correctly stores mimo_mtp/ernie_mtp method names."""

    def test_mimo_mtp_method_property(self, mock_cpugpubuffer, mock_multimodal, mock_shared_expert_dp):
        vllm_config, device, runner = _make_base_vllm_config("mimo_mtp")
        with _setup_ascend_env(vllm_config):
            proposer = AscendEagleProposer(vllm_config, device, runner)
        assert proposer.method == "mimo_mtp"

    def test_ernie_mtp_method_property(self, mock_cpugpubuffer, mock_multimodal, mock_shared_expert_dp):
        vllm_config, device, runner = _make_base_vllm_config("ernie_mtp")
        with _setup_ascend_env(vllm_config):
            proposer = AscendEagleProposer(vllm_config, device, runner)
        assert proposer.method == "ernie_mtp"

    def test_mimo_mtp_in_use_draft_model_check(self, mock_cpugpubuffer, mock_multimodal, mock_shared_expert_dp):
        vllm_config, device, runner = _make_base_vllm_config("mimo_mtp")
        with _setup_ascend_env(vllm_config):
            proposer = AscendEagleProposer(vllm_config, device, runner)
        assert not proposer.speculative_config.uses_draft_model()

    def test_ernie_mtp_in_use_draft_model_check(self, mock_cpugpubuffer, mock_multimodal, mock_shared_expert_dp):
        vllm_config, device, runner = _make_base_vllm_config("ernie_mtp")
        with _setup_ascend_env(vllm_config):
            proposer = AscendEagleProposer(vllm_config, device, runner)
        assert not proposer.speculative_config.uses_draft_model()


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
