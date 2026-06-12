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
# Patch target: vllm/config/speculative.py
# - Allow mimo_mtp and ernie_mtp method in SpeculativeConfig._verify_args
#   so that --speculative-config '{"method": "mimo mtp"}' and
#   '{"method": "ernie mtp"}' can pass CLI validation on Ascend NPU.
#

import logging

logger = logging.getLogger(__name__)


def _patch_mimo_ernie_mtp_whitelist() -> None:
    """Allow mimo_mtp and ernie_mtp methods in SpeculativeConfig validation.

    Upstream vLLM validates that the speculative method is in a known set
    during SpeculativeConfig initialization. This patch intercepts the
    ValueError and allows mimo_mtp and ernie_mtp to pass through.
    """
    try:
        from vllm.config.speculative import SpeculativeConfig  # type: ignore
    except Exception:
        logger.warning(
            "SpeculativeConfig is not found, skip patching mimo_mtp/ernie_mtp checks."
        )
        return

    original_verify_args = getattr(SpeculativeConfig, "_verify_args", None)
    if original_verify_args is None:
        logger.warning(
            "SpeculativeConfig._verify_args is not found, skip patching "
            "mimo_mtp/ernie_mtp checks."
        )
        return
    if getattr(original_verify_args, "_vllm_ascend_mimo_ernie_mtp_patched", False):
        logger.warning("mimo_mtp/ernie_mtp checks have already been patched.")
        return

    decorators = getattr(SpeculativeConfig, "__pydantic_decorators__", None)
    mv = None
    if decorators is not None:
        model_validators = getattr(decorators, "model_validators", None)
        if isinstance(model_validators, dict):
            mv = model_validators.get("_verify_args")
    inner_verify = mv.func if mv is not None and getattr(mv, "func", None) is not None else original_verify_args

    def _patched_verify_args(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            return inner_verify(self, *args, **kwargs)
        except ValueError as e:
            method = getattr(self, "method", None)
            if method not in ("mimo_mtp", "ernie_mtp"):
                raise

            msg = str(e).lower()
            if "only supported for" in msg and "models" in msg:
                verify_vocab = getattr(self, "verify_equal_vocab_size_if_draft_model", None)
                if callable(verify_vocab):
                    verify_vocab()
                return self
            raise

    _patched_verify_args._vllm_ascend_mimo_ernie_mtp_patched = True  # type: ignore[attr-defined]
    SpeculativeConfig._verify_args = _patched_verify_args  # type: ignore[assignment]

    if mv is not None:
        try:
            mv.func = _patched_verify_args  # type: ignore[misc]
        except (TypeError, AttributeError):
            object.__setattr__(mv, "func", _patched_verify_args)
    else:
        logger.warning(
            "Could not find SpeculativeConfig.__pydantic_decorators__.model_validators["
            "'_verify_args']; mimo_mtp/ernie_mtp patch may not run at init validation."
        )

    try:
        from pydantic.dataclasses import rebuild_dataclass  # type: ignore
    except Exception as e:
        logger.warning(
            "Cannot import rebuild_dataclass (%s); mimo_mtp/ernie_mtp patch "
            "may not apply at instance construction time.",
            e,
        )
    else:
        try:
            rebuild_dataclass(SpeculativeConfig, force=True)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(
                "rebuild_dataclass(SpeculativeConfig) failed (%s); mimo_mtp/ernie_mtp patch may not apply.",
                e,
            )
        try:
            from vllm.config.vllm import VllmConfig  # type: ignore
        except Exception:
            pass
        else:
            try:
                rebuild_dataclass(VllmConfig, force=True)  # type: ignore[arg-type]
            except Exception as e:
                logger.warning(
                    "rebuild_dataclass(VllmConfig) failed (%s); VllmConfig(...) may "
                    "still use stale nested SpeculativeConfig validation.",
                    e,
                )


_patch_mimo_ernie_mtp_whitelist()
