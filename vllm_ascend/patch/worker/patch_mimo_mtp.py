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
# Patch target: vllm/model_executor/models/mimo_mtp.py
#
# The upstream MiMoMultiTokenPredictorLayer.forward contains an in-place
# operation:
#   inputs_embeds[positions == 0] = 0
#
# This in-place mask may interfere with NPU graph capture and cause
# stability issues during speculative decoding. This patch replaces the
# in-place assignment with a functional torch.where expression.
#

import logging

import torch
import vllm
from vllm.model_executor.models.mimo_mtp import MiMoMultiTokenPredictorLayer

logger = logging.getLogger(__name__)


class AscendMiMoMultiTokenPredictorLayer(MiMoMultiTokenPredictorLayer):
    """Patched predictor layer that avoids in-place mask on inputs_embeds."""

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # masking inputs at position 0, as not needed by MTP
        # Use functional form instead of in-place for NPU graph compatibility
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.token_layernorm(inputs_embeds)
        previous_hidden_states = self.hidden_layernorm(previous_hidden_states)

        hidden_states = self.input_proj(
            torch.cat([previous_hidden_states, inputs_embeds], dim=-1)
        )

        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        hidden_states = residual + hidden_states
        return self.final_layernorm(hidden_states)


logger.info("Patching MiMoMultiTokenPredictorLayer.forward "
            "(in-place mask -> torch.where)")
vllm.model_executor.models.mimo_mtp.MiMoMultiTokenPredictorLayer = AscendMiMoMultiTokenPredictorLayer
