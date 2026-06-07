---
name: mimo-ernie-mtp
description: >
  Add mimo_mtp / ernie_mtp speculative decoding method to vLLM-Ascend.
  Covers requirement analysis, code trace, routing patch, in-place mask fix,
  and multi-level verification. For adding new methods that share an existing
  proposer path (e.g. mtp -> AscendEagleProposer).
---

# MiMo / Ernie MTP Method Support

## Overview

This skill walks through adding mimo_mtp and ernie_mtp speculative decoding to
vLLM-Ascend. It starts from requirement analysis, traces the CLI-to-NPU call chain,
implements routing extension and in-place mask patches, and validates with unit tests
and E2E tests on real hardware.

All changes stay in vllm_ascend/ (plugin side). Upstream vLLM is not modified.

## Use When

- Adding a new speculative method name that reuses an existing execution path
- The upstream model has in-place ops that need functional replacement
- You need a repeatable workflow for speculative method adaptation

## Prerequisites

- Ascend NPU (910B2C/910B/310P) with CANN 8.5.1+
- vLLM-Ascend installed (editable install recommended)
- vLLM source at /vllm-workspace/vllm
- Target model downloaded, e.g. MiMo-7B-RL at /data/models/MiMo-7B-RL
- VLLM_BATCH_INVARIANT=1 (bypasses AddRmsNormBias on CANN 8.5.1)

## Workflow

```
Phase 1: Requirement Analysis
  Check upstream enum -> find in-place mask risk

Phase 2: Code Localization
  Trace CLI -> proposer -> grep all method conditionals

Phase 3: Routing Extension
  Update routing + all method checks + CLI whitelist patch

Phase 4: In-Place Mask Patch
  Subclass upstream layer -> replace with torch.where

Phase 5: Verification
  Unit tests (no NPU) -> E2E tests (NPU + model)

Phase 6: Delivery
  SKILL.md -> PR
```

## Phase 1: Requirement Analysis

### 1.1 Check upstream enum

The method names must exist in vLLM's MTPModelTypes.

```python
from typing import get_args
from vllm.config.speculative import MTPModelTypes
"mimo_mtp" in get_args(MTPModelTypes)
"ernie_mtp" in get_args(MTPModelTypes)
```

Both are present since v0.19.1. If not, they must be added upstream first.

### 1.2 Find in-place mask risk

Search the upstream model source:

```bash
grep -rn "inputs_embeds\[" /vllm-workspace/vllm/vllm/model_executor/models/ \
  --include="*.py" | grep -i "mimo\|ernie"
```

For MiMo, this finds at vllm/model_executor/models/mimo_mtp.py:77:

```python
inputs_embeds[positions == 0] = 0   # in-place, not NPU-safe
```

This needs a functional replacement.

## Phase 2: Code Localization

### 2.1 Trace the call chain

```
CLI: --speculative-config '{"method": "mimo_mtp", ...}'
  -> EngineArgs -> SpeculativeConfig (vllm/config/speculative.py)
  -> get_spec_decode_method (vllm_ascend/spec_decode/__init__.py)
  -> AscendEagleProposer (vllm_ascend/spec_decode/eagle_proposer.py)
```

### 2.2 Find all method conditionals

```bash
grep -rn '"mtp"' vllm_ascend/ --include="*.py" | grep -v __pycache__
```

Results for v0.19.1:

| File | Count |
|------|-------|
| vllm_ascend/spec_decode/eagle_proposer.py | 15 |
| vllm_ascend/worker/model_runner_v1.py | 3 |
| vllm_ascend/ops/rotary_embedding.py | 2 |
| vllm_ascend/worker/v2/attn_utils.py | 2 |
| vllm_ascend/attention/mla_v1.py | 1 |
| vllm_ascend/utils.py | 1 |
| vllm_ascend/spec_decode/__init__.py | 1 |

## Phase 3: Routing Extension

### 3.1 Update routing

In vllm_ascend/spec_decode/__init__.py:

```python
# Before
elif method in ("eagle", "eagle3", "mtp"):
# After
elif method in ("eagle", "eagle3", "mtp", "mimo_mtp", "ernie_mtp"):
```

### 3.2 Update all method checks

Apply the same pattern to all 7 files. Use tuple membership everywhere:

```python
method in ("mtp", "mimo_mtp", "ernie_mtp")
```

### 3.3 CLI whitelist patch

Create vllm_ascend/patch/platform/patch_mimo_ernie_mtp.py.
It intercepts SpeculativeConfig._verify_args ValueError and allows the new methods.

Register in vllm_ascend/patch/platform/__init__.py:

```python
import vllm_ascend.patch.platform.patch_mimo_ernie_mtp  # noqa
```

## Phase 4: In-Place Mask Patch

### 4.1 Create the patch

vllm_ascend/patch/worker/patch_mimo_mtp.py:

```python
class AscendMiMoMultiTokenPredictorLayer(MiMoMultiTokenPredictorLayer):
    def forward(self, inputs_embeds, positions, ...):
        # functional form instead of in-place for NPU graph compat
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        # ... rest of forward unchanged
```

### 4.2 Register

In vllm_ascend/patch/worker/__init__.py:

```python
import vllm_ascend.patch.worker.patch_mimo_mtp  # noqa
```

## Phase 5: Verification

### 5.1 Unit tests (no NPU needed)

```bash
python -m pytest tests/ut/spec_decode/test_mimo_ernie_mtp.py -v
```

Expected: 11 passed. Tests cover recognition, routing, EagleProposer properties,
method detection, and patch loading.

### 5.2 E2E tests (NPU + MiMo-7B-RL needed)

```bash
VLLM_BATCH_INVARIANT=1 python -m pytest tests/e2e/singlecard/spec_decode/test_mimo_ernie_mtp.py -v -s
```

Expected: 7 passed. Tests cover dummy load, correctness (>=66% match),
multi-request stability (3 consecutive calls), and OpenAI API serving.

### 5.3 Manual check

```bash
vllm serve /data/models/MiMo-7B-RL \
  --speculative-config '{"method": "mimo_mtp", "num_speculative_tokens": 1}'
```

Verify: server starts, /v1/completions returns non-empty, engine shows
SpecDecoding metrics.

## Phase 6: Delivery

### File summary

**New files:**
- vllm_ascend/patch/platform/patch_mimo_ernie_mtp.py (CLI whitelist)
- vllm_ascend/patch/worker/patch_mimo_mtp.py (in-place mask fix)
- tests/ut/spec_decode/test_mimo_ernie_mtp.py (11 unit tests)
- tests/e2e/singlecard/spec_decode/test_mimo_ernie_mtp.py (4 E2E tests)
- .agents/skills/mimo-ernie-mtp/SKILL.md (this file)

**Modified files:**
- vllm_ascend/spec_decode/__init__.py (routing)
- vllm_ascend/spec_decode/eagle_proposer.py (15 conditionals)
- vllm_ascend/worker/model_runner_v1.py (3)
- vllm_ascend/ops/rotary_embedding.py (2)
- vllm_ascend/worker/v2/attn_utils.py (2)
- vllm_ascend/attention/mla_v1.py (1)
- vllm_ascend/utils.py (1)
- vllm_ascend/patch/platform/__init__.py (register)
- vllm_ascend/patch/worker/__init__.py (register)

### PR description template

```
## What

Add mimo_mtp and ernie_mtp speculative decoding methods.
Reuses existing mtp -> AscendEagleProposer path.

## Changes

- Routing: extend method tuples across 8 source files
- CLI: whitelist new methods in SpeculativeConfig._verify_args
- Model: replace in-place mask with torch.where

## Tests

- Unit: 11/11 pass (no NPU)
- E2E: 7/7 pass on 910B2C with MiMo-7B-RL
  - dummy load, correctness (>=66%), multi-request (3x), OpenAI API

## Limitations

- num_speculative_tokens=1 only
```

## Common Issues

| Problem | Fix |
|---------|-----|
| Ascend config not initialized | Use _setup_ascend_env() context manager |
| runner.pin_memory is MagicMock | Set runner.pin_memory = False |
| model_config.convert_type is wrong | Set convert_type = None |
| server exits unexpectedly | Add --trust-remote-code |
| aclnnAddRmsNormBias failed | Set VLLM_BATCH_INVARIANT=1 |

## Constraints

- All changes in vllm_ascend/ only, no upstream vLLM changes
- No in-place tensor ops, use torch.where instead
- Use tuple ("mtp", "mimo_mtp", "ernie_mtp") everywhere
- Model path default /data/models/MiMo-7B-RL, override via MIMO_MODEL_PATH

## Limitations

- num_speculative_tokens=1 only (upstream restricts spec_step_idx==0)
- Engine metrics visible in logs but not captured by tests
- Validated on 910B2C + CANN 8.5.1 only
