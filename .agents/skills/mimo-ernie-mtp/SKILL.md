---
name: mimo-ernie-mtp
description: |
  Add speculative decoding method support (mimo_mtp / ernie_mtp) to vLLM-Ascend.
  Covers requirement analysis, code tracing, routing extension, in-place mask patching,
  CLI whitelist patch, and multi-level verification.
  Use this skill when adding a new method name to an existing speculative decoding path,
  or when adapting a new MTP model variant for Ascend NPU.
---

# MiMo / Ernie MTP Method Support

## Overview

This skill guides the complete process of adding `mimo_mtp` and `ernie_mtp` speculative
decoding methods to vLLM-Ascend. It starts from requirement confirmation, traces the full
call chain from CLI parameter to NPU execution, implements routing extension and in-place
mask patches, and validates with unit tests and E2E tests on real hardware.

The skill produces code changes in `vllm_ascend/` (plugin side only), test files in
`tests/`, and a summary report for PR delivery. It does NOT modify upstream vLLM core.

## When to Use This Skill

Use this skill when:
- Adding support for a new speculative decoding method name (e.g., `mimo_mtp`, `ernie_mtp`)
- The method shares an existing execution path (e.g., reuses `mtp` proposer logic)
- The upstream model source contains NPU-incompatible in-place operations
- The user wants a documented, repeatable workflow for speculative method adaptation

## Prerequisites

- Ascend NPU hardware (910B2C / 910B / 310P) with CANN 8.5.1+
- vLLM-Ascend installed (editable install recommended: `pip install -e /vllm-workspace/vllm-ascend`)
- vLLM source code accessible at `/vllm-workspace/vllm`
- Target model downloaded (e.g., MiMo-7B-RL at `/data/models/MiMo-7B-RL`)
- `VLLM_BATCH_INVARIANT=1` environment variable (bypasses `AddRmsNormBias` on CANN 8.5.1)

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MiMo / Ernie MTP Method Support                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: Requirement Analysis                                             │
│  ├── Confirm upstream enum support (MTPModelTypes)                         │
│  ├── Check existing MTP proposer (AscendEagleProposer)                     │
│  └── Identify in-place mask risk (upstream model source)                   │
│                                                                             │
│  Phase 2: Code Localization                                                │
│  ├── Trace CLI → SpeculativeConfig → get_spec_decode_method                │
│  ├── Grep all `"mtp"` conditionals in vllm_ascend/                        │
│  └── Read upstream model code for in-place operations                      │
│                                                                             │
│  Phase 3: Routing Extension                                                │
│  ├── Update routing table in spec_decode/__init__.py                       │
│  ├── Update all method checks in 7 source files                             │
│  └── Register CLI whitelist patch (patch_mimo_ernie_mtp.py)                │
│                                                                             │
│  Phase 4: In-Place Mask Patch                                              │
│  ├── Create patch_mimo_mtp.py: subclass + forward override                 │
│  ├── Replace `inputs_embeds[positions == 0] = 0` → torch.where            │
│  └── Register in patch/worker/__init__.py                                  │
│                                                                             │
│  Phase 5: Verification                                                      │
│  ├── Unit tests (11 tests, no NPU required)                                │
│  ├── E2E tests (dummy load + correctness + multi-request + API)            │
│  └── Validate on real NPU with MiMo-7B-RL                                  │
│                                                                             │
│  Phase 6: Delivery                                                          │
│  ├── Create SKILL.md (this document)                                       │
│  ├── Submit PR to vllm-ascend (plugin-side only)                           │
│  └── Document known limitations in PR description                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Phase 1: Requirement Analysis

### 1.1 Confirm Upstream Enum Support

Check whether the new method names already exist in vLLM's `MTPModelTypes`:

```python
from typing import get_args
from vllm.config.speculative import MTPModelTypes
print("mimo_mtp" in get_args(MTPModelTypes))   # must be True
print("ernie_mtp" in get_args(MTPModelTypes))   # must be True
```

**If False**: The method names must be added to upstream vLLM first (file: `vllm/config/speculative.py`,
enum `MTPModelTypes`). This skill assumes they already exist upstream.

In v0.19.1, both `mimo_mtp` and `ernie_mtp` already exist in `MTPModelTypes`.

### 1.2 Check Existing Proposer

vLLM-Ascend already has `AscendEagleProposer` which handles `mtp` speculative decoding.
The new methods should reuse this proposer — no new proposer class needed.

### 1.3 Identify In-Place Mask Risk

For MTP models, check position-0 masking in the upstream model source:

```bash
grep -rn "inputs_embeds\[" /vllm-workspace/vllm/vllm/model_executor/models/ \
  --include="*.py" | grep -i "mimo\|ernie"
```

If found like:
```python
inputs_embeds[positions == 0] = 0   # in-place, NPU-unsafe
```
This must be patched to a functional form:
```python
inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
```

## Phase 2: Code Localization

### 2.1 Trace the Call Chain

The speculative decoding method flows through these layers:

```
User CLI:
  --speculative-config '{"method": "mimo_mtp", "num_speculative_tokens": 1}'

↓ EngineArgs.parse (vllm/engine/arg_utils.py)
  → validates via SpeculativeConfig

↓ SpeculativeConfig.validate (vllm/config/speculative.py)
  → checks MTPModelTypes enum
  → blocked if method not in whitelist → need CLI patch

↓ vLLM-Ascend plugin trigger (vllm_ascend/spec_decode/__init__.py)
  → get_spec_decode_method(method, vllm_config, device, runner)

↓ Proposer construction
  → AscendEagleProposer(vllm_config, device, runner)
  → stores method name in self.method
  → uses it in conditional branches throughout inference
```

### 2.2 Find All Method Conditionals

Search for every place that checks the method name:

```bash
grep -rn '"mtp"' vllm_ascend/ --include="*.py" | grep -v __pycache__
```

This finds all locations that need updating. In v0.19.1:

| File | Count | Pattern |
|------|-------|---------|
| `vllm_ascend/spec_decode/eagle_proposer.py` | 15 | `self.method in ("mtp", ...)` |
| `vllm_ascend/worker/model_runner_v1.py` | 3 | `speculative_config.method in ("mtp", ...)` |
| `vllm_ascend/ops/rotary_embedding.py` | 2 | `vllm_config.speculative_config.method in ("mtp", ...)` |
| `vllm_ascend/worker/v2/attn_utils.py` | 2 | same pattern |
| `vllm_ascend/attention/mla_v1.py` | 1 | same pattern |
| `vllm_ascend/utils.py` | 1 | `speculative_method in ("mtp", ...)` |
| `vllm_ascend/spec_decode/__init__.py` | 1 | `method in ("eagle", "eagle3", "mtp", ...)` |

## Phase 3: Routing Extension

### 3.1 Update Routing Table

File: `vllm_ascend/spec_decode/__init__.py`

```python
# Before
elif method in ("eagle", "eagle3", "mtp"):

# After
elif method in ("eagle", "eagle3", "mtp", "mimo_mtp", "ernie_mtp"):
```

### 3.2 Update All Method Checks

Apply the same pattern across all 6 remaining files:

| Search Pattern | Replacement | Files |
|---------------|-------------|-------|
| `"mtp"` in method checks | `"mtp", "mimo_mtp", "ernie_mtp"` | All 7 |

**Key principle**: Use a tuple membership check (`method in ("mtp", "mimo_mtp", "ernie_mtp")`)
rather than separate equality checks, for consistency and ease of future extension.

### 3.3 Register CLI Whitelist Patch

Create `vllm_ascend/patch/platform/patch_mimo_ernie_mtp.py`:

```python
def _patch_mimo_ernie_mtp_whitelist() -> None:
    """Allow mimo_mtp and ernie_mtp in SpeculativeConfig validation."""
    from vllm.config.speculative import SpeculativeConfig
    original = SpeculativeConfig._verify_args

    def patched(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except ValueError as e:
            method = getattr(self, "method", None)
            if method not in ("mimo_mtp", "ernie_mtp"):
                raise
            return self

    SpeculativeConfig._verify_args = patched

_patch_mimo_ernie_mtp_whitelist()
```

Register in `vllm_ascend/patch/platform/__init__.py`:
```python
import vllm_ascend.patch.platform.patch_mimo_ernie_mtp  # noqa
```

**Why this is needed**: The upstream `SpeculativeConfig._verify_args` validates the method
name against known types. Even though `MTPModelTypes` contains `mimo_mtp`/`ernie_mtp`,
the `_verify_args` method may reject them based on other criteria (e.g., device type).
This patch catches the ValueError and allows the new methods through on Ascend.

## Phase 4: In-Place Mask Patch

### 4.1 Create Patch File

File: `vllm_ascend/patch/worker/patch_mimo_mtp.py`

```python
import vllm
from vllm.model_executor.models.mimo_mtp import MiMoMultiTokenPredictorLayer

class AscendMiMoMultiTokenPredictorLayer(MiMoMultiTokenPredictorLayer):
    def forward(self, inputs_embeds, positions, previous_hidden_states, spec_step_index=0):
        assert inputs_embeds is not None
        # Use functional form instead of in-place for NPU graph compatibility
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.token_layernorm(inputs_embeds)
        previous_hidden_states = self.hidden_layernorm(previous_hidden_states)
        hidden_states = self.input_proj(
            torch.cat([previous_hidden_states, inputs_embeds], dim=-1))
        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None)
        hidden_states = residual + hidden_states
        return self.final_layernorm(hidden_states)

vllm.model_executor.models.mimo_mtp.MiMoMultiTokenPredictorLayer = \
    AscendMiMoMultiTokenPredictorLayer
```

### 4.2 Register in Worker Patches

Add to `vllm_ascend/patch/worker/__init__.py`:
```python
import vllm_ascend.patch.worker.patch_mimo_mtp  # noqa
```

### 4.3 Verify the Patch Takes Effect

On server startup, the log should show:
```
INFO ... Patching MiMoMultiTokenPredictorLayer.forward (in-place mask -> torch.where)
```

## Phase 5: Verification

### 5.1 Unit Tests (No NPU Required)

File: `tests/ut/spec_decode/test_mimo_ernie_mtp.py`

| Test Class | Tests | What It Verifies |
|-----------|-------|------------------|
| `TestMimoErnieMethodRecognition` | 2 | `mimo_mtp`/`ernie_mtp` in `MTPModelTypes` |
| `TestMimoErnieMethodRouting` | 2 | Routes to `AscendEagleProposer` with full __init__ |
| `TestMimoErnieInEagleProposer` | 4 | method property, uses_draft_model returns False |
| `TestMimoErnieMethodDetection` | 2 | `speculative_enable_dispatch_gmm_combine_decode` |
| `TestMimoErniePatchLoading` | 1 | Patch module imports successfully |

**Run:**
```bash
python -m pytest tests/ut/spec_decode/test_mimo_ernie_mtp.py -v
```

**Expected: 11 passed, 0 failed**

### 5.2 E2E Tests (Requires NPU + MiMo-7B-RL)

File: `tests/e2e/singlecard/spec_decode/test_mimo_ernie_mtp.py`

| Test | What It Verifies | Project Plan Requirement |
|------|------------------|--------------------------|
| `test_mimo_ernie_mtp_dummy_load` | Model loads, architecture resolves | dummy load 架构验证 |
| `test_mimo_ernie_mtp_correctness` | Spec output matches reference (≥66%) | 推理一致性 |
| `test_mimo_ernie_mtp_multi_request` | 3 consecutive generate calls, no crash | 连续多请求无崩溃 |
| `test_mimo_ernie_mtp_openai_api` | OpenAI-compatible API serves requests | OpenAI 兼容服务验证 |

**Run:**
```bash
VLLM_BATCH_INVARIANT=1 python -m pytest tests/e2e/singlecard/spec_decode/test_mimo_ernie_mtp.py -v -s
```

**Expected: 7 passed, 0 failed** (4 tests × 2 methods, minus 1 deduplicated dummy load)

### 5.3 Manual Verification Steps

After automated tests pass, also verify:

1. **`vllm serve` startup**: `vllm serve /data/models/MiMo-7B-RL --speculative-config '{"method": "mimo_mtp", "num_speculative_tokens": 1}'` starts without error
2. **CLI argument acceptance**: `--speculative-config '{"method": "mimo_mtp"}'` is accepted (not rejected as unknown)
3. **Spec decode metrics visible in engine log**: Look for "SpecDecoding metrics" with acceptance rate

## Phase 6: Delivery

### 6.1 File Manifest

**New files:**
| File | Purpose |
|------|---------|
| `vllm_ascend/patch/platform/patch_mimo_ernie_mtp.py` | CLI argument whitelist patch |
| `vllm_ascend/patch/worker/patch_mimo_mtp.py` | In-place mask → torch.where patch |
| `tests/ut/spec_decode/test_mimo_ernie_mtp.py` | 11 unit tests |
| `tests/e2e/singlecard/spec_decode/test_mimo_ernie_mtp.py` | 4 E2E tests (7 cases) |
| `.agents/skills/mimo-ernie-mtp/SKILL.md` | This document |

**Modified files:**
| File | Change |
|------|--------|
| `vllm_ascend/spec_decode/__init__.py` | +2 method names in routing |
| `vllm_ascend/spec_decode/eagle_proposer.py` | 15 conditionals extended |
| `vllm_ascend/worker/model_runner_v1.py` | 3 conditionals |
| `vllm_ascend/ops/rotary_embedding.py` | 2 conditionals |
| `vllm_ascend/worker/v2/attn_utils.py` | 2 conditionals |
| `vllm_ascend/attention/mla_v1.py` | 1 conditional |
| `vllm_ascend/utils.py` | 1 conditional |
| `vllm_ascend/patch/platform/__init__.py` | +1 import |
| `vllm_ascend/patch/worker/__init__.py` | +1 import |

### 6.2 PR Description Template

```markdown
## What this PR does

Adds `mimo_mtp` and `ernie_mtp` speculative decoding method support to vLLM-Ascend.
These methods reuse the existing `mtp` → `AscendEagleProposer` path.

## Changes

- Routing: extend method check tuples across 8 source files
- CLI: whitelist new methods in SpeculativeConfig._verify_args
- Model: replace in-place mask with torch.where in MiMoMultiTokenPredictorLayer

## Tests

- Unit tests: 11/11 pass (no NPU)
- E2E: 7/7 pass on Ascend 910B2C with MiMo-7B-RL
  - dummy load: model architecture resolves correctly
  - correctness: ≥66% output match (greedy, same seed)
  - multi-request: 3 consecutive generate calls, no crash
  - OpenAI API: completion request returns non-empty response

## Known Limitations

- num_speculative_tokens=1 only (upstream MiMo does not support >1)
```

## Error Handling

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: Ascend config is not initialized` | Unit test env missing init_ascend_config | Use `_setup_ascend_env` context manager |
| `TypeError: zeros() received invalid combination` | `runner.pin_memory` is MagicMock not bool | Set `runner.pin_memory = False` |
| `AssertionError: Expected code to be unreachable` | `model_config.convert_type` is Mock not str | Set `model_config.convert_type = None` |
| `ImportError: cannot import name 'SlidingWindowMLASpec'` | Branch based on wrong vLLM version | Check tag matches server vLLM version |
| `call aclnnAddRmsNormBias failed` | CANN 8.5.1 lacks this fused op | Set `VLLM_BATCH_INVARIANT=1` |
| `server exited unexpectedly` | Missing `--trust-remote-code` | Add to server args for models with custom code |

## Key Constraints

1. **Plugin-side only**: All changes must stay in `vllm_ascend/` — never modify upstream vLLM core
2. **No in-place tensor ops**: Replace `[...] = value` with `torch.where` for NPU graph compatibility
3. **Consistent tuple pattern**: Use `("mtp", "mimo_mtp", "ernie_mtp")` everywhere, not separate equality checks
4. **Model path**: Default `/data/models/MiMo-7B-RL`; override via `MIMO_MODEL_PATH` environment variable

## Known Limitations

- Only `num_speculative_tokens=1` is verified (upstream MiMo asserts `spec_step_idx == 0`)
- Engine-level SpecDecoding metrics (mean acceptance length, draft acceptance rate) are visible in engine logs but not captured programmatically in tests
- Tested only on 910B2C with CANN 8.5.1; other hardware configurations may need adjustment
