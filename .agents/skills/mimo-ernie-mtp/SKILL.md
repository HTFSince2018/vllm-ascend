---
name: mimo-ernie-mtp
description: |
  Add speculative decoding method support (mimo_mtp / ernie_mtp) to vLLM-Ascend.
  Covers parameter tracing, routing extension, in-place mask patching, and verification.
  Use this skill when the user wants to add a new method name to an existing speculative
  decoding path, or when adapting a new MTP model variant for Ascend NPU.
---

# MiMo / Ernie MTP Method Support

Add `mimo_mtp` and `ernie_mtp` speculative decoding methods to vLLM-Ascend.

## Workflow Overview

1. Parameter entry tracing
2. Routing extension
3. In-place mask patching
4. Test verification
5. PR delivery

---

## Step 1: Parameter Entry Tracing

Trace how `--speculative-config` flows from CLI to the actual proposer:

```
CLI args → EngineArgs → SpeculativeConfig (vllm/config/speculative.py)
    → get_spec_decode_method (vllm_ascend/spec_decode/__init__.py)
    → AscendEagleProposer (vllm_ascend/spec_decode/eagle_proposer.py)
```

**Check**: Does the method name exist in `MTPModelTypes` enum?  
If not, it must be added upstream vLLM first.  
In v0.19.1, `mimo_mtp` and `ernie_mtp` already exist in `MTPModelTypes` upstream.

---

## Step 2: Routing Extension

Every place where `method == "mtp"` is checked must also match the new method names.

**Search pattern:**
```
grep -rn '"mtp"' vllm_ascend/ --include="*.py"
```

**Replace pattern:**
```python
# Before
if method == "mtp" or method in ("mtp", ...):

# After
if method in ("mtp", "mimo_mtp", "ernie_mtp"):
```

**Files modified** (8 total):
| File | Changes |
|------|---------|
| `vllm_ascend/spec_decode/__init__.py` | Route method to `AscendEagleProposer` |
| `vllm_ascend/spec_decode/eagle_proposer.py` | 15 conditional checks |
| `vllm_ascend/worker/model_runner_v1.py` | 3 checks |
| `vllm_ascend/ops/rotary_embedding.py` | 2 checks |
| `vllm_ascend/worker/v2/attn_utils.py` | 2 checks |
| `vllm_ascend/attention/mla_v1.py` | 1 check |
| `vllm_ascend/utils.py` | 1 check |
| `vllm_ascend/patch/platform/__init__.py` | Register new patch module |

---

## Step 3: CLI Argument Whitelist

Create `vllm_ascend/patch/platform/patch_mimo_ernie_mtp.py`:

```python
# Patch SpeculativeConfig._verify_args to allow mimo_mtp/ernie_mtp
# The upstream _verify_args raises ValueError for unknown methods.
# Catch it and allow the new method names to pass through.
```

**Register** in `vllm_ascend/patch/platform/__init__.py`:
```python
import vllm_ascend.patch.platform.patch_mimo_ernie_mtp  # noqa
```

---

## Step 4: In-Place Mask Patch

Check the upstream model source for in-place operations:

| Source File | Pattern | Patch |
|------------|---------|-------|
| `vllm/model_executor/models/mimo_mtp.py:77` | `inputs_embeds[positions == 0] = 0` | Replace with `torch.where(...)` |

Create `vllm_ascend/patch/worker/patch_mimo_mtp.py`:
- Subclass `MiMoMultiTokenPredictorLayer`
- Override `forward` with functional `torch.where` instead of in-place assignment
- Monkey-patch the upstream module at import time

**Register** in `vllm_ascend/patch/worker/__init__.py`:
```python
import vllm_ascend.patch.worker.patch_mimo_mtp  # noqa
```

---

## Step 5: Verification Matrix

### Unit Tests (`tests/ut/spec_decode/test_mimo_ernie_mtp.py`)

| Test Class | Tests | What It Verifies |
|-----------|-------|------------------|
| `TestMimoErnieMethodRecognition` | 2 | `mimo_mtp`/`ernie_mtp` in `MTPModelTypes` |
| `TestMimoErnieMethodRouting` | 2 | `get_spec_decode_method` routes to `AscendEagleProposer` |
| `TestMimoErnieInEagleProposer` | 4 | Method property stored correctly, `uses_draft_model` returns False |
| `TestMimoErnieMethodDetection` | 2 | `speculative_enable_dispatch_gmm_combine_decode` detection |
| `TestMimoErniePatchLoading` | 1 | Patch module imports successfully |

**Total: 11 tests, no NPU required**

### E2E Tests (`tests/e2e/singlecard/spec_decode/test_mimo_ernie_mtp.py`)

| Test | What It Verifies |
|------|------------------|
| `test_mimo_ernie_mtp_dummy_load` | Model loads, architecture resolves |
| `test_mimo_ernie_mtp_correctness` | Spec output matches reference (≥66%) |
| `test_mimo_ernie_mtp_multi_request` | 3 consecutive generate calls, no crash |
| `test_mimo_ernie_mtp_openai_api` | OpenAI-compatible API serves requests |

**Total: 4 tests × 2 methods = 8 test cases, requires NPU + MiMo-7B-RL model**

### Run Commands

```bash
# Unit tests (no NPU)
python -m pytest tests/ut/spec_decode/test_mimo_ernie_mtp.py -v

# E2E tests (requires NPU + model)
VLLM_BATCH_INVARIANT=1 python -m pytest tests/e2e/singlecard/spec_decode/test_mimo_ernie_mtp.py -v -s
```

---

## Key Constraints

- All changes must stay in `vllm_ascend/` (plugin side) — never modify upstream vLLM core
- In-place tensor operations (`[...] = value`) must be replaced with functional forms for NPU graph compatibility
- The E2E model path defaults to `/data/models/MiMo-7B-RL`; override via `MIMO_MODEL_PATH` env var
- Set `VLLM_BATCH_INVARIANT=1` to bypass `AddRmsNormBias` operator issue on CANN 8.5.1

## Known Limitations

- Only `num_speculative_tokens=1` is verified (spec_step_idx > 0 is not supported upstream for MiMo)
- Engine-level SpecDecoding metrics (`mean acceptance length`, `draft acceptance rate`) are printed to stdout but not captured programmatically in tests
