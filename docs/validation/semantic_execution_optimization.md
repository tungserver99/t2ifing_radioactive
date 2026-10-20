# Feature execution optimization

The frozen semantic extractor defines features as spatially averaged UNet mid-block output. Previously each feature forward also ran all later up-blocks and output layers, then discarded their predictions. The optimized path stops after capturing that same tensor using a private hook exception. It always removes its hook and propagates unrelated errors. Ordinary clean/teacher noise predictions and merged image generation still use complete UNet forwards.

This removes unused decoder work in raw/cycle features, including gradient-checkpoint recomputation. It does not change FP32 extraction, CFG, timestep masking, losses, weights, cycle frequency, candidates, validation or student training. CPU tests compare full-forward features, input gradients and RNG exactly, both with and without non-reentrant activation checkpointing, and assert that ordinary prediction still executes its decoder. Actual A100 speedup must be measured; no GPU/model training is performed locally.

## Continue the current threshold-CFG experiment

Wait for a complete checkpoint marker (`checkpoint-N/checkpoint.json`, saved every 100 updates), then stop the old foreground run with Ctrl+C and wait for its processes to exit before updating code. Do not pull code while workers are running: later workers import Python files anew.

```sh
git pull
bash run_semantic_fp.sh --resume YOUR_CURRENT_THRESHOLD_RUN --upgrade-execution
```

Use the current new threshold-CFG run, not the earlier fixed-alpha source run. The explicit upgrade permits only the known fddd62e versions of `run.py` and `semantic_carrier.py` plus the new execution policy/test modules. The destination runtime hashes are pinned too; unknown edits to either runtime file or any unrelated manifest change are rejected. Under the existing run lock, it saves the original and current manifests in `execution_upgrades/midblock_features_v1.json` before atomically replacing the code manifest. Config, adapters, AdamW/scaler, sampler, RNG and completed step are unchanged; normal checkpoint integrity verification still runs. This is a recorded equivalent execution upgrade, not a claim that CUDA execution across versions is bitwise reproducible.

Resume restores the most recent complete checkpoint. Updates since that checkpoint are replayed; if no complete checkpoint exists, that candidate starts at zero. The 12-update readiness probe runs again and restores weights/RNG before real training. Subsequent resumes may use ordinary `--resume`. The generic shell launcher remains one relative `python main.py "$@"` command.

## Local verification

- Semantic CPU suite: 61 tests, 60 passed and one optional Diffusers test skipped.
- Full CPU suite: 90 tests, 85 passed, one skipped, and the same four pre-existing Windows failures (two missing `fcntl`, ZIP-backslash handling, and a POSIX/Windows path expectation).
- Python compileall and git diff whitespace checks passed. No dependency installs, model downloads or GPU runs.

Conceptual independent review of the supplied patch found no correctness blocker; direct reviewer filesystem access was unavailable. The reviewer suggested pinning destination runtime hashes, which is implemented and tested. Shared-UNet access remains serialized as in the existing microbatch loop.
