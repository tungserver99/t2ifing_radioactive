# Native Semantic Radioactive Fingerprint

Current teacher objective: threshold-calibrated coherent shift with CFG7.5 (threshold_cfg_v1). It trains three fresh teacher LoRA candidates, merges the selected teacher into ordinary SD2.1 weights and evaluates unchanged SD1.4 student adaptation.

```bash
bash run_semantic_fp.sh
```

To reuse the exact old clean images, PCA/key and frozen threshold while training new teachers from clean weights:

```bash
bash run_semantic_fp.sh --reuse-preparation run_semantic_fp_20261019_225912_275755
```

The launcher runs only python main.py, without environment setup. Logs/results stay beside main.py. One candidate uses each visible GPU; on one GPU40GB the three candidates run sequentially. Full teacher state is saved every100 updates.

- [Current usage](README_semantic_fp.md)
- [Current teacher protocol and review assessment](docs/threshold_cfg_v1.md)
- [Historical original specification](native_semantic_radioactive_fingerprint_codex_spec.md)
- [CPU verification and limitations](docs/validation/semantic_threshold_cfg.md)

Teacher defaults to BF16, microbatch1/effective batch4, readiness checks and checkpointed reference/cycle gradients. Student training remains FP16 with the original MSE-only checkpoint selection. No actual model/GPU performance or revised fingerprint inheritance has been verified locally.
