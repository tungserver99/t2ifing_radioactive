# Native Semantic Radioactive Fingerprint

Current objective: **Redundant Semantic Sign Fingerprint (`semantic_sign_v1`)**, implementing the locked [RSSF specification](semantic_sign_fingerprint_codex_spec.md). Reuses the frozen SD2.1 extractor/PCA carrier, native teacher inference and unchanged SD1.4 student protocol.

```bash
bash run_semantic_fp.sh
```

To reuse verified clean images/prompts from the completed experiment, while rebuilding the 32-bit carrier/key/medians and starting each teacher from clean SD2.1:

```bash
bash run_semantic_fp.sh --reuse-preparation run_semantic_fp_20261020_135117_009783
```

The launcher only runs `python main.py "$@"`: no environment setup, installs or interpreter paths. Logs and results remain beside `main.py`. The three fixed margins 0.10/0.20/0.30 run independently, one process per GPU; a single GPU executes them in order. Full teacher checkpoints every100 updates retain AdamW, data order and RNG state.

- [Usage and protocol](README_semantic_fp.md)
- [Approved RSSF specification](semantic_sign_fingerprint_codex_spec.md)
- [Implementation and CPU verification](docs/validation/semantic_sign_v1.md)
- [Historical threshold experiment](docs/threshold_cfg_v1.md)

Teacher selection requires analytic image TPR>=90%, CLIP drop<=0.01 and paired LPIPS<=0.15. Among passing candidates choose lowest LPIPS, pixel MSE, then margin. If none pass, students do not start and no gates/key are automatically changed. Quality and transfer of this revision require server measurements; local verification is CPU logic only.
