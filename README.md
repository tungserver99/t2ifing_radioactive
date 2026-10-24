# Native Semantic Radioactive Fingerprint

The shell default is the locked **RSSF teacher trajectory rescue**
(`semantic_sign_v2_trajectory`), preserving the successful carrier and analytic
verifier while addressing teacher image utility.

```bash
bash run_semantic_fp.sh
```

One teacher, mu=.10, K_train=24, 1500 updates. Native PNDM trace anchors preserve
the trajectory; final-image RSSF + LPIPS + pixel MSE differentiate through the
last four real scheduler calls. Native milestone screening selects the teacher;
full 512-image validation enforces TPR>=.90, CLIP drop<=.01 and LPIPS<=.15.
The job stops before any student work, whether the teacher passes or fails.

Resume after a job limit with `bash run_semantic_fp.sh --resume RUN_DIR`.
The launcher uses plain `python`; install the pinned environment on the server.
Local verification uses CPU tensors only; real model replay and scientific
utility gates run on the GPU server.

- [Usage, artifacts and historical objectives](README_semantic_fp.md)
- [Locked teacher rescue specification](rssf_teacher_trajectory_rescue_codex_spec.md)
- [Original RSSF specification](semantic_sign_fingerprint_codex_spec.md)
- [Historical threshold experiment](docs/threshold_cfg_v1.md)
