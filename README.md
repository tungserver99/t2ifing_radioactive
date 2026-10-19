# Native Semantic Radioactive Fingerprint

Teacher-side semantic fingerprinting for SD2.1, with ordinary SD1.4 student adaptation using the existing Serum experiment harness.

After preparing your server environment, run from this repository root:

```bash
bash run_semantic_fp.sh
```

The launcher calls `python main.py` without environment activation or automatic installation. The default is the pilot; `--setting full` enables all six student branches, and `--plan-only` prints the protocol without GPU work.

Runs and logs are saved beside `main.py` in `run_semantic_fp_*/logs/`. Teacher training defaults to microbatch1 accumulated to effective batch4, with activation checkpointing for the differentiable reference/cycle path.

- [Full usage and environment requirements](README_semantic_fp.md)
- [Method specification](native_semantic_radioactive_fingerprint_codex_spec.md)
- [Specification audit and verification limits](docs/validation/semantic_spec_audit.md)

Only CPU logic checks have been run locally. Real-model merge equivalence, GPU execution and fingerprint inheritance remain to be validated on the server. The audit records four retained failures/errors in the legacy Windows/POSIX and ZIP tests.
