# RSSF tail-control oracle verification

Date: 2026-10-26 (Australia/Sydney).
Spec: rssf_tail_control_oracle_codex_spec.md, unchanged user-supplied locked
oracle-feasibility design. Implementation is standalone and does not modify
historical runtime modules, teacher weights or source-run artifacts.

## Scientific implementation

- Four independent FP32 residuals per image modify only cached guided outputs
  in the real restored final four PNDM calls. No denoising UNet is rerun.
- The unchanged frozen RSSF semantic extractor is used for detector coordinates;
  no model parameters, key signs, cache tensors or clean targets receive gradients.
- Current Top24 margin loss, RGB8 STE, adaptive never-feasible-only detector
  weights, float LPIPS/pixel quality and normalized residual energy follow the
  fixed protocol. Candidate selection is independent per image; the final
  iterate is not automatically selected.
- Native original-dtype zero controls are exact. The FP32 graph baseline uses
  existing latent atol/rtol2e-3; canonical zero PNG/scorer coordinates use the
  declared frozen2e-3 coordinate checks. Both baselines are recorded and any
  failure aborts before optimizing that batch.
- Actual canonical FP16 VAE PNG readback and unchanged float32 verifier evidence
  are authoritative. PNG per-image quality/detection plus aggregate CLIP decides
  oracle_pass/fail. Float eligibility and model evidence remain diagnostic.
- Immutable source/model hashes, code/version protocol, output overlap guards,
  file locking and marker-last commits support read-only-source batch resume.
  Final targets are half residuals plus exact FP32 selected latents, with the
  residual storage error recorded explicitly.

## Test evidence

Interpreter: D:/anaconda3/envs/topmost/python.exe, CPU PyTorch2.5.1. Existing
Diffusers0.35.1, Transformers4.51.3, PEFT0.17.1 and LPIPS0.1.4 test dependencies
were loaded from a separate temporary dependency directory; the HPC environment
and source models were not changed.

- Oracle suites: **32 passed** (16 math/driver, 7 source/store, 9 reporting).
- Full semantic suite: **181 passed, 2 subtests passed**, with 10 existing
  Diffusers/torchvision/LPIPS warnings.
- Initial missing-feature tests were observed failing before implementation.
- A real tiny Diffusers VAE + semantic UNet + PNDM detector loss produces finite
  nonzero gradients to every delta call, while all frozen model/cache/key
  gradients remain absent. These use random tiny weights, not pretrained SD2.1.
- End-to-end CPU orchestration simulates interruption in batch2, resumes without
  recomputing/rewriting batch1, verifies unchanged source bytes, completes reports
  and reopens completed output without loading models.
- Other tests cover dynamic current Top24, identity STE matching actual
  Diffusers/PIL/PNG values, candidate tie order and fallback, finite guards,
  seeded sample IDs, locked CLI, source mutations, corrupt output, overlap,
  interrupted initialization and batch commits, PNG-vs-float disagreement,
  go/no-go/CLIP veto, Wilson CI, clean-K bins, frontier and representative panels.
- A reporting regression proves float64 mean-sign reduction can change K_M
  relative to the canonical float32 verifier. Evidence remains float32 and
  binomial p-values are validated exactly, including very small tails.
- An error-output regression first reproduced writes into an unrelated output
  containing generic config/source_manifest filenames. The CLI now recognizes
  its own oracle version and source path before writing failure metadata.
- Python compilation and git diff --check pass. PBS content is syntax-checked
  through Bash stdin and delivered only in chat; no PBS file is added.

The full bare pytest command was attempted. Collection stops in
test_bootstrap.py because bootstrap_env.py imports POSIX fcntl on Windows.
This pre-existing platform issue was not changed for the oracle.

Independent review inspected all four oracle modules against the locked spec,
then reran all32 oracle tests after the final changes; no actionable issue
remained.

## Limits

No pretrained SD2.1 CUDA oracle optimization, 256-image result or HPC timing was
measured locally. CPU tests establish implementation/control-flow properties,
not the scientific feasibility outcome. The HPC job must execute all per-batch
zero-control checks, source validations and actual PNG evaluation.

Oracle success is only evidence of a low-distortion solution in this privileged
per-image control space. It does not establish teacher-imprinting success.
Failure applies to the locked control space and 120-step optimization budget;
it is not a proof that no mathematical carrier solution exists. No model
training follows automatically in either case.
