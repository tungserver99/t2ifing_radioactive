# Correct native float replay validation

This narrow patch implements the team's replay-gate correction for
semantic_sign_v2_trajectory. The previous diagnostics-only patch preserved an
incorrect gate operand: the LPIPS serialization baseline from float VAE decode
to saved 8-bit PNG could reject an exact native replay.

## Exact gate before and after

Before: replay/PNG LPIPS <= 1e-4 and replay/PNG pixel max <= png_pixel_max.

After, ALL must pass:

- replay/reference float decode LPIPS <= the unchanged lpips_max (1e-4);
- replay/PNG pixel max <= the unchanged png_pixel_max (2/255 + 2e-3);
- reference/PNG pixel max <= the same png_pixel_max;
- all existing anchor comparisons, final latent comparison, exact cached-output
  latent replay and both scheduler-state comparisons.

Replay/PNG and reference/PNG LPIPS remain in the JSON as `lpips` and
`reference_png_lpips`, respectively, for backward-compatible diagnostics. The
hard LPIPS gate is `replay_reference_lpips`. Reports/console distinguish these
roles, record exact failed fields and identify the gate as
`float_reference_lpips_v2`. Replay proof identity includes this definition, so a
proof created for the PNG gate cannot silently satisfy the new definition.

No tolerance number, PNDM algorithm, training objective, 32-bit RSSF carrier,
Binomial verifier, key, data generation, checkpoint selection or final teacher
utility gate is changed. The original locked specification file is left intact
to preserve its hash in existing runs. This note documents the explicitly
approved correction to the replay correctness operand.

## Audit and resume

The existing --upgrade-execution entry point dispatches to policy
`pretraining_replay_gate_definition_v2` for a run whose manifest contains the
known c2651fe diagnostics revision. It accepts only v2 teacher-only runs before
training, and rejects trajectory training directories, checkpoints/adapters,
trained markers, selected teachers and completed teacher campaigns.

`replay_gate_upgrade_v2.json` lists exact before/after hashes of every changed or
added Python source, including the upgrade module itself and tests. Both LF and
CRLF checkout hashes are listed. Every other manifest entry must stay unchanged;
unknown revisions or unrelated changes are rejected. A source revision from
before the diagnostics patch cannot silently skip directly to this new gate.

Before changing the active manifest, the upgrade verifies trace manifest,
config/prompt provenance, unchanged replay tolerances, carrier/key/PCA hashes,
clean image hashes, score/image alignment and score provenance. It records SHA256
of all protected scientific inputs and checks they remain unchanged afterward.
The existing `SemanticCarrier.load` and `score_rows` validators also check
carrier structure, numeric scores, coordinates, bits/K/p-values and saved
evidence. Existing threshold/null score hashes must match before any audit or
report invalidation is written.

Audit: `execution_upgrades/replay_gate_definition_v2.json`. It records
`scientific_change=true`, the exact gate operand change and false modification
flags for training state, config, traces, carrier, key, clean images and numeric
replay tolerances. This is a gate-definition change, not diagnostics-only.

Old replay diagnostics, replay proofs, blocked reports, README and any execution
error report are archived as their exact UTF-8 text and SHA256 before active
copies are removed. This includes `teacher_trace/replay_diagnostics.json` and
`result/replay_failure.json`. The audit is written first, report invalidation is
retryable, and the new manifest is committed last. No scientific artifact is
removed or rewritten. The subsequent normal pipeline reuses clean preparation,
traces, carrier/key and scores, then recomputes replay under the corrected gate.

After updating the server checkout, use the same command inside the server PBS:

```bash
bash run_semantic_fp.sh \
  --resume /shared/homes/u26466553/projects/t2ifing_radioactive/run_semantic_fp_20261024_213309_944621 \
  --upgrade-execution
```

If direct LPIPS is zero, both PNG pixel checks and all numerical checks pass,
the corrected gate passes and teacher training begins. Any remaining failure
prints its exact failed fields and keeps the run blocked. No additional gate
is bypassed and no 5000-image clean dataset or trajectory regeneration is needed.

## Verification

Regression cases cover the reported quantization-baseline failure, real direct
replay error even when replay matches PNG, corrupt reference/PNG pairing,
immutable byte preservation, source hash restrictions, training-artifact
rejection, stale-proof/report invalidation and interrupted-upgrade retry.
Final semantic suite: **149 passed, 2 subtests passed**, with 10 existing upstream
Diffusers/torchvision/LPIPS warnings. This includes all existing scheduler,
latent, gradient-through-tail, CFG, exact resume, verifier and teacher-only
orchestration tests. Python compilation and git diff --check passed. AST
comparison against c2651fe confirmed unchanged REPLAY_TOLERANCES and unchanged
cfg_forward, restore_scheduler, replay_tail, NativeTrace and assert_state_close.
All six after-source hashes match the actual checkout.

The full harness command was also attempted; collection is blocked by the
existing POSIX fcntl import in test_bootstrap.py on Windows. GPU SD2.1 replay on
the actual HPC run is not performed locally.

Independent review identified missing numeric score verification in the initial
upgrade. Regression tests first reproduced acceptance of an edited score; the
existing production score validator and frozen threshold/null score-hash checks
now reject this before any mutation. All 22 gate-upgrade tests pass after that
fix, including edited coordinates, K and frozen score hashes. Follow-up review
found no remaining actionable issue.

## Files changed

- `Malicious_quantization/bitmark_lora/semantic_replay_diagnostics.py`: corrected
  LPIPS operand, both PNG pixel checks and explicit gated/diagnostic wording.
- `Malicious_quantization/bitmark_lora/semantic_trajectory.py`: versioned replay
  proof identity and reporting; numerical replay algorithm/checks unchanged.
- `Malicious_quantization/bitmark_lora/semantic_replay_upgrade.py`: dispatch to
  the new audited gate policy for known diagnostics-baseline runs.
- `Malicious_quantization/bitmark_lora/semantic_replay_gate_upgrade.py`: exact
  source transition, immutable input validation, archived reports and retry.
- `Malicious_quantization/bitmark_lora/replay_gate_upgrade_v2.json`: exact
  c2651fe before/after LF and CRLF source hashes.
- `Malicious_quantization/bitmark_lora/test_semantic_replay_diagnostics.py` and
  `test_semantic_replay_gate.py`: corrected gate and upgrade regressions.
- `README_semantic_fp.md` and this note: current gate and resume instructions.
