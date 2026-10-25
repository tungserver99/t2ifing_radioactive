# Replay diagnostics and orderly teacher-only failure

The user selected: collect all replay diagnostics and report the failure; train only when replay passes. This preserves rescue specification sections 14, 30, 31, 44 and 52.

## Design

Keep every frozen replay threshold, trace, image, carrier, key, training objective and final selection rule unchanged. Collect all four native-batch replay samples instead of raising at the first failed image. Record anchor, latent, cached-output and scheduler-state comparisons, plus replay/PNG, original-decoded/PNG and replay/original-decoded pixel/LPIPS metrics. Additional comparisons explain failures and never replace the frozen pass criteria.

A numerical replay rejection writes replay diagnostics and a teacher-only blocked report, prints an explicit failed-validation outcome and returns without training or students. Unexpected runtime/integrity failures write an execution error report and retain a nonzero exit. Successful replay retains the existing replay proof and training path. Existing successful replay proofs remain reusable.

Resume after source changes requires the explicit existing --upgrade-execution flag, extended narrowly to this pre-training v2 diagnostics revision. Verify exact baseline/current source hashes, reject unrelated changes and already-started training, and record an immutable manifest-upgrade audit. Never edit the experiment configuration or trace protocol.

## Alternatives

- Let training proceed after failed replay: rejected by the user and section 14.
- Increase LPIPS tolerance automatically: rejected; insufficient GPU evidence and changes a frozen scientific prerequisite.
- Finish diagnostics then stop with a report: selected by the user.

## Implementation and verification

1. Add failing CPU tests for collecting every image, frozen gates, blocked reports, unexpected errors and narrow pre-training upgrades.
2. Implement the diagnostic helper and update replay/campaign orchestration without modifying the math.
3. Add the audited pre-training upgrade policy and regression tests.
4. Run focused tests, the existing semantic suite and full harness suite where dependencies permit. State GPU verification limits.
5. Document the exact PBS resume command and expected reports. A blocked replay is not a completed teacher experiment, so it must not create teacher_complete.json.

## Review focus

- Failed replay must never invoke training or produce a success proof.
- PNG quantization diagnostics must not relax the existing gate.
- Runtime errors must not be converted into successful jobs.
- Code upgrades must not bypass unrelated integrity checks or alter trained runs.
- Failed diagnostic reports must be inspectable without GPU inference and without requiring teacher_complete.json.
