# Native RSSF margin refinement implementation plan

Spec: GPT_guide/rssf_native_margin_refinement_student_distill_codex_spec.md (FINAL/LOCKED design supplied by user).

- [x] Separate v3 live PNDM rollout and math; historical v2 loss preserved.
- [x] Strict source checkpoint-100/model/carrier/key/prompt/image/caption lineage; fresh optimizer and audited resume.
- [x] Mandatory server GPU smoke/native-equivalence gates; fixed 300 batch events with explicit zero-gradient exception.
- [x] Screen 50/100/200/300 on first 256 validation rows; TPR/CLIP eligibility then D26/pixel/earlier step.
- [x] Full validation gate, merged production teacher, ordinary RGB student data and existing SD1.4 trainer/MSE selection.
- [x] Frozen-test analytic metrics, clean control, utility, panels and report; separate Python-only shell launcher.
- [x] Limited CPU verification using topmost; independent final review and regression fixes.

Review focus covered source mutation on staged resume, adapter identity, continuous PNDM history, skipped student updates, caption metadata and test leakage.

Validation evidence and GPU limitations: docs/validation/semantic_native_margin_v3.md.

User constraints override routine workflow: existing checkout; no GPU execution and no broad local test campaign. GPU integration gates run on server. No scientific redesign/sweeps. The source spec was the supplied design authority. Source paths must remain explicit; there is no automatic source-folder choice.
