# RSSF v3 implementation verification — 2026-10-27

Specification: GPT_guide/rssf_native_margin_refinement_student_distill_codex_spec.md.

Implemented: separate versioned objective and launcher; strict v2 checkpoint-100 lineage; immutable source carrier/key/prompts/clean RGB data; current-weight live PNDM50 prefix plus ten differentiable calls; RGB8 STE; dynamic Top26/mu=.02; separate effective-batch gradients and .5 fidelity cap; fresh AdamW/resumable state; 256-validation screening and full-validation gate; ordinary RGB-caption SD1.4 LoRA with 7500 applied updates and MSE-only selection; clean control, frozen-test scoring, utility and fixed panels.

Local verification used D:/anaconda3/envs/topmost/python.exe only:

- Ten focused synthetic CPU unittest contracts: PASS (1.478 seconds test runtime).
- Syntax parsing of twelve new/modified Python files: PASS.
- All vendored file hashes agree with vendor/provenance.json: PASS.
- Shell byte content is exactly `python main_native_margin.py "$@"` plus LF: PASS.
- CLI help and two locked rank32 attention_conv student commands: PASS.
- Student command audit contains no carrier/key/detector/coordinate/oracle input: PASS.
- git diff --check: PASS (line-ending normalization advisory only).

Independent read-only reviewer found two important integrity defects: caption metadata was not audited, and direct refinement stage execution could bypass recorded source lineage. Both were fixed, with focused regression coverage. No other confirmed PNDM/gradient-cap/selection/applied-count defect was reported in static review.

Not executed locally: model loading, native numerical equivalence, GPU memory smoke, actual current-prefix perturbation and real ten-call tail-gradient checks, teacher/student optimization, generation or quality measurements. Real-model checks are mandatory in server preflight and abort on failure. The CPU checks establish software contracts, not successful scientific outcomes or GPU fit.

Engineering decisions: source run is always explicit; output defaults to a separate deterministic sibling directory; scientific configuration cannot change inside a resumed run; teacher zero-margin-gradient exception consumes one of 300 fixed batch events but performs no optimizer/weight-decay update, and applied/skipped totals are reported; microbatch 1 is allowed only as an explicit new-run setting; matched clean student is trained sequentially rather than reusing an unverified control. Historical objective/loss semantics remain unchanged; student applied-step and full-state resume changes require the new opt-in flag.

Pre-existing deleted root spec files and untracked GPT_guide/validation.json were left as found.
