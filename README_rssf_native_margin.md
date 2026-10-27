# RSSF v3 native margin refinement

This campaign implements `GPT_guide/rssf_native_margin_refinement_student_distill_codex_spec.md` independently of the historical trajectory and oracle launchers.

Run from the repository root, in the existing GPU environment:

```bash
bash run_rssf_native_margin.sh --source-run /path/to/completed_v2_run --output /path/to/new_v3_run
```

The shell file contains exactly one command: `python main_native_margin.py "$@"`. It performs no installation, environment activation, directory change, or Python-path setup.

`--source-run` is mandatory: the program never guesses a source run. It verifies `teacher_fp/trajectory/checkpoint-100`, the source config/checkpoint hashes, exact adapter tensor/state identity, rank/targets, frozen carrier/key/prompts/splits, clean PNG hashes, clean model and PNDM protocol. A source validation split smaller than 256 aborts. No oracle data or residuals are used.

If the source teacher-only run has no student snapshot, the campaign uses the existing pinned SD1.4 asset downloader/cache. An existing verified pinned snapshot can be supplied with `--student-snapshot /path/to/sd14`. This is model asset preparation, not environment setup.

Default native microbatch is two, effective batch is four. The server runs the real-model clean-native equivalence, current-weight prefix and ten-call gradient checks, followed by a complete four-example/two-gradient smoke batch before training. For a GPU that cannot fit microbatch two, start a separate output with explicit `--native-micro-batch 1`. Configuration cannot change inside an existing immutable run.

Rerun the identical command after a PBS allocation ends. Committed teacher steps, PNG datasets, screening records and completed students are verified/reused. Teacher full state is committed every event; only the previous non-milestone internal state is pruned. Student full Accelerate states are saved every 250 applied updates. The opt-in trainer flag counts applied FP16 updates rather than overflow attempts; snapshots remain 2500/5000/7500. Ordinary student loss, optimizer, base model and adapter scope are shared with the existing trainer. Both marked and clean-data controls are run sequentially.

Teacher training follows the specification's explicit zero-gradient exception: exactly 300 batch events (1200 sample exposures); an event with margin-gradient norm <=1e-12 is logged/skipped, without AdamW decay or a fidelity-only update. Such skips do not extend the fixed event budget. Applied/skipped counts are recorded separately. Nonfinite values abort rather than consume an event silently.

Selection uses actual native FP16 PNGs on the first 256 source validation rows. Eligibility is TPR >=.90 and CLIP drop <=.01; choose lowest D26, paired pixel MSE, then earlier step. LPIPS is diagnostic only. The selected merged model must pass the same TPR/CLIP gate on every source validation row before downstream data/training starts. A failed gate saves diagnostics and stops the campaign.

The SD1.4 student sees only ordinary RGB/caption data and standard denoising training arguments. Student selection freezes minimum held-out denoising MSE (earlier step on exact ties), before any detector/utility/test scoring. Frozen test generation starts after both student selections. Owner-key metrics and wrong-key controls use saved actual-PNG coordinates and the unchanged RSSF verifier.

Outputs include `source_lineage.json`, runtime content hashes plus optional Git HEAD in `code_manifest.json`, `teacher_fp/native_margin_refine/`, `result/teacher_screening.json`, `result/teacher_full_validation.json`, `result/student_selection/`, `result/final_rssf.json`, `result/final_lpips.json`, `result/utility.json`, deterministic visual panels and `result/README.md`. Stage subprocess logs are in `logs/`; a subprocess error's traceback is in its corresponding log.

Local validation was deliberately limited to syntax and small synthetic CPU contracts in `test_semantic_native_margin.py` using the topmost environment. The actual-model CUDA integration gates have not been run on this CPU-only machine and must pass on the server before training.

PBS execution requires no Git executable and works without a `.git` directory. Commit provenance is read directly from available Git metadata; otherwise `git_head` is `null`. File hashes remain mandatory and enforce unchanged code on resume. After the reported missing-Git preparation failure, update the project files and resubmit the same PBS script/output directory; no run deletion or new environment setup is needed.

Source adapter scope is compared using PEFT target matching against actual clean UNet layer names, including suffix-compacted configs. Missing/extra layers still abort, and source tensor/state equality checks are unchanged. If a run failed before smoke/training, resubmitting after an initialization fix archives its old manifest/protocol under `code_updates/` and updates only the code identity. Any smoke proof, checkpoint, training loss, selection or downstream artifact prevents this automatic upgrade. Non-code protocol inputs must still match exactly.
