# Oracle FP32/native drift: real source regression

## Evidence

User-provided `oracle_replay_debug.pt` SHA256:
`e9f49f0b87fcd9f21135dbaacdf026c0f1b9f24e27b39350f486eb8cac6a74c2`.
The bundle is not committed. It contains the first four selected IDs:
21, 24, 27 and 28, from the immutable historical source run.
Source Torch is 2.7.1+cu128; local reproduction is Torch 2.5.1+cpu with
the audited Diffusers 0.35.1 PNDMScheduler and the saved scheduler config.

ID27 reproduces the server failure at index `(0,1,39,0)`:

| Quantity | Value |
| --- | ---: |
| Cached native FP16 final latent | -2.015625 |
| FP32 replay | -2.021880865097046 |
| FP64 replay | -2.021880844278418 |
| Absolute FP32/native difference | 0.006255865097046 |
| Existing atol + rtol * abs(reference) | 0.0060312505811452866 |

There is exactly one failing element among 16384 for ID27. All four records
have finite FP32 results; IDs21,24,28 pass the old cross-precision comparison.
Maximum FP32/FP64 difference is below 8e-7 for every record. This supports
FP16/native rounding versus FP32 algebra as the source of the observed drift,
rather than insufficient precision of the FP32 computation.

CPU native-FP16 replay does not reproduce GPU source latents bitwise (maximum
differences 0.0078125 to 0.015625 across these four records). Thus CPU replay
is not used to certify the native GPU identity gate. In the supplied server
traceback, the earlier native exact/PNG/RSSF gates had passed before ID27's
FP32 comparison failed. End-to-end pretrained GPU execution remains untested
locally.

## Approved policy

The user explicitly approved keeping FP32 optimization, recording cross-dtype
drift as diagnostics and using a new output directory. This is a declared spec
amendment, not a tolerance increase. The protocol version is now
`rssf_tail_control_oracle_v2_fp32_drift`.

Native-dtype exact replay, canonical PNG, stored RSSF coordinate checks and
nonfinite failures remain fatal. FP32/native maximum error, RMS error,
source dtype and the number of elements outside the unchanged reference
atol/rtol are logged and saved per image, with aggregate diagnostics in JSON
and README. The FP32 graph has no correction or offset. Quality references,
optimizer, sample subset and PNG/LPIPS/CLIP verdict criteria are unchanged.

Use the new output `run_rssf_tail_control_oracle_20261026_v2`; old v1 identities
are incompatible and are not migrated. Teacher source artifacts are unchanged.

## Regression coverage

The small JSON fixture `fixtures/oracle_fp32_drift_id27.json` contains the four
channels at spatial pixel (39,0), cropped from the real record, with its native
scheduler state/config and source-bundle hash. PNDM is spatially elementwise,
so this reproduces the same failing scalar without storing the original bundle.
The test uses real scheduler steps, verifies FP32/FP64 agreement, and checks
finite nonzero gradients to each of the four residual controls.

Observed RED: the new diagnostic import/report fields were missing and
`begin_batch` aborted on finite cross-precision drift (3 failures, 4 guard
tests passed). Observed GREEN: all 40 oracle tests passed, including native,
PNG, RSSF-coordinate and nonfinite abort guards, unchanged original-image
reference, aggregate PNG verdict and interruption/resume coverage.

The full `test_semantic*.py` regression suite passed: **189 tests and 2
subtests**, 10 existing upstream warnings, 118.75 seconds. Python compilation
of the three changed runtime modules and `git diff --check` passed. These are
CPU checks; no complete pretrained oracle run on the HPC GPU is claimed.
