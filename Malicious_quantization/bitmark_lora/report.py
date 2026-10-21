"""Frozen calibration and paired, clean-adjusted radioactivity reporting."""
import argparse
import csv
import math
import random
import statistics
from pathlib import Path

from common import finite, freeze, image_name, read_json, sha256, write_json
from student import branches


def calibrate(values, fpr):
    finite(values)
    if not 0 < fpr < 1:
        raise ValueError("FPR must lie in (0,1)")
    allowed = math.floor(len(values) * fpr)
    # Strictly above the (allowed+1)-th highest score; ties cannot inflate FPR.
    return math.nextafter(sorted(values, reverse=True)[allowed], math.inf)


def wilson(k, n):
    if not n or not 0 <= k <= n:
        raise ValueError("Invalid detection counts")
    z = 1.959963984540054
    p = k / n
    c = (p + z*z/(2*n)) / (1+z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [max(0., c-half), min(1., c+half)]


def paired_ci(differences, seed=0, repeats=2000):
    finite(differences)
    rng = random.Random(seed)
    n = len(differences)
    samples = sorted(sum(differences[rng.randrange(n)] for _ in range(n))/n for _ in range(repeats))
    return [samples[int(.025*repeats)], samples[min(repeats-1, int(.975*repeats))]]


def detector_value(row, convention):
    if convention=='analytic_binomial':
        return row['K']
    if convention.startswith(("serum_score", "semantic_score")):
        return row["score"]
    return abs(row["z_score"]) if convention == "absolute_z" else row["z_score"]


def conventions(threshold):
    return ("serum_score", "serum_score_student_calibrated") if threshold.get("primary") == "serum_score" else ("absolute_z", "signed_z")


def scores(run, label, n):
    rows = read_json(run / "scores" / f"{label}.json")
    if len(rows) != n or [r["file"] for r in rows] != [image_name(i) for i in range(n)]:
        raise ValueError(f"Incomplete/unaligned score rows: {label}")
    if read_json(run / "config.json").get("watermark_method") == "semantic_fp":
        from semantic_report import score_rows
        return score_rows(run, label, n)
    key = "score" if read_json(run / "config.json").get("watermark_method") == "serum" else "z_score"
    values = [r[key] for r in rows]
    finite(values)
    for r in rows:
        if sha256(run / "images" / label / r["file"]) != r["sha256"]:
            raise RuntimeError(f"Scored image changed: {label}/{r['file']}")
    return rows


def freeze_threshold(run):
    if read_json(run / "config.json").get("watermark_method") == "semantic_fp":
        from semantic_report import freeze_thresholds
        return freeze_thresholds(run)
    c = read_json(run / "config.json")
    rows = scores(run, "teacher_clean_calibration", c["calibration_n"])
    if c.get("watermark_method") == "serum":
        student_rows = scores(run, "unadapted_calibration", c["calibration_n"])
        result = {"source": "disjoint clean SERUM SD2.1 teacher calibration; no test data",
                  "calibration_score_sha256": sha256(run / "scores/teacher_clean_calibration.json"),
                  "student_calibration_score_sha256": sha256(run / "scores/unadapted_calibration.json"),
                  "student_source": "unadapted SD1.4 on held-out calibration prompts, no adaptation/test data",
                  "n": len(rows), "target_fpr": c["fpr"], "comparison": ">= threshold",
                  "serum_score": calibrate([r["score"] for r in rows], c["fpr"]),
                  "serum_score_student_calibrated": calibrate([r["score"] for r in student_rows], c["fpr"]),
                  "primary": "serum_score", "note": "Author learned detector probability; no message bit accuracy. Secondary threshold accounts for clean student domain shift."}
    else:
        z = [r["z_score"] for r in rows]
        result = {"source": "disjoint clean Infinity calibration; never student test data",
                  "calibration_score_sha256": sha256(run / "scores/teacher_clean_calibration.json"),
                  "n": len(z), "target_fpr": c["fpr"], "comparison": ">= threshold",
                  "absolute_z": calibrate([abs(v) for v in z], c["fpr"]),
                  "signed_z": calibrate(z, c["fpr"]),
                  "primary": "absolute_z", "note": "Paper describes |z|; upstream ROC script uses signed z. Both reported."}
    freeze(run / "result/threshold.json", result)
    if (run / "scores/teacher_wm_test.json").exists():
        reference = scores(run, "teacher_wm_test", c["test_n"])
        detected = sum(detector_value(r, result["primary"]) >= result[result["primary"]] for r in reference)
        rate = detected / len(reference)
        write_json(run / "result/teacher_validation.json", {
            "detected": detected, "n": len(reference), "detection_rate": rate,
            "warning": None if rate >= .9 else "Weak positive control; continue and report, do not attribute low transfer to LoRA."})
        print(f"Teacher positive control: {detected}/{len(reference)} detected ({rate:.2%}); proceeding with audit.", flush=True)
    return result


def summarize(run):
    if read_json(run / "config.json").get("watermark_method") == "semantic_fp":
        from semantic_report import summarize as semantic_summary
        return semantic_summary(run)
    c = read_json(run / "config.json")
    has_clean_controls = c.get("train_clean_controls", True)
    threshold = freeze_threshold(run)
    labels = ["teacher_clean_test", "teacher_wm_test", "unadapted"] + [b["name"] for b in branches(c)]
    rows_by_label = {label: scores(run, label, c["test_n"]) for label in labels}
    tables, effects = {}, {}
    for convention in conventions(threshold):
        table, indicators = [], {}
        for label in labels:
            rows = rows_by_label[label]
            values = [detector_value(r, convention) for r in rows]
            detected = [int(v >= threshold[convention]) for v in values]
            indicators[label] = detected
            count = sum(detected)
            table.append({"branch": label, "detected": count, "n": len(rows), "detection_rate": count/len(rows),
                          "ci95_low": wilson(count, len(rows))[0], "ci95_high": wilson(count, len(rows))[1],
                          **({"mean_score": statistics.mean(r["score"] for r in rows)} if c.get("watermark_method") == "serum" else
                             {"mean_z": statistics.mean(r["z_score"] for r in rows),
                              "mean_green_fraction": statistics.mean(r["green_fraction"] for r in rows)})})
        comparisons = {}
        def effect(name, diff):
            comparisons[name] = {"estimate_pp": 100*statistics.mean(diff),
                                 "paired_bootstrap_ci95_pp": [100*v for v in paired_ci(diff, c["seed"])],
                                 "ci_scope": "paired image sampling, conditional on this one training seed and threshold"}
        if not c["lora_only"]:
            full_reference = "full_clean" if has_clean_controls else "unadapted"
            full = [a-b for a, b in zip(indicators["full_wm"], indicators[full_reference])]
            effect("full_clean_adjusted_transfer" if has_clean_controls else "full_wm_minus_unadapted", full)
        for b in branches(c):
            if b["kind"] != "lora" or b["condition"] != "wm":
                continue
            prefix = b["name"].removesuffix("_wm")
            lora_reference = prefix + "_clean" if has_clean_controls else "unadapted"
            lora = [a-b for a, b in zip(indicators[b["name"]], indicators[lora_reference])]
            effect(prefix + ("_clean_adjusted_transfer" if has_clean_controls else "_wm_minus_unadapted"), lora)
            if not c["lora_only"]:
                effect(f"full_minus_{prefix}" + ("_clean_adjusted" if has_clean_controls else ""),
                       [a-b for a, b in zip(full, lora)])
        for rank in c["ranks"]:
            conv, attention = f"lora_r{rank}_conv_wm", f"lora_r{rank}_wm"
            if conv in indicators and attention in indicators:
                effect(f"conv_minus_attention_r{rank}", [a-b for a,b in zip(indicators[conv], indicators[attention])])
        for b in branches(c):
            if b.get("use_dora") and b["condition"] == "wm":
                baseline = f"lora_r{b['rank']}_conv_wm"
                if baseline in indicators:
                    effect(f"dora_minus_lora_conv_r{b['rank']}",
                           [a-bb for a,bb in zip(indicators[b['name']], indicators[baseline])])
        tables[convention], effects[convention] = table, comparisons
    warnings = ["Single training seed; image bootstrap does not capture training-seed variance.",
                "Matched full controls have no EMA, matching LoRA. Not an exact reproduction of the paper's full+EMA run.",
                "Clean calibration uses generated Infinity images, not natural COCO images; inspect clean student false-positive rates.",
                "Green fraction is NOT message bit accuracy: BitMark tests bit-pattern bias."]
    if c.get("watermark_method") == "serum":
        warnings = [warnings[0],
                    "Teacher SD2.1 and student SD1.4 differ; this is a new capacity/scope experiment, not the BitMark setup.",
                    "Full control uses a separately declared learning rate; parameter count and compute are not matched to LoRA.",
                    "Teacher and clean student thresholds are both frozen on calibration, never test; compare within one threshold convention.",
                    "SERUM scores are learned watermark probabilities, NOT decoded bit accuracy."]
    if c.get("dora"):
        warnings.append("DoRA is an exploratory magnitude/direction adapter, not an established radioactivity improvement. Same steps/LR/scope as LoRA do not equalize FLOPs or parameter count.")
        if not any(b["name"] == f"lora_r{max(c['ranks'])}_conv_wm" for b in branches(c)):
            warnings.append("No same-rank convolution LoRA comparator for DoRA in this run.")
    primary_key = threshold["primary"]
    primary = {r["branch"]: r for r in tables[primary_key]}
    full_effect = "full_clean_adjusted_transfer" if has_clean_controls else "full_wm_minus_unadapted"
    if not has_clean_controls:
        warnings.append("No clean-trained controls: contrasts use the unadapted baseline and cannot isolate generic fine-tuning effects.")
    if primary["teacher_wm_test"]["detection_rate"] < .9:
        warnings.append("Teacher positive control is weak: investigate generation/detection before attributing weak transfer to LoRA.")
    if c["lora_only"]:
        warnings.append("No matched full controls: cannot attribute low transfer specifically to LoRA.")
    elif effects[primary_key][full_effect]["paired_bootstrap_ci95_pp"][0] <= 0:
        warnings.append("Matched full positive control has not established transfer; LoRA-weakness hypothesis remains inconclusive.")
    for label in ["unadapted"] + [b["name"] for b in branches(c) if b["condition"] == "clean"]:
        if primary[label]["ci95_low"] > c["fpr"]:
            warnings.append(f"{label}: clean detection exceeds nominal FPR; raw TPR alone is insufficient evidence of transfer.")
    if c.get("watermark_method") == "serum":
        secondary = {r["branch"]: r for r in tables["serum_score_student_calibrated"]}
        if secondary["unadapted"]["ci95_low"] > c["fpr"]:
            warnings.append("Clean-student calibrated threshold still exceeds target FPR on independent clean test; inspect domain/generalization before interpreting detection.")
        if threshold["serum_score"] > 1 or threshold["serum_score_student_calibrated"] > 1:
            warnings.append("SERUM calibration scores saturate at 1: a frozen threshold is unattainable. Zero detection under that convention does not demonstrate absent radioactivity.")
    utility = read_json(run / "result/utility.json") if (run / "result/utility.json").exists() else None
    if not utility:
        warnings.append("Utility metrics unavailable; no image-quality preservation claim can be made.")
    selections = {}
    if c.get("checkpoint_steps"):
        for b in branches(c):
            selections[b["name"]] = read_json(run / "result/selection" / f"{b['name']}.json")
        warnings.append("Checkpoint selected by held-out denoising MSE; validation detector/utility curves are descriptive only.")
        warnings.append("Ranks/scopes share step budgets, not equal trainable parameters or FLOPs; inspect adapter_audit.json.")
    result = {"threshold": threshold, "tables": tables, "paired_effects": effects, "warnings": warnings,
              "utility": utility, "selection": ("minimum held-out denoising MSE, earlier-step tie break" if selections
                                                  else "fixed final epoch, no detector-guided selection"),
              "selected_checkpoints": selections,
              "config": c, "source_sha256": sha256(Path(__file__).with_name("vendor") / "provenance.json")}
    write_json(run / "result/radioactivity.json", result)
    for convention, table in tables.items():
        with (run / f"result/detection_{convention}.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    title = "SERUM SD2.1 → SD1.4: LoRA radioactivity" if c.get("watermark_method") == "serum" else "BitMark → SD2.1: LoRA radioactivity"
    lines = ["# " + title, "", f"Primary metric: {primary_key} at frozen {100*c['fpr']:g}% target teacher FPR.", "",
             "| Branch | Detected | Rate | 95% Wilson CI |", "|---|---:|---:|---:|"]
    for r in tables[primary_key]:
        lines.append(f"| {r['branch']} | {r['detected']}/{r['n']} | {100*r['detection_rate']:.2f}% | {100*r['ci95_low']:.2f}–{100*r['ci95_high']:.2f}% |")
    if c.get("watermark_method") == "serum":
        lines += ["", "## Secondary: frozen clean-student calibration", "",
                  "| Branch | Detected | Rate | 95% Wilson CI |", "|---|---:|---:|---:|"]
        for r in tables["serum_score_student_calibrated"]:
            lines.append(f"| {r['branch']} | {r['detected']}/{r['n']} | {100*r['detection_rate']:.2f}% | {100*r['ci95_low']:.2f}–{100*r['ci95_high']:.2f}% |")
    if selections:
        lines += ["", "## Validation-selected checkpoints", "", "See checkpoint_metrics.csv for all milestones; test images never select checkpoints.", ""]
        for name, selected in selections.items():
            lines.append(f"- {name}: step {selected['selected_step']}; held-out MSE {selected['denoising_mse']:.8f}.")
    interpretation = ("Positive full-minus-LoRA interaction means stronger transfer under full adaptation after subtracting clean controls."
                      if has_clean_controls else
                      "Full-minus-LoRA is the paired detection-rate difference on watermarked-trained students; no clean-trained adjustment is available.")
    lines += ["", interpretation, ""]
    for name, e in effects[primary_key].items():
        low, high = e["paired_bootstrap_ci95_pp"]
        lines.append(f"- {name}: {e['estimate_pp']:.2f} pp; paired CI [{low:.2f}, {high:.2f}] pp.")
    lines += ["", "## Interpretation limits", ""] + [f"- {w}" for w in warnings]
    (run / "result/README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--calibrate-only", action="store_true")
    a = p.parse_args()
    (freeze_threshold if a.calibrate_only else summarize)(a.run.resolve())
