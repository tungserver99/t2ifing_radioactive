"""Create compact review bundles for completed experiment results."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


MARKER = "generated_by_wmq_bundle"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def replace_generated(staging, destination):
    destination = Path(destination)
    marker = destination / "BUNDLE_MANIFEST.json"
    if destination.exists():
        if not marker.is_file() or not json.loads(marker.read_text(encoding="utf-8")).get(MARKER):
            raise FileExistsError(f"Refusing to replace non-generated directory: {destination}")
        shutil.rmtree(destination)
    staging.replace(destination)


def result_files(run):
    run = Path(run)
    files = [p for p in run.iterdir() if p.is_file() and p.suffix in (".json", ".jsonl", ".csv")]
    tradeoff = run / "tradeoff"
    if tradeoff.is_dir():
        files.extend(p for p in tradeoff.rglob("*") if p.is_file())
    branches = run / "branches"
    if branches.is_dir():
        for name in ("selection.json", "residual_calibration.json", "discriminator.json"):
            files.extend(branches.glob(f"*/{name}"))
    return sorted(set(files))


def bundle_result(run, destination=None):
    run = Path(run).resolve()
    if not (run / "manifest.json").is_file():
        raise FileNotFoundError(f"Not a completed/reportable run directory: {run}")
    destination = Path(destination).resolve() if destination else run / "result"
    if destination.parent != run:
        raise ValueError("Result bundle must be a direct child of its run directory")
    staging = Path(tempfile.mkdtemp(prefix=".result-review-", dir=run))
    try:
        records = []
        for source in result_files(run):
            relative = source.relative_to(run)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            records.append({"path": relative.as_posix(), "sha256": sha256(source), "bytes": source.stat().st_size})
        write_json(staging / "BUNDLE_MANIFEST.json", {MARKER: True, "kind": "result_review_bundle",
            "source_run": str(run), "files": records,
            "included": "Top-level reports/tables, tradeoff analysis, and compact per-branch selections.",
            "excluded": "Large checkpoints, images, and per-update branch logs remain at their canonical paths."})
        replace_generated(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    result = subparsers.add_parser("result")
    result.add_argument("--run", required=True, type=Path)
    result.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = bundle_result(args.run, args.output)
    print(output, flush=True)


if __name__ == "__main__":
    main()
