"""Download and verify the public Stable Signature owner-side extractor."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import urllib.request

from wmq_fixture_utils import fixture_lock, verify_sha256


EXTRACTOR_URL = "https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt"
# Audited 1,226,802-byte response from the official Meta URL on 2026-09-18.
EXPECTED_EXTRACTOR_SHA256 = "77cd0a2040b9391233bbcd79c1adf00816b196089cbb844da40035f854637a04"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    provenance = output.with_suffix(output.suffix + ".provenance.json")
    with fixture_lock(output):
        if output.exists():
            digest = verify_sha256(output, EXPECTED_EXTRACTOR_SHA256)
            print(f"Reusing verified owner extractor: {output} (sha256={digest})", flush=True)
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".part", dir=output.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            urllib.request.urlretrieve(EXTRACTOR_URL, temporary)
            digest = verify_sha256(temporary, EXPECTED_EXTRACTOR_SHA256)
            # Hard-link publication is atomic and fails if an external writer won the race.
            os.link(temporary, output)
            provenance.write_text(json.dumps({"url": EXTRACTOR_URL, "sha256": digest,
                "bytes": output.stat().st_size, "role": "owner-only post-freeze evaluator"}, indent=2),
                encoding="utf-8")
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Verified owner extractor saved: {output} (sha256={digest})", flush=True)


if __name__ == "__main__":
    main()
