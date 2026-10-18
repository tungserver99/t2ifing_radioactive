#!/usr/bin/env bash
set -euo pipefail
cd /root/workspace/serum
mkdir -p assets/serum_author logs
python3 -m venv .venv-download
.venv-download/bin/pip install 'gdown==5.2.0'
.venv-download/bin/gdown 12NTXeyIMFX1NWYzCJDeRX54T7BgEvXuQ -O assets/serum_author/latest_checkpoint.pt.part
mv assets/serum_author/latest_checkpoint.pt.part assets/serum_author/latest_checkpoint.pt
.venv-download/bin/gdown 17MFAZ2rsmaIlRNghzTbShLE73t6QCCrN -O assets/serum_author/sd21_original_cache.zip.part
mv assets/serum_author/sd21_original_cache.zip.part assets/serum_author/sd21_original_cache.zip
sha256sum assets/serum_author/latest_checkpoint.pt assets/serum_author/sd21_original_cache.zip > assets/serum_author/SHA256SUMS
echo 'AUTHOR_ASSETS_DOWNLOADED'
