#!/usr/bin/env bash
# Fetch CIFAR-100 test images for repro/rankme_vision.py.
#
# We use the fast.ai imageclas mirror (CIFAR-100 as PNG image folders) rather
# than the canonical https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz,
# because that host redirects to cave.cs.toronto.edu which is throttled to
# ~45 KB/s (a 169 MB download that stalls torchvision). The fast.ai images are
# pixel-identical CIFAR-100.
#
# Layout produced (100 fine classes live under 20 coarse superclasses):
#   data/cifar100/test/<superclass>/<fineclass>/*.png
#
# Usage:  bash repro/fetch_cifar100.sh [DATA_ROOT]   (default: data)
set -euo pipefail

DATA_ROOT="${1:-data}"
URL="https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz"

mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"
echo "Downloading CIFAR-100 (fast.ai mirror) ..."
curl -L -o cifar100_fastai.tgz "$URL"
echo "Extracting ..."
tar -xzf cifar100_fastai.tgz
echo "Done: $DATA_ROOT/cifar100/test ($(find cifar100/test -type f | wc -l) images)"
