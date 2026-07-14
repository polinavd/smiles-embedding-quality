#!/usr/bin/env bash
# Fetch CIFAR-10 / CIFAR-100 test images for repro/rankme_vision.py.
#
# We use the fast.ai imageclas mirror (CIFAR as PNG image folders) rather than
# the canonical https://www.cs.toronto.edu/~kriz/ tarballs, because that host
# redirects to cave.cs.toronto.edu which is throttled to ~45 KB/s (a stalled
# download for torchvision). The fast.ai images are pixel-identical CIFAR.
#
# Layouts produced:
#   cifar10:  data/cifar10/test/<class>/*.png                   (10 flat classes)
#   cifar100: data/cifar100/test/<superclass>/<fineclass>/*.png (100 fine classes
#             under 20 coarse superclasses)
#
# Usage:  bash repro/fetch_cifar.sh {cifar10|cifar100} [DATA_ROOT]
#         (DATA_ROOT default: data)
set -euo pipefail

DATASET="${1:-cifar100}"
DATA_ROOT="${2:-data}"

case "$DATASET" in
  cifar10)  URL="https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz" ;;
  cifar100) URL="https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz" ;;
  *) echo "Unknown dataset '$DATASET' (expected cifar10 or cifar100)" >&2; exit 1 ;;
esac

mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"
echo "Downloading $DATASET (fast.ai mirror) ..."
curl -L -o "${DATASET}_fastai.tgz" "$URL"
echo "Extracting ..."
tar -xzf "${DATASET}_fastai.tgz"
echo "Done: $DATA_ROOT/$DATASET/test ($(find "$DATASET/test" -type f | wc -l) images)"
