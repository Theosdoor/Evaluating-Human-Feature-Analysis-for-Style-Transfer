#!/usr/bin/env bash
set -euo pipefail

# External dependencies installer.
# Add new items with this pattern:
# 1) define URL/destination variables,
# 2) check whether the destination already exists,
# 3) download/clone only when missing or invalid.

# Resolve project root (parent of scripts/ directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_ROOT/models"
EXTERNAL_DIR="$PROJECT_ROOT/external"

mkdir -p "$MODELS_DIR" "$EXTERNAL_DIR"

download_file() {
  local url="$1"
  local dst="$2"
  local tmp_dst="${dst}.tmp"

  rm -f "$tmp_dst"
  if command -v wget >/dev/null 2>&1; then
    wget -O "$tmp_dst" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$tmp_dst"
  else
    echo "Error: neither wget nor curl is available." >&2
    exit 1
  fi
  mv "$tmp_dst" "$dst"
}

verify_sha256() {
  local file="$1"
  local expected="$2"
  if [[ -z "$expected" ]]; then
    return 0
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    echo "$expected  $file" | sha256sum -c - >/dev/null 2>&1
  elif command -v shasum >/dev/null 2>&1; then
    local actual
    actual="$(shasum -a 256 "$file" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]]
  else
    echo "Error: no sha256 tool found (need sha256sum or shasum)." >&2
    exit 1
  fi
}

# DINOv2 ViT-B/14 pretrained checkpoint
DINO_URL="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"
DINO_DST="$MODELS_DIR/dinov2_vitb14_reg4_pretrain.pt"
DINO_SHA256="${DINO_SHA256:-}"

need_dino_download=0

if [[ -f "$DINO_DST" ]]; then
  if [[ -n "$DINO_SHA256" ]]; then
    if verify_sha256 "$DINO_DST" "$DINO_SHA256"; then
      echo "[skip] Found verified $DINO_DST"
    else
      echo "[warn] Existing DINO checkpoint failed checksum, re-downloading"
      rm -f "$DINO_DST"
      need_dino_download=1
    fi
  else
    echo "[skip] Found $DINO_DST"
    echo "[note] Set DINO_SHA256 to enforce checksum verification"
  fi
else
  need_dino_download=1
fi

if [[ "$need_dino_download" -eq 1 ]]; then
  echo "[download] $DINO_URL -> $DINO_DST"
  download_file "$DINO_URL" "$DINO_DST"
  if [[ -n "$DINO_SHA256" ]]; then
    verify_sha256 "$DINO_DST" "$DINO_SHA256"
    echo "[ok] Checksum verified for $DINO_DST"
  fi
fi

# (forked & mildly modified) CUT repository
CUT_REPO_URL="https://github.com/Theosdoor/contrastive-unpaired-translation.git"
CUT_DST="$EXTERNAL_DIR/contrastive-unpaired-translation"

if [[ -d "$CUT_DST/.git" ]]; then
  echo "[skip] Found git repo at $CUT_DST"
elif [[ -e "$CUT_DST" ]]; then
  echo "Error: $CUT_DST exists but is not a git clone." >&2
  echo "Remove it manually (or move it) and re-run this script." >&2
  exit 1
else
  echo "[clone] $CUT_REPO_URL -> $CUT_DST"
  git clone --depth 1 "$CUT_REPO_URL" "$CUT_DST"
fi

echo "External setup complete."
