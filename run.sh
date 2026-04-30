#!/usr/bin/env bash
# run.sh — build + launch the IsometricWorldGenerator demo.
#
# Usage:
#   ./run.sh                       # bigbuffer renderer (default)
#   ./run.sh simple                # simple renderer (one draw per sprite)
#   ./run.sh batch                 # batch renderer
#   ./run.sh bigbuffer             # bigbuffer renderer (the algorithm)
#   ./run.sh --debug bigbuffer     # debug build with validation layers
#
# Env:
#   RUST_LOG=debug   verbose Vulkan validation messages
#   ASSETS_DIR=...   override default asset path

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="${ASSETS_DIR:-$ROOT/isometric-world-generator/assets}"

PROFILE_FLAG="--release"
RENDERER="bigbuffer"

while (( $# > 0 )); do
  case "$1" in
    --debug|-d)
      PROFILE_FLAG=""    # debug build — slower, but validation layers active
      shift ;;
    --release)
      PROFILE_FLAG="--release"
      shift ;;
    simple|batch|bigbuffer)
      RENDERER="$1"
      shift ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0 ;;
    *)
      echo "unknown arg: $1" >&2
      exit 1 ;;
  esac
done

# M9+: assets become required. Until then it just clears the screen, so we
# only warn and continue.
if [ ! -d "$ASSETS_DIR" ] || [ -z "$(ls -A "$ASSETS_DIR" 2>/dev/null | grep -v '^\.gitkeep$\|^README' || true)" ]; then
  echo "note: $ASSETS_DIR has no asset files yet."
  echo "      For later milestones, drop the scrabling 32×32 isometric tileset PNGs there."
  echo "      Pack: https://scrabling.itch.io/pixel-isometric-tiles  (CC BY 4.0)"
fi

cd "$ROOT"
echo "[run.sh] building ($PROFILE_FLAG)…"
cargo build $PROFILE_FLAG -p isometric-world-generator

echo "[run.sh] running with renderer=$RENDERER"
exec cargo run $PROFILE_FLAG -p isometric-world-generator -- --renderer "$RENDERER"
