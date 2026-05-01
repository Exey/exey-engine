#!/usr/bin/env bash
# compile_shaders.sh — rebuild the SPIR-V blobs from GLSL sources.
#
# We commit the .spv files to the repo and `include_bytes!` them at engine
# load time, so a vanilla `cargo build` does not require glslang/glslc to
# be installed. Run this script after editing any `*.vert` / `*.frag` file
# under `exey-engine/shaders/`. Commit the changed `.spv` alongside.
#
# Tries (in order): glslang -V, glslangValidator -V, glslc.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/exey-engine/shaders"
OUT="$ROOT/exey-engine/shaders/spv"

mkdir -p "$OUT"

if   command -v glslang          >/dev/null 2>&1; then COMPILER="glslang -V"
elif command -v glslangValidator >/dev/null 2>&1; then COMPILER="glslangValidator -V"
elif command -v glslc            >/dev/null 2>&1; then COMPILER="glslc"
else
    echo "error: no shader compiler found." >&2
    echo "       install one of: glslang-tools / vulkan-tools (Linux apt)," >&2
    echo "       glslang (Homebrew on macOS), or the LunarG Vulkan SDK (any OS)." >&2
    exit 1
fi
echo "using: $COMPILER"

stage_for() {
    case "$1" in
        *.vert) echo vert ;;
        *.frag) echo frag ;;
        *.comp) echo comp ;;
        *) echo "unknown stage for $1" >&2; exit 1 ;;
    esac
}

shopt -s nullglob
for src in "$SRC"/*.vert "$SRC"/*.frag "$SRC"/*.comp; do
    name="$(basename "$src")"
    dst="$OUT/${name}.spv"
    stage="$(stage_for "$src")"
    case "$COMPILER" in
        glslc) $COMPILER -fshader-stage="$stage" "$src" -o "$dst" ;;
        *)     $COMPILER -S "$stage" "$src" -o "$dst" ;;
    esac
    echo "  $name  ->  ${dst#$ROOT/}"
done
echo "done."
