# ExeyEngine

Rust + Vulkan port of an ActionScript 3 / Stage3D 2D isometric sprite engine
written by Exey Panteleev in 2014. A 2D isometric sprite renderer with
graph-topological depth sorting and a state-change-batched draw path.

This repo is a Cargo workspace with two crates:

| Crate                          | Role                                                                |
|--------------------------------|---------------------------------------------------------------------|
| [`exey-engine`](exey-engine)   | The engine. Vulkan-side rendering, sprite/animation, iso math, sorting. |
| [`isometric-world-generator`](isometric-world-generator) | A demo: random isometric world generation, click-to-walk pathfinding, save/load as Tiled `.tmx`. |

## Quick start

```sh
./run.sh                  # release build, BigBuffer renderer
./run.sh simple           # SimpleRenderer (one draw call per sprite, easy to debug)
./run.sh batch            # BatchRenderer (group by render-op identity)
./run.sh bigbuffer        # BigBufferRenderer (the algorithm, see below)
./run.sh --debug          # debug build with Vulkan validation layers
RUST_LOG=debug ./run.sh   # verbose validation output
```

You need the **Vulkan SDK** installed (LunarG). On Linux that's `vulkan-tools`
plus your GPU's Mesa or proprietary driver; on Windows install the SDK from
lunarg.com; on macOS, install MoltenVK.

The asset packs we use:

- **Demo tileset**: scrabling's [32×32 Pixel Isometric Tiles](https://scrabling.itch.io/pixel-isometric-tiles) (CC BY 4.0)
- The engine also supports the larger 256×128 tile sample from [tipsy/isometric-tiles](https://github.com/tipsy/isometric-tiles)
- Drop PNGs into `isometric-world-generator/assets/`. License terms forbid us
  from redistributing them in this repo.

---

## The architecture, in two paragraphs

The original engine layered into:
`ExeyEngineCore` (root) → `RenderCore` (3 layers: background/world/gui) →
pluggable `IRenderer` strategy + pluggable `ISorter` for the world layer.
The renderer/sorter split was the key idea: it let Exey try `SimpleRenderer`,
`BatchRenderer`, and `BigBufferRenderer` against `IsometricRectangleSorter` and
`ScreenYSorter` and pick winners based on actual measurements.

The Rust port keeps that exact shape. `Engine` owns `RenderCore`. `RenderCore`
holds a `Box<dyn IRenderer>` and a `Box<dyn ISorter>`. The `--renderer` flag
just constructs a different concrete renderer at startup. Everything else
— sprite, animation, frame, tile coordinates — is renderer-agnostic.

---

## Algorithm 1 — `BigBufferRenderer`

This is the centrepiece of the engine and the most interesting algorithm to
read. It addresses two costs that dominated AS3/Stage3D rendering and still
matter on Vulkan:

**Cost A — buffer uploads.** Re-uploading vertex/index buffers per sprite is
death by a thousand `vkCmdCopyBuffer`s. `BigBufferRenderer` packs every visible
sprite's vertices and indices into one giant streaming pair of buffers and
uploads each pair exactly once.

**Cost B — draw calls.** Even with one buffer, a naive renderer issues one
`vkCmdDrawIndexed` per sprite. `BigBufferRenderer` walks the (already iso-sorted)
list of sprites and only emits a draw when *something visible-state-affecting*
changes: texture, transform, camera, alpha, or blend mode. Identical state
just bumps the run length and keeps coalescing.

### The 65,536 boundary

Stage3D's index format is u16. That caps a single buffer at 65,535 indices.
`RenderBufferPair::populate()` watches `lastVertexIndex * 4` (each sprite is
4 vertices); when the next sprite would push it past 65,493 (= 65,535 − 4×4
slack) the pair "closes" and the renderer starts a new pair.

Vulkan doesn't enforce this — we'd be free to use `VK_INDEX_TYPE_UINT32` and
go bigger. The Rust port preserves the u16 cap on purpose:

1. it keeps the algorithm honest — an arbitrary scene, not a single megabuffer,
2. it matches the original's batching characteristics so timings transfer,
3. for sprite-grade workloads (10k–50k visible quads) one or two pairs covers
   everything and the per-pair overhead is invisible.

### The state-change loop

Inside one buffer pair, the per-frame inner loop is:

```text
for each render-op in iso-sorted order:
    if (texture | transform | camera | alpha | blend) changed:
        flush the current run with cmd_draw_indexed
        start a new run starting at this op's index
        re-bind whatever changed
    else:
        run_length += 6 (one quad = 2 triangles = 6 indices)
```

State changes drive the cost; the geometry is "free" because it was uploaded
in one shot. The original measured this against `BatchRenderer` (group by
render-op identity, draw each group separately) and `SimpleRenderer` (one draw
per sprite) — BigBuffer won by 5–10× on busy scenes.

You can switch to `Simple` or `Batch` from `run.sh` to see the exact same
scene rendered three ways. That's the engine's best learning tool.

---

## Algorithm 2 — `IsometricRectangleSorter`

Iso depth sorting is *not* "sort by Y." Two tiles at the same Y can occlude
each other depending on grid position; tall sprites overhang multiple tiles.
A naive sort will flicker.

The classical solution (which Exey credits to his earlier C++ work) is:

1. **Compute iso bounds for every sprite.** An iso bound is a rectangle
   `(isoX1, isoY1)–(isoX2, isoY2)` in world (logical) tile coordinates.
   `isoLeft  = isoX1 − isoY2` and `isoRight = isoX2 − isoY1` are the screen-
   space-x extremes after iso projection — they tell us which sprites can
   *possibly* overlap on the X axis.
2. **Sort by `isoLeft`** (left-to-right scanline order).
3. **Sweep.** Maintain two ordered lists of "currently overlapping" sprites:
   one ordered by iso depth, one ordered by `isoRight`. When a new sprite
   arrives, drop everything from both lists whose `isoRight` is now to the
   left of the newcomer (they no longer overlap on X). The sprites that
   *do* still overlap are exactly those that need a depth-order edge.
4. **Build a graph.** For each newcomer, look up its insertion position in
   the depth-ordered list and add an edge `prev → newcomer` and
   `newcomer → next`. The edge says "prev must draw before newcomer."
5. **Topological sort.** Standard topo sort over the graph yields the draw
   order. Because we only made edges between *actually overlapping* sprites,
   the graph is sparse and the topo sort is fast.

Result: O(n log n) average for the scanline + O(V+E) for the topo sort, with
E proportional to actual visual overlap rather than n². And — crucially — it
produces a *correct* order even when sprites have non-cubic iso bounds (which
the Vituri TRPG pack does: characters are 2 tiles tall, decorations
overhang).

The trade-off: `IsometricRectangleSorter` allocates more than `ScreenYSorter`,
which is just `Vec::sort_by_key(|s| s.y)`. For a flat ground-only map, ScreenY
is fine. For everything else, use the topological sorter.

---

## Status

This document covers M1 (window + clear color + FPS + renderer-selection CLI),
M2 (one textured quad), M3 (bouncing flock of textured sprites), M4
(`IsometricCamera2D` + iso grid), and M5 (`IsometricRectangleSorter` —
graph-topological depth sort over iso bounds, with mouse pan + keyboard
zoom + in-window bitmap-font FPS overlay). See [`PLAN.md`](PLAN.md) for
the milestone roadmap.

## Credits

- Original ExeyEngine — **Exey Panteleev**, 2014.
- Vulkan via [`vulkanalia`](https://github.com/KyleMayes/vulkanalia) (Kyle Mayes).
- Iso depth-sort algorithm originally from Exey's earlier C++ work.
- Demo asset pack: scrabling, CC BY 4.0.
- Tiled `.tmx` file format: [mapeditor.org](https://www.mapeditor.org/).
