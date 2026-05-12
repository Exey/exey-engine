# Milestone roadmap

Each milestone produces a runnable binary. We build incrementally; nothing
breaks the build between milestones.

| #   | Goal                                | Engine work                                                                 | Demo work                                          | Status |
|-----|-------------------------------------|------------------------------------------------------------------------------|----------------------------------------------------|--------|
| M1  | Window + clear color + FPS          | instance, device, swapchain, dynamic-rendering clear, frame sync             | runnable demo, `--renderer` flag, FPS in title    | ✅ done |
| M2  | One textured quad on screen         | sprite pipeline, image upload, descriptors, GLSL→SPIR-V build step           | hardcoded tile draw                                | ✅ done |
| M3  | Bouncing flock of textured sprites  | shared mesh + descriptor; per-sprite world push constants; renderer issues N draws | flock of 32 sprites bouncing off framebuffer edges | ✅ done |
| M4  | `IsometricCamera2D` + iso math      | iso↔screen, world↔logic, ortho projection                                    | tiles laid out in iso projection                   | ✅ done |
| M5  | `IsometricRectangleSorter` ★        | iso bounds, scanline + sorted lists, topological sort                        | depth correct with overlapping characters/decor    | ✅ done |
| M6  | `BigBufferRenderer` ★               | 65k-cap streaming + state-change batching                                    | same scene, ~1 draw call                           |       |
| M7  | `Animation2D`                        | frame manager, time-driven advance                                           | 2-frame idle on characters                         |       |
| M8  | TMX loader + writer                 | quick-xml + base64 + zlib roundtrip                                          | save/load `.tmx` with Tiled, openable in mapeditor |       |
| M9  | Map generator                        | —                                                                            | random rooms+bridges, seedable, regen on key press |       |
| M10 | A* pathfinding + click-to-walk      | input mapping, 4-conn A*                                                     | select character, click tile, walk path with indicators |       |

## What "★" means

M5 and M6 are the algorithmic centrepieces of the engine — the ones the README
documents at length. They deserve careful, isolated implementation. If we
hit time/context pressure I'd rather pause M6 and pick it up in a fresh
session than rush it.

## Out of scope (for now)

- 3D rendering — original was 2D-iso, this stays 2D-iso.
- GUI framework — the AS3 engine had `Button2D`/`Container2D`/`StyledContainer2D`;
  the demo doesn't need them so we ship without.
- Audio.
- Networking.
- Mobile / web targets.

These are noted because the original engine had them or hinted at them.
Re-adding any is a follow-up project, not a milestone.
