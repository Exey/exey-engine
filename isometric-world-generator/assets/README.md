# assets/

Runtime assets for the demo. Most are not redistributable so this folder is
gitignored except for this README.

## What to download

| File / pack                                         | Source                                             | License   |
|-----------------------------------------------------|----------------------------------------------------|-----------|
| `isometric tileset.zip` (or extracted PNGs)         | https://scrabling.itch.io/pixel-isometric-tiles    | CC BY 4.0 |
| `IsometricTRPGAssetPack.zip` (optional, characters) | https://gvituri.itch.io/isometric-trpg             | custom (free, no redistribution) |
| `sample.tmx` (optional, big-tile demo)              | https://github.com/tipsy/isometric-tiles           | MIT       |
| `wolf-all.png` (M7 — animated wolves)               | user-provided spritesheet                          | (replace with your asset's licence) |

After download, unzip directly into this folder so the PNGs sit at e.g.
`assets/tiles.png`, `assets/spritesheet.png`, etc.

## M7 / wolf-all.png

The demo loads `assets/wolf-all.png` for the wolf characters. The layout is
assumed to be **15 columns × 16 rows of 64×64 cells** (the wolf sheet that
shipped with this milestone). Rows 9–12 are the 2-frame idle animation,
one row per facing (SW / SE / NW / NE).

If `wolf-all.png` is missing at runtime, the demo logs a warning and falls
back to a tiny procedural 2-frame silhouette so it always runs out-of-the-box.
The fallback paints only into row 9, so wolves at other facings will be
invisible until you drop in a real sheet.
