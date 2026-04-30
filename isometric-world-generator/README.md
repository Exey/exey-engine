# isometric-world-generator

Demo binary for ExeyEngine. Procedurally generates a small isometric world,
lets you click characters and click destination tiles to walk via A*
pathfinding, and round-trips the map to/from Tiled `.tmx`.

## Run

From the workspace root:

```sh
../run.sh                 # default (bigbuffer)
../run.sh simple          # SimpleRenderer
../run.sh batch           # BatchRenderer
../run.sh bigbuffer       # BigBufferRenderer
../run.sh --debug         # validation layers on
```

Or directly with cargo:

```sh
cargo run --release -p isometric-world-generator -- --renderer bigbuffer
```

## Controls (planned by milestone)

| Milestone | Input                       | Action                              |
|-----------|-----------------------------|-------------------------------------|
| M1 ✅     | Esc / window close          | quit                                |
| M9        | `G`                         | regenerate random map (new seed)    |
| M9        | `Ctrl+S`                    | save map to `assets/last.tmx`       |
| M9        | `Ctrl+O`                    | load `assets/last.tmx`              |
| M10       | left-click character        | select unit                         |
| M10       | left-click walkable tile    | walk selected unit there via A*     |
| M4        | arrow keys / mouse drag     | pan camera                          |
| M4        | mouse wheel                 | zoom                                |

## Assets

Drop these PNGs into `assets/`:

- **scrabling 32×32 isometric tileset** (CC BY 4.0):
  https://scrabling.itch.io/pixel-isometric-tiles
- (optional, larger tile demo) **tipsy/isometric-tiles** 256×128 sample:
  https://github.com/tipsy/isometric-tiles

The license forbids us redistributing the Vituri TRPG pack; for other packs
we obey their terms. Treat `assets/` as gitignored content.
