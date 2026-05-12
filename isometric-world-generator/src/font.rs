//! Tiny embedded bitmap font for the on-screen FPS counter.
//!
//! M5 doesn't need a real text-rendering system (that's M8). It just
//! needs to draw a few digits and a handful of letters. So we hand-bake
//! a small 5×7-pixel monospace font and ship it as a const RGBA atlas.
//!
//! The atlas is a horizontal strip: each glyph is `GLYPH_W` pixels wide,
//! `GLYPH_H` tall, with `GLYPHS` characters laid out side by side. UV
//! sub-region for glyph `i` is:
//!     `uv_offset = (i / GLYPHS, 0)`
//!     `uv_scale  = (1 / GLYPHS, 1)`
//!
//! Pixels are encoded as bit rows in [`GLYPH_BITS`], one byte per row
//! (top to bottom). Bit 4 = leftmost pixel. The other 3 bits are unused.
//! [`build_atlas_rgba`] rasterises the bits to an RGBA buffer ready for
//! `Texture::from_rgba`.
//!
//! Supported chars (16 slots): "0123456789.: FPS-"

/// Glyph dimensions in pixels.
pub const GLYPH_W: u32 = 5;
pub const GLYPH_H: u32 = 7;
/// Number of glyph slots in the atlas.
pub const GLYPHS: u32 = 16;
/// Atlas dimensions.
pub const ATLAS_W: u32 = GLYPH_W * GLYPHS;
pub const ATLAS_H: u32 = GLYPH_H;

/// Character → glyph index. Anything not in this map renders as a
/// blank (returns `None`).
pub fn glyph_index(c: char) -> Option<u32> {
    match c {
        '0' => Some(0),
        '1' => Some(1),
        '2' => Some(2),
        '3' => Some(3),
        '4' => Some(4),
        '5' => Some(5),
        '6' => Some(6),
        '7' => Some(7),
        '8' => Some(8),
        '9' => Some(9),
        '.' => Some(10),
        ':' => Some(11),
        ' ' => Some(12),
        'F' | 'f' => Some(13),
        'P' | 'p' => Some(14),
        'S' | 's' => Some(15),
        _ => None,
    }
}

/// Bit rows for each glyph: 7 rows × 16 glyphs. Each byte encodes one
/// row, with bit 4 = leftmost pixel of a 5-wide glyph.
/// Layout: [glyph_idx][row].
#[rustfmt::skip]
const GLYPH_BITS: [[u8; 7]; 16] = [
    // 0
    [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    // 1
    [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    // 2
    [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
    // 3
    [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110],
    // 4
    [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    // 5
    [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
    // 6
    [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
    // 7
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    // 8
    [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
    // 9
    [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
    // .
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100],
    // :
    [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000],
    // SPACE
    [0b00000; 7],
    // F
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
    // P
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
    // S
    [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
];

/// Build the RGBA atlas: white opaque on font pixels, fully transparent
/// elsewhere. The renderer multiplies by the per-sprite tint, so the
/// text colour is set per draw, not baked in.
pub fn build_atlas_rgba() -> Vec<u8> {
    let mut rgba = vec![0u8; (ATLAS_W * ATLAS_H * 4) as usize];
    for g in 0..GLYPHS {
        for row in 0..GLYPH_H {
            let bits = GLYPH_BITS[g as usize][row as usize];
            for col in 0..GLYPH_W {
                // Bit 4 = leftmost pixel (col 0).
                let bit = (bits >> (GLYPH_W - 1 - col)) & 1;
                if bit != 0 {
                    let x = g * GLYPH_W + col;
                    let y = row;
                    let i = ((y * ATLAS_W + x) * 4) as usize;
                    rgba[i]     = 255;
                    rgba[i + 1] = 255;
                    rgba[i + 2] = 255;
                    rgba[i + 3] = 255;
                }
            }
        }
    }
    rgba
}

/// Layout state for emitting text as a series of sprites.
pub struct FontAtlas;

impl FontAtlas {
    /// Append sprites for `text` starting at `(x, y)` in world pixels.
    /// Each glyph is rendered at `(GLYPH_W * scale) × (GLYPH_H * scale)`
    /// pixel size. Unsupported characters are skipped (no advance).
    /// Returns the x-coord after the last glyph (for chaining).
    pub fn emit(
        out: &mut Vec<exey_engine::Sprite>,
        text: &str,
        x: f32,
        y: f32,
        scale: f32,
        tint: [f32; 4],
        mesh_idx: u8,
    ) -> f32 {
        let gw = GLYPH_W as f32 * scale;
        let gh = GLYPH_H as f32 * scale;
        // 1-pixel-equivalent advance between glyphs so they don't touch.
        let advance = gw + scale;
        let mut cur_x = x;
        for c in text.chars() {
            if let Some(idx) = glyph_index(c) {
                let mut s = exey_engine::Sprite::new(
                    [cur_x, y],
                    [gw, gh],
                    [0.0, 0.0],
                    tint,
                );
                s.uv_offset = [idx as f32 / GLYPHS as f32, 0.0];
                s.uv_scale = [1.0 / GLYPHS as f32, 1.0];
                s.mesh_idx = mesh_idx;
                out.push(s);
            }
            cur_x += advance;
        }
        cur_x
    }
}
