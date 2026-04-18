//! Runtime BC1 / BC3 / BC4 encoding for glTF textures.
//!
//! Compiled only when the `bc-encode` cargo feature is enabled.
//! Produces [`TextureData`] values whose `format` is one of the BC
//! variants, ready for `blinc_gpu::GpuImage::from_compressed`.
//!
//! Per-slot format picks (chosen elsewhere, see `material::parse_material`):
//!
//! - **Diffuse / base color** — BC1 when alpha ≥ 0.95 everywhere
//!   (the same "Opaque" profile the demotion heuristic already
//!   computes), BC3 when alpha is used. BC1 is 4 bpp, BC3 is 8 bpp;
//!   both have sRGB variants in wgpu.
//! - **Occlusion** — BC4 on the red channel. 4 bpp, linear only.
//!
//! Normal maps (BC5) and metallic-roughness (BC5) are not yet
//! encoded — the `tbc` crate doesn't ship a BC5 encoder. Adding a
//! pure-Rust BC5 (two BC4 halves on R + G) is ~60 LoC of follow-up.
//!
//! Encoder cost at load time is ~50-150 ms per 2K × 2K texture;
//! runs on whatever thread the caller invokes from. Async asset
//! pipelines (see `blinc_app_examples/examples/strangler_demo.rs`)
//! already spawn the scene load off-thread, so the cost is
//! invisible at the frame rate.

use blinc_core::draw::{TextureData, TexturePixelFormat};

/// Zero-copy reinterpret a packed `[r, g, b, a, r, g, b, a, ...]`
/// u8 slice as `&[tbc::color::Rgba8]`.
///
/// Safety: `tbc::color::Rgba8` is `#[repr(C)]` with exactly four
/// `u8` fields — `(r, g, b, a)` — so its layout is identical to
/// `[u8; 4]`. Alignment of `Rgba8` is 1 (same as `u8`). The byte
/// count must be a multiple of 4 for the slice length math to
/// land on a whole number of pixels; the caller-facing functions
/// `debug_assert` that invariant before this helper runs.
///
/// Cutting the `Vec<Rgba8>` allocation saves an extra N × 4 bytes
/// of transient memory per encode — on the strangler rig (12
/// diffuse × 16 MB) that's ~176 MB off the load-time peak.
#[inline]
fn rgba8_view(pixels: &[u8]) -> &[tbc::color::Rgba8] {
    debug_assert_eq!(pixels.len() % 4, 0);
    // SAFETY: see doc comment above. `#[repr(C)]` struct with four
    // `u8` fields has the same memory layout as `[u8; 4]`.
    unsafe {
        std::slice::from_raw_parts(
            pixels.as_ptr() as *const tbc::color::Rgba8,
            pixels.len() / 4,
        )
    }
}

/// Encode an RGBA8 buffer as BC1 (4 bpp, sRGB-decoded on sample).
/// The alpha channel is ignored — caller must have already
/// validated that `pixels` is effectively opaque.
///
/// `width` and `height` must both be positive. Non-multiple-of-4
/// dimensions get padded to the next block boundary by the
/// encoder; the reported `width`/`height` on the returned
/// `TextureData` stays at the original values so the sampler uses
/// them verbatim.
pub(crate) fn encode_bc1(pixels: &[u8], width: u32, height: u32) -> TextureData {
    debug_assert_eq!(pixels.len(), (width as usize) * (height as usize) * 4);
    let bytes = tbc::encode_image_bc1_conv_u8(rgba8_view(pixels), width as usize, height as usize);
    tracing::debug!(
        format = "Bc1",
        rgba8_bytes = pixels.len(),
        bc_bytes = bytes.len(),
        width,
        height,
        "encoded texture"
    );
    TextureData::new_compressed(bytes, TexturePixelFormat::Bc1, width, height)
}

/// Encode an RGBA8 buffer as BC3 (8 bpp, sRGB-decoded on sample).
/// Preserves the alpha channel. Use for diffuse textures with
/// authored transparency.
pub(crate) fn encode_bc3(pixels: &[u8], width: u32, height: u32) -> TextureData {
    debug_assert_eq!(pixels.len(), (width as usize) * (height as usize) * 4);
    let bytes = tbc::encode_image_bc3_conv_u8(rgba8_view(pixels), width as usize, height as usize);
    tracing::debug!(
        format = "Bc3",
        rgba8_bytes = pixels.len(),
        bc_bytes = bytes.len(),
        width,
        height,
        "encoded texture"
    );
    TextureData::new_compressed(bytes, TexturePixelFormat::Bc3, width, height)
}

/// Encode the **red channel** of an RGBA8 buffer as BC4 (4 bpp,
/// linear). The G / B / A channels are discarded. Use for
/// occlusion maps (glTF convention: AO in `.r`).
pub(crate) fn encode_bc4_red(pixels: &[u8], width: u32, height: u32) -> TextureData {
    debug_assert_eq!(pixels.len(), (width as usize) * (height as usize) * 4);
    // `tbc::encode_image_bc4_r8_conv_u8` wants `&[Red8]`, a
    // newtype around one `u8`. Can't zero-copy from the RGBA stride
    // (different element size); allocate a tight `Vec<Red8>` of
    // length N — this is width*height bytes, same as the final BC4
    // output size.
    let red: Vec<tbc::color::Red8> = pixels
        .chunks_exact(4)
        .map(|c| tbc::color::Red8 { red: c[0] })
        .collect();
    let bytes = tbc::encode_image_bc4_r8_conv_u8(&red, width as usize, height as usize);
    tracing::debug!(
        format = "Bc4",
        rgba8_bytes = pixels.len(),
        bc_bytes = bytes.len(),
        width,
        height,
        "encoded texture"
    );
    TextureData::new_compressed(bytes, TexturePixelFormat::Bc4, width, height)
}

/// Encode the **red + green channels** of an RGBA8 buffer as BC5
/// (8 bpp, linear). The B / A channels are discarded. Use for
/// tangent-space normal maps — the shader reconstructs the Z (B)
/// component from the RG pair (`b = sqrt(1 - r² - g²)`).
///
/// Produces the canonical BC5 block layout: two 8-byte BC4 halves
/// per 4×4 block, R followed by G. Matches wgpu's `Bc5RgUnorm`.
pub(crate) fn encode_bc5_rg(pixels: &[u8], width: u32, height: u32) -> TextureData {
    debug_assert_eq!(pixels.len(), (width as usize) * (height as usize) * 4);
    let rg: Vec<tbc::color::RedGreen8> = pixels
        .chunks_exact(4)
        .map(|c| tbc::color::RedGreen8 {
            red: c[0],
            green: c[1],
        })
        .collect();
    let bytes = tbc::encode_image_bc4_rg8_conv_u8(&rg, width as usize, height as usize);
    tracing::debug!(
        format = "Bc5",
        rgba8_bytes = pixels.len(),
        bc_bytes = bytes.len(),
        width,
        height,
        "encoded texture"
    );
    TextureData::new_compressed(bytes, TexturePixelFormat::Bc5, width, height)
}
