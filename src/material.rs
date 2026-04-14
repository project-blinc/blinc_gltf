//! glTF material → [`blinc_core::draw::Material`] mapping.
//!
//! Covers the full `pbrMetallicRoughness` block plus standard texture
//! slots (base color, normal, metallic-roughness, emissive, occlusion)
//! and the `KHR_materials_unlit` extension. Alpha mode maps directly;
//! the glTF `alphaCutoff` value for `MASK` mode isn't yet threaded into
//! `blinc_core::draw::Material` (it assumes the shader's default 0.5
//! cutoff).

use blinc_core::draw::{AlphaMode, Material, TextureData};

/// Decode a glTF material into Blinc's `Material`. Texture slots read
/// from the `images` array; missing textures fall back to the scalar
/// factors alone.
pub fn parse_material(mat: &gltf::Material, images: &[gltf::image::Data]) -> Material {
    let pbr = mat.pbr_metallic_roughness();

    let base_color_texture = pbr
        .base_color_texture()
        .and_then(|info| image_to_texture(&images[info.texture().source().index()]));
    let metallic_roughness_texture = pbr
        .metallic_roughness_texture()
        .and_then(|info| image_to_texture(&images[info.texture().source().index()]));
    let normal_info = mat.normal_texture();
    let normal_map = normal_info
        .as_ref()
        .and_then(|info| image_to_texture(&images[info.texture().source().index()]));
    let normal_scale = normal_info.as_ref().map(|n| n.scale()).unwrap_or(1.0);
    let occlusion_info = mat.occlusion_texture();
    let occlusion_texture = occlusion_info
        .as_ref()
        .and_then(|info| image_to_texture(&images[info.texture().source().index()]));
    let occlusion_strength = occlusion_info
        .as_ref()
        .map(|o| o.strength())
        .unwrap_or(1.0);
    let emissive_texture = mat
        .emissive_texture()
        .and_then(|info| image_to_texture(&images[info.texture().source().index()]));

    Material {
        base_color: pbr.base_color_factor(),
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        emissive: mat.emissive_factor(),
        base_color_texture,
        normal_map,
        normal_scale,
        metallic_roughness_texture,
        emissive_texture,
        occlusion_texture,
        occlusion_strength,
        // Displacement mapping has no glTF core analog; extension
        // support would go through `KHR_materials_displacement` (no
        // standard at the time of writing).
        displacement_map: None,
        displacement_scale: 0.05,
        unlit: mat.unlit(),
        alpha_mode: match mat.alpha_mode() {
            gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
            gltf::material::AlphaMode::Mask => AlphaMode::Mask,
            gltf::material::AlphaMode::Blend => AlphaMode::Blend,
        },
        receives_shadows: true,
        casts_shadows: true,
    }
}

/// Convert a decoded glTF image into Blinc's [`TextureData`].
///
/// glTF images come in several pixel layouts (`R8`, `R8G8`, `R8G8B8`,
/// `R8G8B8A8`, plus 16-bit variants). We expand everything to RGBA8
/// since Blinc's `TextureData` is a flat RGBA buffer. 16-bit formats
/// are downcast to 8-bit; this matches the shader's 8-bit sampling
/// target but loses some precision — see BACKLOG for 16-bit passthrough.
fn image_to_texture(img: &gltf::image::Data) -> Option<TextureData> {
    use gltf::image::Format;
    let width = img.width;
    let height = img.height;
    let n = (width as usize) * (height as usize);
    let rgba = match img.format {
        Format::R8 => expand_r_to_rgba(&img.pixels, n),
        Format::R8G8 => expand_rg_to_rgba(&img.pixels, n),
        Format::R8G8B8 => expand_rgb_to_rgba(&img.pixels, n),
        Format::R8G8B8A8 => img.pixels.clone(),
        Format::R16 => downcast_r16_to_rgba(&img.pixels, n),
        Format::R16G16 => downcast_rg16_to_rgba(&img.pixels, n),
        Format::R16G16B16 => downcast_rgb16_to_rgba(&img.pixels, n),
        Format::R16G16B16A16 => downcast_rgba16(&img.pixels, n),
        Format::R32G32B32FLOAT => downcast_rgb32f_to_rgba8(&img.pixels, n),
        Format::R32G32B32A32FLOAT => downcast_rgba32f_to_rgba8(&img.pixels, n),
    };
    Some(TextureData {
        rgba,
        width,
        height,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// 8-bit channel expansion
// ─────────────────────────────────────────────────────────────────────────────

fn expand_r_to_rgba(src: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for &r in src.iter().take(n) {
        out.extend_from_slice(&[r, r, r, 255]);
    }
    out
}

fn expand_rg_to_rgba(src: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        out.extend_from_slice(&[src[i * 2], src[i * 2 + 1], 0, 255]);
    }
    out
}

fn expand_rgb_to_rgba(src: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        out.extend_from_slice(&[src[i * 3], src[i * 3 + 1], src[i * 3 + 2], 255]);
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// 16-bit → 8-bit downcasts (little-endian source layout)
// ─────────────────────────────────────────────────────────────────────────────
//
// Every pair of bytes is one u16 channel; we keep the high byte (= `>> 8`)
// as the 8-bit value. Sufficient for normal maps / AO; precision loss
// is on the BACKLOG.

fn downcast_r16_to_rgba(src: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let r = src[i * 2 + 1]; // high byte
        out.extend_from_slice(&[r, r, r, 255]);
    }
    out
}

fn downcast_rg16_to_rgba(src: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let r = src[i * 4 + 1];
        let g = src[i * 4 + 3];
        out.extend_from_slice(&[r, g, 0, 255]);
    }
    out
}

fn downcast_rgb16_to_rgba(src: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let r = src[i * 6 + 1];
        let g = src[i * 6 + 3];
        let b = src[i * 6 + 5];
        out.extend_from_slice(&[r, g, b, 255]);
    }
    out
}

fn downcast_rgba16(src: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let r = src[i * 8 + 1];
        let g = src[i * 8 + 3];
        let b = src[i * 8 + 5];
        let a = src[i * 8 + 7];
        out.extend_from_slice(&[r, g, b, a]);
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// HDR → 8-bit tone-mapping (Reinhard)
// ─────────────────────────────────────────────────────────────────────────────
//
// Uses a simple Reinhard operator (`x / (1 + x)`) to compress HDR floats
// into the `[0, 1]` range before converting to 8 bits. Good enough for
// preview / base-color sampling; dedicated HDR materials want
// floating-point textures and belong on the BACKLOG.

fn reinhard_u8(x: f32) -> u8 {
    let y = (x / (1.0 + x)).clamp(0.0, 1.0);
    (y * 255.0 + 0.5) as u8
}

fn downcast_rgb32f_to_rgba8(src: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let base = i * 12;
        let r = f32::from_le_bytes(src[base..base + 4].try_into().unwrap());
        let g = f32::from_le_bytes(src[base + 4..base + 8].try_into().unwrap());
        let b = f32::from_le_bytes(src[base + 8..base + 12].try_into().unwrap());
        out.extend_from_slice(&[reinhard_u8(r), reinhard_u8(g), reinhard_u8(b), 255]);
    }
    out
}

fn downcast_rgba32f_to_rgba8(src: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let base = i * 16;
        let r = f32::from_le_bytes(src[base..base + 4].try_into().unwrap());
        let g = f32::from_le_bytes(src[base + 4..base + 8].try_into().unwrap());
        let b = f32::from_le_bytes(src[base + 8..base + 12].try_into().unwrap());
        let a = f32::from_le_bytes(src[base + 12..base + 16].try_into().unwrap());
        out.extend_from_slice(&[
            reinhard_u8(r),
            reinhard_u8(g),
            reinhard_u8(b),
            (a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        ]);
    }
    out
}
