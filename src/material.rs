//! glTF material → [`blinc_core::draw::Material`] mapping.
//!
//! Covers the full `pbrMetallicRoughness` block plus standard texture
//! slots (base color, normal, metallic-roughness, emissive, occlusion)
//! and the `KHR_materials_unlit` extension. Alpha mode + cutoff both
//! thread through to `Material.alpha_mode` and `Material.alpha_cutoff`.

use blinc_core::draw::{AlphaMode, Material, TextureData};

/// Summary of a base-color texture's alpha channel. Drives the BLEND
/// demotion at parse time — see [`analyze_alpha_distribution`] and the
/// `resolved_alpha_mode` block in [`parse_material`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AlphaProfile {
    /// ≥95% of texels have α ≥ 0.95. The remaining 5% is typically
    /// anti-aliasing at UV island edges — visually indistinguishable
    /// from fully opaque when rendered forward.
    Opaque,
    /// ≥99% of texels sit at either α ≤ 0.05 or α ≥ 0.95, with <1%
    /// midrange. Classic alpha cutout (tree leaves, hair strands,
    /// cloth trim) — safe to demote to MASK with cutoff 0.5.
    Binary,
    /// Genuine partial transparency — meaningful midrange alpha.
    /// Smoke, frosted glass, gradient vignettes. Stays BLEND so the
    /// renderer's OIT pass composites it correctly.
    Partial,
}

/// Walk the RGBA8 buffer and classify the alpha channel. Returns
/// `None` when the buffer has been released (GPU-only after upload) —
/// callers should treat that as "can't demote, keep BLEND".
///
/// Cost: one linear scan of 4 bytes × width × height. For a 2K × 2K
/// texture that's ~4M alpha reads — sub-millisecond on desktop,
/// executed once per material at load, never per-frame.
fn analyze_alpha_distribution(tex: &TextureData) -> Option<AlphaProfile> {
    tex.with_bytes(|bytes| {
        let total = bytes.len() / 4;
        if total == 0 {
            return AlphaProfile::Opaque;
        }
        // Thresholds expressed in raw u8 space to avoid per-pixel
        // conversion (≥95% → ≥242/255, ≤5% → ≤12/255).
        const HI: u8 = 242;
        const LO: u8 = 12;
        let mut hi_count = 0usize;
        let mut lo_count = 0usize;
        let mut mid_count = 0usize;
        for i in 0..total {
            let a = bytes[i * 4 + 3];
            if a >= HI {
                hi_count += 1;
            } else if a <= LO {
                lo_count += 1;
            } else {
                mid_count += 1;
            }
        }
        let hi_frac = hi_count as f32 / total as f32;
        let mid_frac = mid_count as f32 / total as f32;
        let binary_frac = (hi_count + lo_count) as f32 / total as f32;

        if hi_frac >= 0.95 {
            AlphaProfile::Opaque
        } else if binary_frac >= 0.99 && mid_frac < 0.01 {
            AlphaProfile::Binary
        } else {
            AlphaProfile::Partial
        }
    })
}

/// Decode a glTF material into Blinc's `Material`.
///
/// `decoded_images[i]` must hold the pre-decoded texture for glTF
/// image index `i` (see [`crate::decode_images_once`]). Sharing a
/// decoded-once image pool across every material call means a single
/// 4K × 4K RGBA image that's referenced by ten materials exists exactly
/// once in memory — `TextureData::clone()` is a refcount bump on the
/// underlying `Arc<[u8]>`.
pub fn parse_material(mat: &gltf::Material, decoded_images: &[Option<TextureData>]) -> Material {
    let tex = |idx: usize| decoded_images.get(idx).and_then(|t| t.clone());
    let pbr = mat.pbr_metallic_roughness();

    // `KHR_materials_pbrSpecularGlossiness` — legacy workflow used by
    // older exporters (Sketchfab archive, Poly Haven pre-2020, the
    // strangler rig). When present it SUPERSEDES the core
    // `pbrMetallicRoughness` block (the spec says consumers that
    // understand the extension must use it), so we detect it first
    // and do a lossy-but-usable conversion into Blinc's MR-only
    // material channels.
    //
    // Conversion heuristic (keeps runtime simple; no texture rebaking
    // at load):
    //   baseColor.rgb = diffuseFactor.rgb
    //   baseColor.a   = diffuseFactor.a
    //   metallic      = ((max(specular) - 0.04) / 0.96)  clamped
    //   roughness     = 1 - glossinessFactor
    //   baseColorTexture = diffuseTexture
    //   metallicRoughnessTexture = None  (channel layouts differ —
    //       specGloss packs specular into RGB + glossiness into A,
    //       MR packs [_, roughness, metallic] — repacking per-pixel
    //       would need a load-time bake we don't do yet)
    //
    // Character assets (the common case for this extension in 2026)
    // are almost always dielectric, so the metallic-from-specular
    // approximation collapses to ~0 and the diffuse factor carries
    // the full base color — visually indistinguishable from the
    // authored look for skin/cloth/hair.
    let (base_color_factor, metallic_factor, roughness_factor, sg_base_color_texture) =
        if let Some(sg) = mat.pbr_specular_glossiness() {
            let diffuse = sg.diffuse_factor();
            let specular = sg.specular_factor();
            let glossiness = sg.glossiness_factor();
            let max_spec = specular[0].max(specular[1]).max(specular[2]);
            let metallic = ((max_spec - 0.04) / 0.96).clamp(0.0, 1.0);
            let roughness = (1.0 - glossiness).clamp(0.0, 1.0);
            let diffuse_tex = sg
                .diffuse_texture()
                .and_then(|info| tex(info.texture().source().index()));
            (diffuse, metallic, roughness, Some(diffuse_tex))
        } else {
            (
                pbr.base_color_factor(),
                pbr.metallic_factor(),
                pbr.roughness_factor(),
                None,
            )
        };

    let base_color_texture = sg_base_color_texture.unwrap_or_else(|| {
        pbr.base_color_texture()
            .and_then(|info| tex(info.texture().source().index()))
    });
    let metallic_roughness_texture = pbr
        .metallic_roughness_texture()
        .and_then(|info| tex(info.texture().source().index()));
    let normal_info = mat.normal_texture();
    let normal_map = normal_info
        .as_ref()
        .and_then(|info| tex(info.texture().source().index()));
    let normal_scale = normal_info.as_ref().map(|n| n.scale()).unwrap_or(1.0);
    let occlusion_info = mat.occlusion_texture();
    let occlusion_texture = occlusion_info
        .as_ref()
        .and_then(|info| tex(info.texture().source().index()));
    let occlusion_strength = occlusion_info.as_ref().map(|o| o.strength()).unwrap_or(1.0);
    let emissive_texture = mat
        .emissive_texture()
        .and_then(|info| tex(info.texture().source().index()));

    // `KHR_materials_emissive_strength` — the `emissiveFactor` in
    // core glTF is clamped to `[0, 1]` per channel; authors who want
    // HDR glow (dashboards, eyes, lit details) use this extension to
    // multiply the factor by a scalar. When the extension is absent
    // the strength defaults to `1.0`, leaving the factor untouched.
    let emissive_factor = mat.emissive_factor();
    let emissive_strength = mat.emissive_strength().unwrap_or(1.0);
    let emissive = [
        emissive_factor[0] * emissive_strength,
        emissive_factor[1] * emissive_strength,
        emissive_factor[2] * emissive_strength,
    ];

    // Auto-demote BLEND based on actual alpha distribution.
    //
    // Many DCC exporters (Sketchfab upload, older Blender versions,
    // spec-gloss pipelines) flag every material as BLEND by default,
    // even when the source texture is fully opaque or uses a hard
    // cutout. With weighted-blended OIT downstream this produces
    // visible artifacts: solid body panels get stacked with
    // semi-random weights, yielding the pastel "washed-out" look
    // (Σ(c*α*w)/Σ(α*w) averaging across N layers with similar
    // weights), and the compositor's coverage factor undershoots
    // even where all layers are authored fully opaque.
    //
    // OIT is the correct framework fix for *genuine* transparency
    // (smoke, glass, soft foliage, thin overlays). Misflagged opaque
    // materials should be demoted so they write depth and composite
    // via the forward path instead.
    //
    // Heuristic (safe by construction — only moves BLEND out, never
    // changes OPAQUE or MASK):
    //
    //   - baseColorFactor.a < 0.99  → keep BLEND (author explicitly
    //     reduced alpha — respects intent even without a texture
    //     to sample).
    //
    //   - No texture & factor.a ≥ 0.99  → OPAQUE.
    //
    //   - Texture profile = Opaque  → OPAQUE (≥95% fully-opaque
    //     texels; the 5% AA tail composites fine as opaque).
    //
    //   - Texture profile = Binary  → MASK with cutoff 0.5 (classic
    //     alpha cutout — writes depth, composites correctly).
    //
    //   - Texture profile = Partial → keep BLEND (meaningful
    //     midrange alpha = genuine translucency).
    //
    // Decisions are traced at `debug` level so assets that misbehave
    // can be diagnosed with `RUST_LOG=blinc_gltf=debug`.
    let authored_alpha_mode = mat.alpha_mode();
    let effective_factor_a = base_color_factor[3];
    let (resolved_alpha_mode, demote_reason) = match authored_alpha_mode {
        gltf::material::AlphaMode::Opaque => (AlphaMode::Opaque, None),
        gltf::material::AlphaMode::Mask => (AlphaMode::Mask, None),
        gltf::material::AlphaMode::Blend if effective_factor_a < 0.99 => (AlphaMode::Blend, None),
        gltf::material::AlphaMode::Blend => {
            let profile = base_color_texture
                .as_ref()
                .and_then(analyze_alpha_distribution);
            match profile {
                None => (AlphaMode::Opaque, Some("no texture, factor.a≈1")),
                Some(AlphaProfile::Opaque) => {
                    (AlphaMode::Opaque, Some("texture ≥95% fully-opaque"))
                }
                Some(AlphaProfile::Binary) => (AlphaMode::Mask, Some("texture strictly binary α")),
                Some(AlphaProfile::Partial) => (AlphaMode::Blend, None),
            }
        }
    };

    tracing::debug!(
        material = mat.name().unwrap_or("<unnamed>"),
        authored = ?authored_alpha_mode,
        resolved = ?resolved_alpha_mode,
        factor_a = effective_factor_a,
        demote_reason,
        "parsed material"
    );

    // Block-compress diffuse + normal + occlusion textures when
    // the `bc-encode` feature is on. Done late (after the demotion
    // decision) so we can pick BC1 for Opaque-resolved diffuse and
    // BC3 for Blend/Mask-resolved diffuse. Other slots (MR,
    // emissive) stay Rgba8 in this revision — MR would benefit
    // from BC5 or a pair of BC4 channel packs but the existing
    // glTF MR layout (occlusion.r, roughness.g, metallic.b) isn't
    // a clean fit for either BC format without a channel rebake.
    //
    // Keeping the legacy Rgba8 path when the feature is off — same
    // TextureData shape flowing into the mesh pipeline, just
    // larger.
    #[cfg(feature = "bc-encode")]
    let (base_color_texture, normal_map, occlusion_texture) = compress_textures(
        base_color_texture,
        normal_map,
        occlusion_texture,
        resolved_alpha_mode,
    );

    Material {
        base_color: base_color_factor,
        metallic: metallic_factor,
        roughness: roughness_factor,
        emissive,
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
        //
        // `displacement_scale` MUST be `0.0` when `displacement_map`
        // is `None`: the mesh shader's parallax block is gated on
        // `scale > 0`, NOT on map presence. With a non-zero scale it
        // marches UVs through the bound displacement texture (the
        // pipeline's default 1×1 fallback when no map is attached)
        // and discards any fragment whose UV drifts out of `[0, 1]`.
        // For curved surfaces viewed at glancing angles that happens
        // at almost every pixel — making meshes invisible. Leave
        // scale at zero unless an actual map is set.
        displacement_map: None,
        displacement_scale: 0.0,
        unlit: mat.unlit(),
        alpha_mode: resolved_alpha_mode,
        // glTF's `alphaCutoff` — only meaningful when alpha_mode is
        // Mask. Absent from the JSON means "use the spec default 0.5".
        // The same value is used when we auto-demote BLEND → MASK.
        alpha_cutoff: mat.alpha_cutoff().unwrap_or(0.5),
        // Shadows on by default. Blinc's mesh pipeline does
        // two-phase shadow mapping: one depth pass over every caster
        // populates a shared shadow map, then the main color pass
        // samples it with PCF. Per-mesh cost is one extra depth-only
        // draw plus an O(1) shadow sampler in the fragment shader —
        // cheap enough that defaulting to off silently hid shadows
        // from everyone who just calls `load_path(...)` and renders.
        // Callers who need the old opt-in behavior can clear both
        // flags via `map_material`.
        receives_shadows: true,
        casts_shadows: true,
    }
}

/// Encode diffuse, normal-map, and occlusion slots into
/// block-compressed variants based on the material's resolved
/// alpha mode.
///
/// - Diffuse + `AlphaMode::Opaque` → BC1 (4 bpp, sRGB)
/// - Diffuse + `AlphaMode::Mask` or `AlphaMode::Blend`
///   → BC3 (8 bpp, sRGB)
/// - Normal map → BC5 (8 bpp, linear; shader reconstructs B from RG)
/// - Occlusion → BC4 on the red channel (4 bpp, linear)
///
/// Returns the possibly-reassigned textures. When the CPU bytes
/// aren't available (for example the `TextureData` was already
/// GPU-uploaded and `drop_cpu_bytes` was called — doesn't happen
/// during material parse but future-proofs against it), falls
/// back to returning the original Rgba8 texture unchanged.
///
/// Only compiled when the `bc-encode` cargo feature is enabled.
#[cfg(feature = "bc-encode")]
fn compress_textures(
    base_color: Option<TextureData>,
    normal: Option<TextureData>,
    occlusion: Option<TextureData>,
    resolved_alpha_mode: AlphaMode,
) -> (
    Option<TextureData>,
    Option<TextureData>,
    Option<TextureData>,
) {
    let encode_one =
        |td: &TextureData, kind: fn(&[u8], u32, u32) -> TextureData| -> Option<TextureData> {
            if td.format.is_compressed() {
                // Already compressed by a caller or a previous pass —
                // don't re-encode.
                return None;
            }
            let w = td.width;
            let h = td.height;
            td.with_bytes(|bytes| kind(bytes, w, h))
        };

    let new_base = base_color.as_ref().and_then(|td| {
        let kind = match resolved_alpha_mode {
            AlphaMode::Opaque => crate::bc_encode::encode_bc1,
            AlphaMode::Mask | AlphaMode::Blend => crate::bc_encode::encode_bc3,
        };
        encode_one(td, kind)
    });
    let new_normal = normal
        .as_ref()
        .and_then(|td| encode_one(td, crate::bc_encode::encode_bc5_rg));
    let new_occl = occlusion
        .as_ref()
        .and_then(|td| encode_one(td, crate::bc_encode::encode_bc4_red));
    (
        new_base.or(base_color),
        new_normal.or(normal),
        new_occl.or(occlusion),
    )
}

/// Convert a decoded glTF image into Blinc's [`TextureData`].
///
/// glTF images come in several pixel layouts (`R8`, `R8G8`, `R8G8B8`,
/// `R8G8B8A8`, plus 16-bit and 32-bit float variants). We expand
/// everything to RGBA8 since Blinc's `TextureData` is a flat RGBA
/// buffer. 16-bit formats are downcast to 8-bit; this matches the
/// shader's 8-bit sampling target but loses some precision — see
/// BACKLOG for 16-bit passthrough.
///
/// The output `rgba` buffer is wrapped in `Arc<[u8]>` so that callers
/// can stamp the returned `TextureData` onto many materials via a
/// cheap refcount bump.
pub fn image_to_texture(img: &gltf::image::Data) -> Option<TextureData> {
    use gltf::image::Format;
    let width = img.width;
    let height = img.height;
    let n = (width as usize) * (height as usize);
    let bytes: Vec<u8> = match img.format {
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
    Some(TextureData::new(bytes, width, height))
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
