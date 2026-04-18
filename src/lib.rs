//! glTF 2.0 loader for Blinc.
//!
//! Parses glTF / GLB content via the upstream [`gltf`] crate and maps it
//! into [`blinc_core`] interchange types (`MeshData`, `Vertex`,
//! `Material`, `Skeleton`, `Bone`) plus a thin scene graph suitable for
//! feeding [`blinc_canvas_kit::SceneKit3D`].
//!
//! # Example
//!
//! ```ignore
//! use blinc_gltf::load_glb;
//! use blinc_canvas_kit::SceneKit3D;
//!
//! let bytes = std::fs::read("DamagedHelmet.glb")?;
//! let scene = load_glb(&bytes)?;
//!
//! let kit = SceneKit3D::new("viewer");
//! let handles = scene.add_to(&kit);  // node transforms baked into vertices
//! println!("{} handles spawned", handles.len());
//! # Ok::<_, Box<dyn std::error::Error>>(())
//! ```
//!
//! # What's mapped (v0)
//!
//! - **Meshes** — one `MeshData` per glTF primitive. Positions,
//!   normals, UV (TEXCOORD_0), vertex colors (COLOR_0), tangents,
//!   joints (JOINTS_0), and weights (WEIGHTS_0) all round-trip.
//! - **Materials** — full PBR metallic-roughness: base color, metallic,
//!   roughness, emissive factors; base-color / normal /
//!   metallic-roughness / emissive / occlusion textures; alpha mode
//!   (`OPAQUE` / `MASK` / `BLEND`); `KHR_materials_unlit`.
//! - **Scene graph** — nodes carry their local TRS (translation /
//!   rotation-quat / scale) or raw 4×4 matrix; world transforms are
//!   composed on demand via [`GltfScene::compute_world_transforms`].
//! - **Skins** — joint hierarchy + inverse-bind matrices into a
//!   `Skeleton`. Runtime posing lives in `blinc_skeleton`.
//! - **Animations** — channels + samplers parsed into typed keyframes
//!   (translation / rotation / scale / morph-weights). Runtime sampling
//!   lives in `blinc_skeleton`.
//!
//! # Known limitations
//!
//! [`GltfScene::add_to`] bakes node world transforms directly into each
//! vertex rather than storing them on `SceneKit3D` objects — the
//! current `SceneKit3D` transform API only exposes Y-rotation. Static
//! meshes render correctly; non-uniform scale produces slightly
//! incorrect lighting (we skip the inverse-transpose normal fix in
//! v0). See [BACKLOG.md](../BACKLOG.md) for the full list.

use std::path::Path;
use std::sync::Arc;

use blinc_canvas_kit::{MeshHandle, SceneKit3D};
use blinc_core::draw::MeshData;
use blinc_core::Mat4;

mod animation;
#[cfg(feature = "bc-encode")]
mod bc_encode;
mod material;
mod mesh;
mod node;
mod skin;

pub use animation::{
    AnimatedProperty, AnimationChannel, AnimationSampler, AnimationTarget, GltfAnimation,
    Interpolation, KeyframeValues,
};
pub use material::parse_material;
pub use mesh::parse_primitive;
pub use node::{GltfNode, NodeTransform};
pub use skin::GltfSkeleton;

// Re-export `blinc_core` types downstream users commonly need so they
// don't need a direct `blinc_core` dep just to consume the loader's
// output.
pub use blinc_core::draw::{Bone, Material, Skeleton, TextureData, Vertex};

/// Walk every primitive in `scene` and hand its `Material` to a
/// user-supplied closure. Lets the caller re-author per-mesh render
/// hints after load — demote BLEND hair to MASK, clamp a too-generous
/// alpha cutoff, disable shadows on an intentionally-transparent
/// decal, stamp emissive on a specific mesh, etc.
///
/// Every existing "post-parse fixup loop" in the demos (rotor-shadow
/// disable, lens emissive stamp, cutegirl hair demotion) collapses
/// into one call with a match arm.
///
/// The closure receives:
/// - `mesh_index` — index into `scene.meshes`
/// - `mesh_name` — from `GltfMesh.name`, `None` for unnamed meshes
/// - `prim_index` — index within the mesh's primitives
/// - `material` — mutable reference to the primitive's material
///
/// Example — demote every BLEND material to MASK with 0.5 cutoff:
///
/// ```ignore
/// apply_material_overrides(&mut scene, |_, _, _, mat| {
///     if mat.alpha_mode == AlphaMode::Blend {
///         mat.alpha_mode = AlphaMode::Mask;
///         mat.alpha_cutoff = 0.5;
///     }
/// });
/// ```
///
/// Example — disable shadows on anything named "Turbine":
///
/// ```ignore
/// apply_material_overrides(&mut scene, |_, name, _, mat| {
///     if name.map_or(false, |n| n.contains("Turbine")) {
///         mat.casts_shadows = false;
///         mat.receives_shadows = false;
///     }
/// });
/// ```
pub fn apply_material_overrides<F>(scene: &mut GltfScene, mut f: F)
where
    F: FnMut(usize, Option<&str>, usize, &mut Material),
{
    for (mesh_index, mesh) in scene.meshes.iter_mut().enumerate() {
        let name = mesh.name.as_deref();
        for (prim_index, prim) in mesh.primitives.iter_mut().enumerate() {
            f(mesh_index, name, prim_index, &mut prim.material);
        }
    }
}

/// Errors returned by the loader.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("glTF parse failed: {0}")]
    Gltf(#[from] gltf::Error),
    #[error("invalid glTF structure: {0}")]
    Invalid(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// A loaded glTF asset — intermediate representation between `gltf::Document`
/// and Blinc runtime types. All fields public for users who want to
/// filter, remap, or feed their own scene system instead of calling
/// [`GltfScene::add_to`].
#[derive(Debug, Clone)]
pub struct GltfScene {
    pub meshes: Vec<GltfMesh>,
    pub nodes: Vec<GltfNode>,
    /// Indices into `nodes` that form the roots of the default scene.
    pub root_nodes: Vec<usize>,
    pub skeletons: Vec<GltfSkeleton>,
    pub animations: Vec<GltfAnimation>,
}

/// A glTF mesh — holds one `MeshData` per primitive (glTF allows
/// multi-material meshes to split geometry across primitives).
#[derive(Debug, Clone)]
pub struct GltfMesh {
    pub name: Option<String>,
    pub primitives: Vec<MeshData>,
    /// Axis-aligned bounding box in the mesh's local coordinate space,
    /// covering every primitive. Sourced from the primitive's
    /// `POSITION` accessor min/max, which the glTF spec requires to be
    /// set. Used by [`GltfScene::world_aabb`] for camera framing.
    pub local_bbox: ([f32; 3], [f32; 3]),
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry points
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a self-contained `.glb` blob (embedded JSON + buffers + images).
pub fn load_glb(bytes: &[u8]) -> Result<GltfScene, Error> {
    let (doc, buffers, images) = gltf::import_slice(bytes)?;
    from_import(&doc, &buffers, &images)
}

/// Parse a `.gltf` JSON file. External buffers / textures referenced
/// via relative URIs resolve against the JSON file's parent directory.
pub fn load_path<P: AsRef<Path>>(path: P) -> Result<GltfScene, Error> {
    let (doc, buffers, images) = gltf::import(path.as_ref())?;
    from_import(&doc, &buffers, &images)
}

/// Tunables for [`load_asset_with_options`]. Everything is optional —
/// `LoadOptions::default()` reproduces the behavior of [`load_asset`].
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// If set, images whose longer dimension exceeds this value are
    /// resized (aspect-preserving) with bilinear filtering before the
    /// decoded pixels are handed to the material parser. Bounded by
    /// `1 = keep one texel`, unbounded above.
    ///
    /// Useful for asset budgets — a 4K texture downscaled to 2K is 4×
    /// smaller on both CPU (during decode) and GPU (after upload) and
    /// is visually indistinguishable at normal viewing distances.
    ///
    /// The cost is paid once at load, inside the `image` crate's
    /// `DynamicImage::resize` with `FilterType::Triangle`. Assets that
    /// ship textures already at or below the cap go through unchanged
    /// with a single dimension check.
    pub max_texture_size: Option<u32>,
}

/// Load a glTF asset through the global `blinc_platform::assets` loader.
///
/// Unlike [`load_path`], which goes directly through the filesystem
/// via `gltf::import`, this entry point defers every read (the main
/// file plus every external buffer and image referenced via relative
/// URI) to whatever `AssetLoader` the host registered with
/// `blinc_platform`. That means the same code path works on:
///
/// - Desktop — default `FilesystemAssetLoader`
/// - Android — APK `AssetManager` loader
/// - iOS — app-bundle loader
/// - Web — HTTP-fetch loader
///
/// `path` is passed verbatim to `blinc_platform::assets::load_asset`
/// for the main file; external URIs are resolved by prepending
/// `path`'s parent directory and forwarding to the same loader.
///
/// Data URIs (`data:...;base64,...`) are decoded inline without
/// involving the asset loader.
///
/// For fine-grained control (texture downsampling, etc.) see
/// [`load_asset_with_options`].
#[cfg(feature = "platform-assets")]
pub fn load_asset(path: &str) -> Result<GltfScene, Error> {
    load_asset_with_options(path, &LoadOptions::default())
}

/// Same as [`load_asset`] but applies the transforms configured in
/// `opts` (texture downsampling, future knobs).
#[cfg(feature = "platform-assets")]
pub fn load_asset_with_options(path: &str, opts: &LoadOptions) -> Result<GltfScene, Error> {
    let bytes = blinc_platform::assets::load_asset(path)
        .map_err(|e| Error::Invalid(format!("asset load '{}': {}", path, e)))?;

    // Detect self-contained binary glTF by its magic header. `.glb`
    // files never reference external URIs, so they can go through the
    // existing slice-based path.
    if bytes.len() >= 4 && &bytes[0..4] == b"glTF" {
        let (doc, buffers, mut images) = gltf::import_slice(&bytes)?;
        if let Some(max) = opts.max_texture_size {
            downsample_images(&mut images, max);
        }
        return from_import(&doc, &buffers, &images);
    }

    // JSON glTF — parse the document first, then manually resolve
    // each buffer/image through `blinc_platform::assets` with the
    // main file's parent directory as the URI base.
    let base_dir: &str = path.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("");
    let gltf_obj = gltf::Gltf::from_slice(&bytes)?;
    let buffers = resolve_buffers(&gltf_obj, base_dir)?;
    let images = resolve_images(&gltf_obj, base_dir, &buffers, opts.max_texture_size)?;
    from_import(&gltf_obj.document, &buffers, &images)
}

/// Shrink any image in `images` whose longer dimension exceeds `max`,
/// preserving aspect. Skipped for images already at or below the cap
/// and for source formats other than RGB8 / RGBA8 (PNG/JPEG covers the
/// common case; 16-bit + float layouts go through unchanged and get
/// the RGBA8 expansion later in [`crate::material::image_to_texture`]).
///
/// Uses bilinear (`Triangle`) filtering — good quality for diffuse /
/// normal / metallic-roughness maps, fast enough for load-time.
#[cfg(feature = "platform-assets")]
fn downsample_images(images: &mut [gltf::image::Data], max: u32) {
    // Per-image resize is embarrassingly parallel (each output
    // image depends only on its own input pixels). On native we
    // farm the loop across rayon's global pool; a 29-texture
    // strangler rig on an 8-core desktop cuts 2-3 s of bilinear
    // resize down to ~400-600 ms. wasm32 has no threads so the
    // sequential fallback stays.
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        images.par_iter_mut().for_each(|img| downsample_one(img, max));
    }
    #[cfg(target_arch = "wasm32")]
    {
        for img in images.iter_mut() {
            downsample_one(img, max);
        }
    }
}

#[cfg(feature = "platform-assets")]
fn downsample_one(img: &mut gltf::image::Data, max: u32) {
    use image::{ImageBuffer, Rgba};
    let max = max.max(1);
    let longer = img.width.max(img.height);
    if longer <= max {
        return;
    }

    // Promote to RGBA8 inline so `ImageBuffer::from_raw` succeeds
    // on the subsequent construction. Only R8G8B8 and R8G8B8A8
    // are handled here; uncommon formats get skipped and pay the
    // full-res expand later in `material::image_to_texture`.
    let rgba_pixels: Vec<u8> = match img.format {
        gltf::image::Format::R8G8B8A8 => std::mem::take(&mut img.pixels),
        gltf::image::Format::R8G8B8 => {
            let mut out = Vec::with_capacity(img.pixels.len() / 3 * 4);
            for chunk in img.pixels.chunks_exact(3) {
                out.extend_from_slice(chunk);
                out.push(255);
            }
            out
        }
        _ => return,
    };

    let Some(buf): Option<ImageBuffer<Rgba<u8>, Vec<u8>>> =
        ImageBuffer::from_raw(img.width, img.height, rgba_pixels)
    else {
        return;
    };
    let dynimg = image::DynamicImage::ImageRgba8(buf);
    let scale = max as f32 / longer as f32;
    let new_w = ((img.width as f32 * scale).round() as u32).max(1);
    let new_h = ((img.height as f32 * scale).round() as u32).max(1);
    let resized = dynimg.resize(new_w, new_h, image::imageops::FilterType::Triangle);
    img.pixels = resized.to_rgba8().into_raw();
    img.width = new_w;
    img.height = new_h;
    img.format = gltf::image::Format::R8G8B8A8;
}

/// Resolve a relative/data URI to raw bytes via `blinc_platform::assets`.
///
/// Kept separate from `load_asset` so buffer and image loops stay
/// small. Data URIs don't touch the asset loader.
#[cfg(feature = "platform-assets")]
fn resolve_uri_bytes(uri: &str, base_dir: &str) -> Result<Vec<u8>, Error> {
    if let Some(rest) = uri.strip_prefix("data:") {
        // `data:[<media type>];base64,<data>` — only base64 payloads
        // are defined for glTF URIs.
        let (_, b64) = rest
            .split_once(";base64,")
            .ok_or_else(|| Error::Invalid(format!("unsupported data URI: {}", uri)))?;
        use base64::Engine;
        return base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| Error::Invalid(format!("base64 decode: {}", e)));
    }
    // Treat everything else as a relative path. glTF also allows
    // `file:` URIs but those are a desktop-only concept that doesn't
    // make sense through an asset loader abstraction — callers who
    // need them should stay on `load_path`.
    let full = if base_dir.is_empty() {
        uri.to_string()
    } else {
        format!("{}/{}", base_dir, uri)
    };
    blinc_platform::assets::load_asset(&full)
        .map_err(|e| Error::Invalid(format!("asset load '{}': {}", full, e)))
}

#[cfg(feature = "platform-assets")]
fn resolve_buffers(
    gltf_obj: &gltf::Gltf,
    base_dir: &str,
) -> Result<Vec<gltf::buffer::Data>, Error> {
    let mut out = Vec::new();
    // `Gltf::blob` holds the BIN chunk of a .glb — we've already
    // shunted .glb through `load_glb`, so here `blob` is always None.
    let mut blob: Option<Vec<u8>> = gltf_obj.blob.clone();
    for buffer in gltf_obj.document.buffers() {
        let data = match buffer.source() {
            gltf::buffer::Source::Bin => blob
                .take()
                .ok_or_else(|| Error::Invalid("missing GLB blob".into()))?,
            gltf::buffer::Source::Uri(uri) => resolve_uri_bytes(uri, base_dir)?,
        };
        // glTF spec requires buffer byte length match the accessor
        // use; short buffers would panic later in accessor reads.
        if data.len() < buffer.length() {
            return Err(Error::Invalid(format!(
                "buffer {} short: expected {}, got {}",
                buffer.index(),
                buffer.length(),
                data.len()
            )));
        }
        // `buffer::Data` is a tuple struct with a pub Vec<u8> field.
        // We also pad to a 4-byte boundary to match the upstream
        // `from_source_and_blob` behavior (accessors require it).
        let mut padded = data;
        while padded.len() % 4 != 0 {
            padded.push(0);
        }
        out.push(gltf::buffer::Data(padded));
    }
    Ok(out)
}

#[cfg(feature = "platform-assets")]
fn resolve_images(
    gltf_obj: &gltf::Gltf,
    base_dir: &str,
    buffers: &[gltf::buffer::Data],
    max_texture_size: Option<u32>,
) -> Result<Vec<gltf::image::Data>, Error> {
    // Only swap in the placeholder once the host loader says no more
    // bytes are coming. Otherwise a retry-loop caller (e.g. the 3D
    // demos' `try_load` polling on 100 ms while preload is in flight)
    // would "succeed" on the first tick with a placeholder baked in,
    // freeze the scene with an untextured model, and the real
    // textures would never land because the scene is already built.
    // Desktop / filesystem loaders report `true` unconditionally, so
    // this extra check is a no-op there — it's purely for async
    // preload hosts (web).
    let preload_settled = blinc_platform::assets::preload_settled();

    // Phase 1 (sequential): resolve every image source to a byte
    // buffer (or a placeholder marker). Cheap — the asset loader is
    // either an in-memory cache lookup (web) or an `fs::read` (native).
    // Ordering matters: the decoded `Vec<gltf::image::Data>` indexes
    // into glTF accessor references, so we collect here before any
    // parallel fan-out.
    enum Encoded<'a> {
        Bytes(Vec<u8>, Option<&'a str>),
        Placeholder,
    }
    let mut encoded: Vec<Encoded<'_>> = Vec::new();
    for image in gltf_obj.document.images() {
        match image.source() {
            gltf::image::Source::Uri { uri, mime_type } => match resolve_uri_bytes(uri, base_dir) {
                Ok(bytes) => encoded.push(Encoded::Bytes(bytes, mime_type)),
                Err(e) if preload_settled => {
                    tracing::warn!(
                        "gltf image '{uri}' skipped ({e:?}) — substituting 1×1 placeholder"
                    );
                    encoded.push(Encoded::Placeholder);
                }
                Err(e) => return Err(e),
            },
            gltf::image::Source::View { view, mime_type } => {
                let buf = &buffers[view.buffer().index()].0;
                let begin = view.offset();
                let end = begin + view.length();
                encoded.push(Encoded::Bytes(buf[begin..end].to_vec(), Some(mime_type)));
            }
        }
    }

    // Phase 2 (parallel on native, sequential on wasm): PNG/JPEG
    // decode + inline downsample. Each image is ~100-300 ms for a
    // 4K texture; 20-30 textures sequentially turns into multi-
    // second stalls at load. Rayon's work-stealing farms the decode
    // across cores without ordering impact — each output slot is
    // written independently.
    //
    // Downsampling folded into the same closure: a 4K RGBA8 buffer
    // is 64 MB, and on wasm the linear memory only grows (never
    // shrinks), so keeping a decoded full-res image alive past its
    // thread's turn pins that 64 MB forever. By resizing before
    // extracting pixels we drop the source buffer inside the
    // closure — each thread's peak is ~80 MB (full-res + resize
    // dest) rather than unbounded, and cuts a strangler-scale wasm
    // peak of ~1.4 GB down to a few hundred MB.
    #[cfg(not(target_arch = "wasm32"))]
    let decoded_vec: Vec<gltf::image::Data> = {
        use rayon::prelude::*;
        encoded
            .into_par_iter()
            .map(|e| decode_image(e, max_texture_size))
            .collect()
    };
    #[cfg(target_arch = "wasm32")]
    let decoded_vec: Vec<gltf::image::Data> = encoded
        .into_iter()
        .map(|e| decode_image(e, max_texture_size))
        .collect();

    return Ok(decoded_vec);

    // A 1×1 opaque-white RGBA8 image. Used when a texture URI fails
    // to resolve or decode so the scene as a whole still loads — a
    // missing diffuse map falls through to `material::parse_material`
    // and becomes an un-textured (white-tinted) primitive rather than
    // blocking every other mesh from rendering. Mirrors the web
    // preloader's partial-failure tolerance: one bad texture
    // shouldn't leave the loading signal stuck forever while the
    // rest of the scene sits ready.
    fn placeholder() -> gltf::image::Data {
        gltf::image::Data {
            pixels: vec![255, 255, 255, 255],
            format: gltf::image::Format::R8G8B8A8,
            width: 1,
            height: 1,
        }
    }

    fn decode_image(enc: Encoded<'_>, max: Option<u32>) -> gltf::image::Data {
        use image::GenericImageView;
        let (bytes, mime) = match enc {
            Encoded::Bytes(b, m) => (b, m),
            Encoded::Placeholder => return placeholder(),
        };
        let decoded = match mime {
            Some("image/png") => {
                image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
            }
            Some("image/jpeg") => {
                image::load_from_memory_with_format(&bytes, image::ImageFormat::Jpeg)
            }
            _ => image::load_from_memory(&bytes),
        };
        match decoded {
            Ok(mut img) => {
                // Resize *before* extracting pixels so the full-res
                // buffer is dropped inside this closure — critical
                // on wasm where linear memory never shrinks. The
                // separate `downsample_images` pass used to see every
                // full-res image at once; now each thread sees at
                // most one.
                if let Some(max) = max {
                    let longer = img.width().max(img.height());
                    if longer > max {
                        let scale = max as f32 / longer as f32;
                        let new_w = ((img.width() as f32 * scale).round() as u32).max(1);
                        let new_h = ((img.height() as f32 * scale).round() as u32).max(1);
                        img = img.resize(new_w, new_h, image::imageops::FilterType::Triangle);
                    }
                }
                let (w, h) = img.dimensions();
                let pixels = img.to_rgba8().into_raw();
                gltf::image::Data {
                    pixels,
                    format: gltf::image::Format::R8G8B8A8,
                    width: w,
                    height: h,
                }
            }
            Err(e) => {
                tracing::warn!("gltf image decode failed ({e}) — substituting 1×1 placeholder");
                placeholder()
            }
        }
    }
}

/// Decode every image in the glTF asset exactly once into an
/// `Arc<[u8]>`-backed [`blinc_core::draw::TextureData`].
///
/// glTF assets commonly reuse a single source image across multiple
/// materials (e.g. one 4K albedo shared by five body primitives).
/// Without this pre-decode step, [`crate::material::parse_material`]
/// would re-decode + reallocate the image for every material
/// reference. Returning one `TextureData` per image lets subsequent
/// material parses share the underlying pixel buffer via refcount
/// bumps, keeping memory proportional to the number of *unique*
/// images, not the number of material-texture-slot references.
pub fn decode_images_once(images: &[gltf::image::Data]) -> Vec<Option<TextureData>> {
    images.iter().map(material::image_to_texture).collect()
}

fn from_import(
    doc: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
) -> Result<GltfScene, Error> {
    let decoded_images = decode_images_once(images);
    let meshes = doc
        .meshes()
        .map(|m| {
            let primitives: Vec<MeshData> = m
                .primitives()
                .map(|p| parse_primitive(&p, buffers, &decoded_images))
                .collect();
            // Union per-primitive POSITION-accessor bounds into a
            // single mesh-local AABB. Every primitive's POSITION
            // accessor is required to carry min/max per the spec, so
            // this is a metadata read — no vertex iteration.
            let local_bbox = m.primitives().fold(
                ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]),
                |(mut mn, mut mx), p| {
                    let bb = p.bounding_box();
                    for i in 0..3 {
                        mn[i] = mn[i].min(bb.min[i]);
                        mx[i] = mx[i].max(bb.max[i]);
                    }
                    (mn, mx)
                },
            );
            GltfMesh {
                name: m.name().map(str::to_string),
                primitives,
                local_bbox,
            }
        })
        .collect();

    let nodes = node::build_nodes(doc);

    let root_nodes = doc
        .default_scene()
        .or_else(|| doc.scenes().next())
        .map(|s| s.nodes().map(|n| n.index()).collect())
        .unwrap_or_default();

    let skeletons = doc.skins().map(|s| skin::parse_skin(&s, buffers)).collect();

    let animations = doc
        .animations()
        .map(|a| animation::parse_animation(&a, buffers))
        .collect();

    Ok(GltfScene {
        meshes,
        nodes,
        root_nodes,
        skeletons,
        animations,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// SceneKit3D integration
// ─────────────────────────────────────────────────────────────────────────────

impl GltfScene {
    /// Look up a node by its authored name. Returns the first match
    /// (glTF doesn't enforce name uniqueness, but DCC-exported scenes
    /// typically don't duplicate them). `None` if no node carries the
    /// name — or if the `gltf` crate's `names` feature isn't enabled
    /// and node names were discarded at parse time.
    ///
    /// Typical use: downstream demos that pin runtime behaviour to a
    /// specific named node (the buster_drone demo's rotor-subtree
    /// scan is the existing example) can replace manual `enumerate +
    /// filter_map` walks with
    /// `scene.node_by_name("Turbine_L")`.
    pub fn node_by_name(&self, name: &str) -> Option<usize> {
        self.nodes
            .iter()
            .position(|n| n.name.as_deref() == Some(name))
    }

    /// Compose each node's local transform with its ancestor chain,
    /// returning world-space 4×4 matrices indexed parallel to `nodes`.
    pub fn compute_world_transforms(&self) -> Vec<Mat4> {
        let mut world = vec![Mat4::IDENTITY; self.nodes.len()];
        for &root in &self.root_nodes {
            walk_transforms(self, root, Mat4::IDENTITY, &mut world);
        }
        world
    }

    /// Axis-aligned bounding box of every mesh-bearing node in world
    /// space, computed at the scene's rest pose (before any animation
    /// is sampled into node transforms).
    ///
    /// Returns `(min, max)` as `[f32; 3]` pairs, or `None` if the
    /// scene has no mesh-bearing nodes. Each mesh's local AABB
    /// ([`GltfMesh::local_bbox`]) is transformed by its node's world
    /// matrix via the 8-corner method — tight enough for camera
    /// framing, conservative enough to avoid popping.
    pub fn world_aabb(&self) -> Option<([f32; 3], [f32; 3])> {
        let world = self.compute_world_transforms();
        let mut accum: Option<([f32; 3], [f32; 3])> = None;
        for (i, node) in self.nodes.iter().enumerate() {
            let Some(mesh_idx) = node.mesh else { continue };
            let Some(mesh) = self.meshes.get(mesh_idx) else {
                continue;
            };
            let (lmin, lmax) = mesh.local_bbox;
            if !lmin[0].is_finite() || !lmax[0].is_finite() {
                continue; // empty-mesh guard
            }
            let corners: [[f32; 3]; 8] = [
                [lmin[0], lmin[1], lmin[2]],
                [lmin[0], lmin[1], lmax[2]],
                [lmin[0], lmax[1], lmin[2]],
                [lmin[0], lmax[1], lmax[2]],
                [lmax[0], lmin[1], lmin[2]],
                [lmax[0], lmin[1], lmax[2]],
                [lmax[0], lmax[1], lmin[2]],
                [lmax[0], lmax[1], lmax[2]],
            ];
            for corner in corners {
                let w = transform_point(&world[i], corner);
                accum = Some(match accum {
                    None => (w, w),
                    Some((mn, mx)) => (
                        [mn[0].min(w[0]), mn[1].min(w[1]), mn[2].min(w[2])],
                        [mx[0].max(w[0]), mx[1].max(w[1]), mx[2].max(w[2])],
                    ),
                });
            }
        }
        accum
    }

    /// Populate a `SceneKit3D` with this asset's meshes. Node world
    /// transforms are **baked into the vertices** at spawn time rather
    /// than applied via the scene's transform setters — this gives a
    /// correct static pose without depending on `SceneKit3D` exposing a
    /// full 4×4 transform API yet.
    ///
    /// Returns one `MeshHandle` per primitive spawned, in DFS node
    /// order. Skins and animations are **not** spawned — those feed
    /// into `blinc_skeleton`.
    pub fn add_to(&self, scene: &SceneKit3D) -> Vec<MeshHandle> {
        let world = self.compute_world_transforms();
        let mut handles = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            let Some(mesh_idx) = node.mesh else { continue };
            let Some(mesh) = self.meshes.get(mesh_idx) else {
                continue;
            };
            for prim in &mesh.primitives {
                let baked = bake_transform(prim, &world[i]);
                handles.push(scene.add_mesh(Arc::new(baked)));
            }
        }
        handles
    }
}

fn walk_transforms(scene: &GltfScene, idx: usize, parent: Mat4, out: &mut [Mat4]) {
    let local = scene.nodes[idx].transform.to_mat4();
    let world = parent.mul(&local);
    out[idx] = world;
    for child in scene.nodes[idx].children.clone() {
        walk_transforms(scene, child, world, out);
    }
}

fn bake_transform(mesh: &MeshData, m: &Mat4) -> MeshData {
    // Normals and tangents need the *inverse-transpose* of the
    // upper-3×3 under non-uniform scale; naïvely applying the same
    // matrix you use for positions (`transform_direction`) skews
    // them toward the squashed axis. For uniform scale /
    // rotation-only transforms the two matrices differ only by a
    // scalar (which normalise3 absorbs), so the bug goes unnoticed
    // until someone applies `node.scale = [1, 0.5, 1]`.
    let normal_matrix = inverse_transpose_upper3x3(m);
    let mut out = mesh.clone();
    // Bake into a unique copy of the shared vertex buffer. `make_mut`
    // clones the inner Vec if the Arc is shared, otherwise mutates
    // in place — the right thing in both cases.
    for v in std::sync::Arc::make_mut(&mut out.vertices) {
        v.position = transform_point(m, v.position);
        v.normal = normalize3(mul3x3(&normal_matrix, v.normal));
        // Tangents are direction vectors too (but stay in the
        // surface's tangent plane — the same inverse-transpose fix
        // applies). `w` is a handedness sign and passes through.
        let t = mul3x3(&normal_matrix, [v.tangent[0], v.tangent[1], v.tangent[2]]);
        let t = normalize3(t);
        v.tangent = [t[0], t[1], t[2], v.tangent[3]];
    }
    out
}

fn transform_point(m: &Mat4, p: [f32; 3]) -> [f32; 3] {
    let c = &m.cols;
    [
        c[0][0] * p[0] + c[1][0] * p[1] + c[2][0] * p[2] + c[3][0],
        c[0][1] * p[0] + c[1][1] * p[1] + c[2][1] * p[2] + c[3][1],
        c[0][2] * p[0] + c[1][2] * p[1] + c[2][2] * p[2] + c[3][2],
    ]
}

fn mul3x3(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    // Column-major 3×3 multiply: `out[i] = Σ m[j][i] * v[j]`.
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// Inverse-transpose of a 4×4 transform's upper-3×3. Falls back to
/// the plain upper-3×3 on singular matrices (determinant near zero) —
/// keeps a valid-ish normal rather than blowing up on assets with
/// a degenerate node scale. Column-major throughout so the return
/// value plugs directly into [`mul3x3`].
fn inverse_transpose_upper3x3(m: &Mat4) -> [[f32; 3]; 3] {
    let a = [
        [m.cols[0][0], m.cols[0][1], m.cols[0][2]],
        [m.cols[1][0], m.cols[1][1], m.cols[1][2]],
        [m.cols[2][0], m.cols[2][1], m.cols[2][2]],
    ];
    // Cofactor expansion along column 0.
    let c00 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
    let c01 = -(a[1][0] * a[2][2] - a[1][2] * a[2][0]);
    let c02 = a[1][0] * a[2][1] - a[1][1] * a[2][0];
    let det = a[0][0] * c00 + a[0][1] * c01 + a[0][2] * c02;
    if det.abs() < 1e-8 {
        // Degenerate — fall back to identity-ish: just use the
        // original upper-3×3 so normals rotate but don't flip.
        return a;
    }
    let inv_det = 1.0 / det;
    // Build `inverse(a)` via cofactor matrix, then take the transpose
    // in one step by writing rows of the inverse as columns.
    let c10 = -(a[0][1] * a[2][2] - a[0][2] * a[2][1]);
    let c11 = a[0][0] * a[2][2] - a[0][2] * a[2][0];
    let c12 = -(a[0][0] * a[2][1] - a[0][1] * a[2][0]);
    let c20 = a[0][1] * a[1][2] - a[0][2] * a[1][1];
    let c21 = -(a[0][0] * a[1][2] - a[0][2] * a[1][0]);
    let c22 = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    [
        [c00 * inv_det, c01 * inv_det, c02 * inv_det],
        [c10 * inv_det, c11 * inv_det, c12 * inv_det],
        [c20 * inv_det, c21 * inv_det, c22 * inv_det],
    ]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-8 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_compiles() {
        // Smoke: asserts the module graph wires together. Real
        // round-trip tests need fixture `.glb` files and belong in the
        // integration-test layer once we ship one.
    }

    #[test]
    fn inverse_transpose_preserves_normal_under_non_uniform_scale() {
        // Column-major transform that scales +Y by 2 and leaves XZ
        // alone. A normal pointing diagonally in the XY plane should,
        // after the transform, still be perpendicular to the surface
        // — which under non-uniform scale means the Y component
        // shrinks, not grows. Compare with naïve "apply the same
        // matrix" to confirm the old path bent the normal the wrong
        // way.
        let m = Mat4 {
            cols: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        let n_before = normalize3([1.0, 1.0, 0.0]);
        let inv_t = inverse_transpose_upper3x3(&m);
        let n_after = normalize3(mul3x3(&inv_t, n_before));
        // Expected: inverse-transpose of scale(1, 2, 1) is
        // scale(1, 0.5, 1). Applied to (√2/2, √2/2, 0) gives
        // (√2/2, √2/4, 0); normalised → (2, 1, 0) / √5.
        let expected = normalize3([2.0, 1.0, 0.0]);
        for i in 0..3 {
            assert!(
                (n_after[i] - expected[i]).abs() < 1e-5,
                "component {i}: got {}, expected {}",
                n_after[i],
                expected[i]
            );
        }
    }

    #[test]
    fn inverse_transpose_pure_rotation_equals_matrix() {
        // For orthonormal rotation-only matrices, inverse-transpose
        // = original. 90° around Y: X axis → -Z, Z axis → +X.
        let m = Mat4 {
            cols: [
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        let inv_t = inverse_transpose_upper3x3(&m);
        let original: [[f32; 3]; 3] = [
            [m.cols[0][0], m.cols[0][1], m.cols[0][2]],
            [m.cols[1][0], m.cols[1][1], m.cols[1][2]],
            [m.cols[2][0], m.cols[2][1], m.cols[2][2]],
        ];
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (inv_t[i][j] - original[i][j]).abs() < 1e-5,
                    "mismatch at [{i}][{j}]: {} vs {}",
                    inv_t[i][j],
                    original[i][j]
                );
            }
        }
    }

    #[test]
    fn node_by_name_finds_matching_node() {
        let scene = GltfScene {
            nodes: vec![
                GltfNode {
                    name: Some("root".into()),
                    parent: None,
                    children: vec![],
                    mesh: None,
                    skin: None,
                    transform: NodeTransform::Trs {
                        translation: [0.0; 3],
                        rotation: [0.0, 0.0, 0.0, 1.0],
                        scale: [1.0; 3],
                    },
                },
                GltfNode {
                    name: Some("turbine_L".into()),
                    parent: None,
                    children: vec![],
                    mesh: None,
                    skin: None,
                    transform: NodeTransform::Trs {
                        translation: [0.0; 3],
                        rotation: [0.0, 0.0, 0.0, 1.0],
                        scale: [1.0; 3],
                    },
                },
            ],
            meshes: vec![],
            skeletons: vec![],
            animations: vec![],
            root_nodes: vec![0],
        };
        assert_eq!(scene.node_by_name("turbine_L"), Some(1));
        assert_eq!(scene.node_by_name("nonexistent"), None);
    }
}
