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
#[cfg(feature = "platform-assets")]
pub fn load_asset(path: &str) -> Result<GltfScene, Error> {
    let bytes = blinc_platform::assets::load_asset(path)
        .map_err(|e| Error::Invalid(format!("asset load '{}': {}", path, e)))?;

    // Detect self-contained binary glTF by its magic header. `.glb`
    // files never reference external URIs, so they can go through the
    // existing slice-based path.
    if bytes.len() >= 4 && &bytes[0..4] == b"glTF" {
        return load_glb(&bytes);
    }

    // JSON glTF — parse the document first, then manually resolve
    // each buffer/image through `blinc_platform::assets` with the
    // main file's parent directory as the URI base.
    let base_dir: &str = path.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("");
    let gltf_obj = gltf::Gltf::from_slice(&bytes)?;
    let buffers = resolve_buffers(&gltf_obj, base_dir)?;
    let images = resolve_images(&gltf_obj, base_dir, &buffers)?;
    from_import(&gltf_obj.document, &buffers, &images)
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
) -> Result<Vec<gltf::image::Data>, Error> {
    use gltf::image::Format;
    use image::GenericImageView;

    let mut out = Vec::new();
    for image in gltf_obj.document.images() {
        let (encoded, mime): (Vec<u8>, Option<&str>) = match image.source() {
            gltf::image::Source::Uri { uri, mime_type } => {
                (resolve_uri_bytes(uri, base_dir)?, mime_type)
            }
            gltf::image::Source::View { view, mime_type } => {
                let buf = &buffers[view.buffer().index()].0;
                let begin = view.offset();
                let end = begin + view.length();
                (buf[begin..end].to_vec(), Some(mime_type))
            }
        };

        // Pick a decoder. Fall back to content-sniffing when the
        // mime type is missing — glTF allows either, and the image
        // crate's own `guess_format` handles png/jpg cleanly.
        let decoded = match mime {
            Some("image/png") => image::load_from_memory_with_format(&encoded, image::ImageFormat::Png),
            Some("image/jpeg") => image::load_from_memory_with_format(&encoded, image::ImageFormat::Jpeg),
            _ => image::load_from_memory(&encoded),
        }
        .map_err(|e| Error::Invalid(format!("image decode: {}", e)))?;

        let (w, h) = decoded.dimensions();
        let pixels = decoded.to_rgba8().into_raw();
        out.push(gltf::image::Data {
            pixels,
            format: Format::R8G8B8A8,
            width: w,
            height: h,
        });
    }
    Ok(out)
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

    let skeletons = doc
        .skins()
        .map(|s| skin::parse_skin(&s, buffers))
        .collect();

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
    let mut out = mesh.clone();
    for v in &mut out.vertices {
        v.position = transform_point(m, v.position);
        v.normal = normalize3(transform_direction(m, v.normal));
        // Tangent's xyz is a direction; w is handedness and passes
        // through unchanged.
        let t = transform_direction(m, [v.tangent[0], v.tangent[1], v.tangent[2]]);
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

fn transform_direction(m: &Mat4, d: [f32; 3]) -> [f32; 3] {
    // Direction vector — w = 0, so translation drops out. For perfect
    // normal transform under non-uniform scale we'd apply the
    // inverse-transpose of the upper-3×3; that's on the BACKLOG.
    let c = &m.cols;
    [
        c[0][0] * d[0] + c[1][0] * d[1] + c[2][0] * d[2],
        c[0][1] * d[0] + c[1][1] * d[1] + c[2][1] * d[2],
        c[0][2] * d[0] + c[1][2] * d[1] + c[2][2] * d[2],
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
    #[test]
    fn crate_compiles() {
        // Smoke: asserts the module graph wires together. Real
        // round-trip tests need fixture `.glb` files and belong in the
        // integration-test layer once we ship one.
    }
}
