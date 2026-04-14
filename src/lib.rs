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

fn from_import(
    doc: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
) -> Result<GltfScene, Error> {
    let meshes = doc
        .meshes()
        .map(|m| GltfMesh {
            name: m.name().map(str::to_string),
            primitives: m
                .primitives()
                .map(|p| parse_primitive(&p, buffers, images))
                .collect(),
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
