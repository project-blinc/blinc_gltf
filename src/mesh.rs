//! Mesh primitive → [`MeshData`] decoding.
//!
//! Each glTF primitive becomes one `MeshData`. Attributes are pulled
//! from the accessor reader — we rely on the `gltf` crate's `utils`
//! feature to handle component-type casting (e.g. `u8`/`u16`/`u32`
//! joint indices), normalization, and sparse accessors automatically.

use blinc_core::draw::{MeshData, TextureData, Vertex};

use crate::material;

/// Parse a single primitive into a `MeshData`, materializing every
/// available vertex attribute.
///
/// `decoded_images` is the pre-decoded image pool produced by
/// [`crate::decode_images_once`]; every material reference to an
/// image index becomes a cheap `TextureData::clone()` sharing the same
/// underlying `Arc<[u8]>` byte buffer.
pub fn parse_primitive(
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
    decoded_images: &[Option<TextureData>],
) -> MeshData {
    let reader = primitive.reader(|b| Some(&buffers[b.index()].0));

    // ── Attribute collection ─────────────────────────────────────────
    //
    // POSITION is effectively required by glTF's validation — any
    // missing/invalid primitive falls back to zero vertices.
    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .map(|iter| iter.collect())
        .unwrap_or_default();
    let vertex_count = positions.len();

    let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|iter| iter.collect());

    // UVs outside [0, 1] are legal glTF — the spec expects the
    // sampler's wrap mode (as declared on the material's sampler) to
    // handle them. Blinc's mesh pipeline's sampler is configured for
    // `Repeat` wrap, so we pass the authored UVs through untouched.
    // Forcing UVs into `[0, 1]` via `u - u.floor()` would break any
    // triangle whose UV shell straddles the 0/1 seam — the shell
    // would snap and streak across the texture.
    let uvs: Option<Vec<[f32; 2]>> = reader
        .read_tex_coords(0)
        .map(|iter| iter.into_f32().collect());
    let colors: Option<Vec<[f32; 4]>> = reader
        .read_colors(0)
        .map(|iter| iter.into_rgba_f32().collect());
    let tangents: Option<Vec<[f32; 4]>> = reader.read_tangents().map(|iter| iter.collect());
    let joints: Option<Vec<[u16; 4]>> = reader.read_joints(0).map(|iter| iter.into_u16().collect());
    let weights: Option<Vec<[f32; 4]>> =
        reader.read_weights(0).map(|iter| iter.into_f32().collect());

    // ── Vertex assembly ──────────────────────────────────────────────
    //
    // Per-attribute fall-backs match `Vertex::new` defaults so missing
    // channels yield sensible values (up-facing normal, identity
    // tangent, white color, no skinning influence). This keeps the
    // invariant that a fully loaded mesh renders even with sparse
    // source data.
    let mut vertices = Vec::with_capacity(vertex_count);
    for i in 0..vertex_count {
        let mut v = Vertex::new(positions[i]);
        if let Some(ns) = &normals {
            v.normal = ns[i];
        }
        if let Some(us) = &uvs {
            v.uv = us[i];
        }
        if let Some(cs) = &colors {
            v.color = cs[i];
        }
        if let Some(ts) = &tangents {
            v.tangent = ts[i];
        }
        if let Some(js) = &joints {
            v.joints = [
                js[i][0] as u32,
                js[i][1] as u32,
                js[i][2] as u32,
                js[i][3] as u32,
            ];
        }
        if let Some(ws) = &weights {
            v.weights = ws[i];
        }
        vertices.push(v);
    }

    // ── Indices ──────────────────────────────────────────────────────
    //
    // Unindexed primitives (no glTF `indices` accessor) are flattened
    // to 0..vertex_count so `MeshData::indices` always holds a valid
    // triangle list. The render path can treat every mesh uniformly.
    let indices: Vec<u32> = reader
        .read_indices()
        .map(|iter| iter.into_u32().collect())
        .unwrap_or_else(|| (0..vertex_count as u32).collect());

    let material = material::parse_material(&primitive.material(), decoded_images);

    // Morph targets — per-vertex deltas on top of the base mesh.
    // `reader.read_morph_targets()` yields `(position, normal, tangent)`
    // iterator tuples per target; each inner iterator is already
    // positional-aligned with the base vertices, so we just collect.
    // Normals / tangents are optional per target.
    let morph_targets: Vec<blinc_core::draw::MorphTarget> = reader
        .read_morph_targets()
        .map(|(pos, nrm, tan)| {
            let delta_positions: Vec<[f32; 3]> = pos.map(|iter| iter.collect()).unwrap_or_default();
            let delta_normals: Option<Vec<[f32; 3]>> = nrm.map(|iter| iter.collect());
            let delta_tangents: Option<Vec<[f32; 3]>> = tan.map(|iter| iter.collect());
            blinc_core::draw::MorphTarget {
                delta_positions,
                delta_normals,
                delta_tangents,
            }
        })
        .collect();

    MeshData {
        vertices: std::sync::Arc::new(vertices),
        indices: std::sync::Arc::new(indices),
        material,
        skin: None, // Skinning data is provided per-frame by blinc_skeleton
        morph_targets: std::sync::Arc::new(morph_targets),
        // Empty until the demo / render path installs per-frame
        // weights from `blinc_skeleton::Pose::morph_weights`.
        morph_weights: Vec::new(),
    }
}
