//! glTF skin → [`blinc_core::draw::Skeleton`] mapping.
//!
//! Produces a bone hierarchy with inverse-bind matrices ready for a
//! runtime poser (e.g. `blinc_skeleton`). Joint parent links are
//! derived from each joint's glTF `children` — we look for children
//! that are themselves in the skin's joint set.
//!
//! # Known limitation
//!
//! If a skin's hierarchy threads *non-joint glue nodes* between two
//! joints (legal but rare), the deeper joint will not have its
//! ancestor joint recorded — it'll be parented to `None` and read as a
//! root by `blinc_skeleton`. Real skeletons exported from Blender /
//! Maya / MotionBuilder parent joints directly and don't hit this
//! path; tracked on the BACKLOG.

use blinc_core::draw::{Bone, Skeleton};

/// A parsed glTF skin. Holds a `Skeleton` for runtime posing plus the
/// list of source node indices (so callers can sync node transforms
/// with their own scene graph).
#[derive(Debug, Clone)]
pub struct GltfSkeleton {
    pub name: Option<String>,
    pub skeleton: Skeleton,
    /// `joint_nodes[i]` in the source document is the glTF node for
    /// `skeleton.bones[i]`. Useful when a renderer wants to sync node
    /// transforms animated elsewhere (FK overlay, procedural IK) with
    /// the bone list.
    pub joint_nodes: Vec<usize>,
}

pub(crate) fn parse_skin(skin: &gltf::Skin, buffers: &[gltf::buffer::Data]) -> GltfSkeleton {
    let joints: Vec<gltf::Node> = skin.joints().collect();
    let joint_nodes: Vec<usize> = joints.iter().map(|n| n.index()).collect();

    // Inverse-bind matrices: optional in glTF. When absent, every IBM
    // defaults to the identity — meaning the bones are already in
    // world space.
    let reader = skin.reader(|b| Some(&buffers[b.index()].0));
    let ibms: Vec<[f32; 16]> = reader
        .read_inverse_bind_matrices()
        .map(|iter| iter.map(flatten_col_major).collect())
        .unwrap_or_else(|| vec![identity_mat4(); joints.len()]);

    // Node-index → joint-index lookup so we can translate a joint's
    // glTF child into the matching bone slot.
    let joint_by_node: std::collections::HashMap<usize, usize> = joint_nodes
        .iter()
        .copied()
        .enumerate()
        .map(|(bone_idx, node_idx)| (node_idx, bone_idx))
        .collect();

    let mut bones: Vec<Bone> = joints
        .iter()
        .enumerate()
        .map(|(i, joint)| Bone {
            name: joint.name().map(str::to_string).unwrap_or_default(),
            parent: None,
            inverse_bind_matrix: ibms.get(i).copied().unwrap_or(identity_mat4()),
        })
        .collect();

    // Populate parent pointers from each joint's glTF children.
    for (parent_bone_idx, joint) in joints.iter().enumerate() {
        for child in joint.children() {
            if let Some(&child_bone_idx) = joint_by_node.get(&child.index()) {
                bones[child_bone_idx].parent = Some(parent_bone_idx);
            }
        }
    }

    GltfSkeleton {
        name: skin.name().map(str::to_string),
        skeleton: Skeleton { bones },
        joint_nodes,
    }
}

fn flatten_col_major(m: [[f32; 4]; 4]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for (ci, col) in m.iter().enumerate() {
        for (ri, v) in col.iter().enumerate() {
            out[ci * 4 + ri] = *v;
        }
    }
    out
}

fn identity_mat4() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, //
    ]
}
