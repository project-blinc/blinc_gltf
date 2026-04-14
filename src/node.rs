//! Scene-graph node parsing.
//!
//! Each glTF node carries either a 4×4 matrix *or* a TRS triple
//! (translation / rotation quaternion / scale). We preserve both forms
//! losslessly so the consumer can choose either static pose baking
//! (matrix) or runtime animation of T/R/S channels without a polar
//! decomposition.

use blinc_core::Mat4;

/// Parsed glTF node. Names and `parent`/`children` links form a tree
/// ready for DFS traversal.
#[derive(Debug, Clone)]
pub struct GltfNode {
    pub name: Option<String>,
    pub transform: NodeTransform,
    /// Index into the mesh list, if this node draws a mesh.
    pub mesh: Option<usize>,
    /// Index into the skeleton list, if this node is a skinned mesh.
    pub skin: Option<usize>,
    /// Parent node index, or `None` if this is a root.
    pub parent: Option<usize>,
    /// Child node indices (built by the loader for convenient DFS).
    pub children: Vec<usize>,
}

/// Node transform as stored in glTF — either a full 4×4 matrix or a
/// TRS triple. Preserved losslessly; call [`to_mat4`](Self::to_mat4) to
/// collapse it.
#[derive(Debug, Clone, Copy)]
pub enum NodeTransform {
    /// Translation, rotation (quaternion xyzw), scale.
    Trs {
        translation: [f32; 3],
        rotation: [f32; 4],
        scale: [f32; 3],
    },
    /// Raw column-major 4×4 matrix.
    Matrix([f32; 16]),
}

impl NodeTransform {
    /// Collapse the transform to a column-major 4×4 matrix suitable for
    /// composition via `Mat4::mul`.
    pub fn to_mat4(&self) -> Mat4 {
        match self {
            NodeTransform::Matrix(m) => Mat4 {
                cols: [
                    [m[0], m[1], m[2], m[3]],
                    [m[4], m[5], m[6], m[7]],
                    [m[8], m[9], m[10], m[11]],
                    [m[12], m[13], m[14], m[15]],
                ],
            },
            NodeTransform::Trs {
                translation: t,
                rotation: q,
                scale: s,
            } => {
                // Compose T · R · S for column-vector convention
                // (v' = T · R · S · v). Scale acts on the vertex
                // first, rotation next, translation last.
                let r = quat_to_mat4(*q);
                let scale = Mat4 {
                    cols: [
                        [s[0], 0.0, 0.0, 0.0],
                        [0.0, s[1], 0.0, 0.0],
                        [0.0, 0.0, s[2], 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                };
                let trans = Mat4::translation(t[0], t[1], t[2]);
                trans.mul(&r).mul(&scale)
            }
        }
    }
}

/// Parse every node in the document and compute child / parent links
/// in one pass. `children` lists are filled based on glTF's explicit
/// parent→children graph; `parent` is the inverse of that.
pub(crate) fn build_nodes(doc: &gltf::Document) -> Vec<GltfNode> {
    let mut nodes: Vec<GltfNode> = doc.nodes().map(|n| parse_node(&n)).collect();
    // glTF stores `children` per node; invert to populate `parent`.
    for node in doc.nodes() {
        for child in node.children() {
            if let Some(c) = nodes.get_mut(child.index()) {
                c.parent = Some(node.index());
            }
        }
    }
    nodes
}

fn parse_node(node: &gltf::Node) -> GltfNode {
    let transform = match node.transform() {
        gltf::scene::Transform::Matrix { matrix } => {
            // gltf returns a [[f32;4];4] matrix; flatten into a
            // column-major [f32;16].
            let mut flat = [0.0f32; 16];
            for (ci, col) in matrix.iter().enumerate() {
                for (ri, v) in col.iter().enumerate() {
                    flat[ci * 4 + ri] = *v;
                }
            }
            NodeTransform::Matrix(flat)
        }
        gltf::scene::Transform::Decomposed {
            translation,
            rotation,
            scale,
        } => NodeTransform::Trs {
            translation,
            rotation,
            scale,
        },
    };
    GltfNode {
        name: node.name().map(str::to_string),
        transform,
        mesh: node.mesh().map(|m| m.index()),
        skin: node.skin().map(|s| s.index()),
        parent: None,
        children: node.children().map(|c| c.index()).collect(),
    }
}

/// Quaternion `[x, y, z, w]` → column-major 4×4 rotation matrix.
fn quat_to_mat4(q: [f32; 4]) -> Mat4 {
    let [x, y, z, w] = q;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;

    // Row-form rotation matrix:
    //   [1-2(yy+zz),  2(xy-wz),    2(xz+wy),   0]
    //   [ 2(xy+wz),   1-2(xx+zz),  2(yz-wx),   0]
    //   [ 2(xz-wy),   2(yz+wx),    1-2(xx+yy), 0]
    //   [0,           0,           0,          1]
    //
    // Transposed into column-major storage: cols[c][r].
    Mat4 {
        cols: [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy), 0.0],
            [2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx), 0.0],
            [2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_quat_produces_identity_rotation() {
        let m = quat_to_mat4([0.0, 0.0, 0.0, 1.0]);
        for c in 0..4 {
            for r in 0..4 {
                let expected = if c == r { 1.0 } else { 0.0 };
                assert!(
                    (m.cols[c][r] - expected).abs() < 1e-6,
                    "mismatch at ({c},{r}): {}",
                    m.cols[c][r]
                );
            }
        }
    }

    #[test]
    fn trs_identity_is_identity_mat4() {
        let t = NodeTransform::Trs {
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        };
        let m = t.to_mat4();
        for c in 0..4 {
            for r in 0..4 {
                let expected = if c == r { 1.0 } else { 0.0 };
                assert!((m.cols[c][r] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn matrix_form_round_trips() {
        // Row-major-looking layout interpreted as column-major: a
        // translate (2, 3, 4).
        let m16 = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            2.0, 3.0, 4.0, 1.0, //
        ];
        let m = NodeTransform::Matrix(m16).to_mat4();
        assert_eq!(m.cols[3][0], 2.0);
        assert_eq!(m.cols[3][1], 3.0);
        assert_eq!(m.cols[3][2], 4.0);
    }
}
