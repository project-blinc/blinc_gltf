# blinc_gltf

glTF 2.0 loader for [Blinc](https://github.com/project-blinc/Blinc).

Parses `.glb` / `.gltf` files via the upstream
[`gltf`](https://crates.io/crates/gltf) crate and maps the result into
Blinc's native interchange types (`MeshData`, `Vertex`, `Material`,
`Skeleton`, `Bone`) plus a thin scene graph ready for
`blinc_canvas_kit::SceneKit3D`.

```rust
use blinc_gltf::load_glb;
use blinc_canvas_kit::SceneKit3D;

let bytes = std::fs::read("DamagedHelmet.glb")?;
let scene = load_glb(&bytes)?;

let kit = SceneKit3D::new("viewer");
let handles = scene.add_to(&kit);
println!("spawned {} primitives", handles.len());
```

## What's mapped (v0)

| glTF concept | Blinc type |
|---|---|
| Primitives (position / normal / UV0 / color / tangent / joints / weights / indices) | `MeshData` / `Vertex` |
| `pbrMetallicRoughness` factors + all five texture slots | `Material` |
| `alphaMode` (OPAQUE / MASK / BLEND) | `AlphaMode` |
| `KHR_materials_unlit` | `Material::unlit` |
| Nodes (TRS or matrix, parent/child links) | `GltfNode` + `NodeTransform` |
| Skins (joints + inverse bind matrices) | `Skeleton` / `Bone` in `GltfSkeleton` |
| Animations (channels + samplers + interpolation) | `GltfAnimation` (data only) |

## Runtime integration

- `SceneKit3D` spawning: [`GltfScene::add_to`](src/lib.rs) bakes each
  node's world transform into its vertices at spawn time (SceneKit3D's
  current transform API exposes only Y-rotation). Skins / animations
  are not applied — those feed into `blinc_skeleton`.
- Per-node world transforms: [`GltfScene::compute_world_transforms`]
  returns a `Vec<Mat4>` parallel to `scene.nodes` for callers who want
  their own scene graph.

## Status

Covers the common PBR static-geometry case — enough to load Khronos
sample assets like `BoxTextured`, `DamagedHelmet`, `FlightHelmet`.
Skin and animation data is parsed and exposed but runtime posing is
deferred to `blinc_skeleton`.

See [BACKLOG.md](./BACKLOG.md) for planned work: secondary UV sets,
morph targets, sparse accessors, compression extensions, and more PBR
extensions.

## License

Apache-2.0.
