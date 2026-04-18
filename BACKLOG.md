# blinc_gltf — Backlog

Outstanding work, grouped by area. Each entry notes **why** it matters
and **how** to approach it so items are pickable cold.

---

## Mesh attribute coverage

- [ ] **Secondary UV sets** (`TEXCOORD_1` and above)
  - **Why:** Occlusion + detail-tiled materials commonly route AO
    through UV1 so lightmaps don't fight the base color's atlas.
  - **How:** `Vertex` only has a single `uv: [f32; 2]`. Extend to
    `uv0` / `uv1` and thread the second set through `parse_primitive`
    via `reader.read_tex_coords(1)`. Renderer needs the shader change
    too.

- [x] **Morph targets.** `parse_primitive` reads `reader
  .read_morph_targets()` into `Vec<MorphTarget { delta_positions,
  delta_normals, delta_tangents }>` on the primitive (wrapped in
  `Arc<Vec>` for cheap per-frame shallow clones when the renderer
  stamps fresh weights). Runtime sampling lives in `blinc_skeleton`
  via `AnimatedProperty::MorphWeights`. Exercised by
  cutegirl_morph_demo (152 targets per face primitive) and
  strangler_demo (13 morph-weights channels for facial expression).

- [ ] **Sparse accessors**
  - **Why:** Highly optimized glTF exports pack mostly-default buffers
    as deltas off a dense baseline. The `gltf` crate handles them
    transparently on the reader side — verify nothing we do breaks
    that, add a regression test with a sparse-accessor fixture.

---

## Material extensions

- [ ] **`KHR_materials_ior` + `KHR_materials_transmission` + `KHR_materials_volume`**
  - **Why:** Glass, thin shells, sub-surface-ish. Modern Sketchfab /
    Khronos sample set relies heavily on these.
  - **How:** Read via `material.extension_value("KHR_materials_*")`
    — all three are JSON-schema extensions without helper methods.
    Extend `blinc_core::draw::Material` with `transmission`,
    `ior`, `thickness`, `attenuation_*`. Needs shader work downstream.

- [ ] **`KHR_materials_clearcoat`**
  - **Why:** Car paint, lacquered surfaces.
  - **How:** Same pattern as above; additional factor + normal +
    roughness textures on the material.

- [ ] **`KHR_texture_transform`**
  - **Why:** Rotating / tiling / offsetting a texture without
    re-authoring it.
  - **How:** Per-texture-info extension carrying `offset`, `rotation`,
    `scale`. Extend `TextureData` (or the material's texture slots)
    with a transform matrix applied in the shader.

- [x] **glTF `alphaCutoff` thread-through.** `Material.alpha_cutoff`
  carries the per-material cutoff (default 0.5 when absent); the
  mesh shader reads `material.alpha_cutoff` in the MASK branch.

- [x] **`KHR_materials_pbrSpecularGlossiness`** (legacy spec-gloss
  workflow). Feature enabled on the `gltf` dep so
  `extensionsRequired` no longer errors; `parse_material` converts
  to metallic-roughness at load time (diffuse → baseColor, specular
  → metallic via `(max(spec) - 0.04) / 0.96`, `1 - glossiness` →
  roughness, diffuseTexture → baseColorTexture). Channel-accurate
  texture re-bake is out of scope — visually indistinguishable for
  character assets (the common case) since specular is low and the
  factor path dominates.

- [x] **Alpha-mode auto-demotion at load.** Many DCC exporters flag
  every material as BLEND by default; weighted-blended OIT in the
  renderer then stacks them statistically and produces washed-out
  output. `parse_material` now analyses the base-color texture's
  alpha histogram once on load and demotes:
  `≥95% fully-opaque texels → OPAQUE`, `strictly binary α → MASK
  (cutoff 0.5)`, `partial α → keep BLEND`. Decisions log at
  `info` so misbehaving assets can be diagnosed with
  `RUST_LOG=blinc_gltf=info`.

---

## Format + performance

- [ ] **Draco geometry decompression** (`KHR_draco_mesh_compression`)
  - **Why:** Draco-compressed glTFs are ~5–10× smaller over the wire.
  - **How:** Enable the `gltf` crate's `extensions` feature and wire
    in the Draco decoder (optional dep; feature-gate on our side).

- [ ] **Meshopt** (`EXT_meshopt_compression`)
  - Similar rationale; Meshopt is the common alternative to Draco.

- [ ] **KTX2 / BasisU textures** (`KHR_texture_basisu`)
  - **Why:** Fraction of PNG/JPEG decode time; GPU-ready block
    compression. Universal texture format for delivery.
  - **How:** Requires a KTX2 decoder (`basisuniversal` via `basis-universal-rs`
    or similar). Feature-gated.

- [x] **Inverse-transpose for non-uniform-scale normals.**
  `bake_transform` now computes `inverse_transpose_upper3x3(m)` and
  routes normals + tangent-xyz through it. Degenerate matrices
  (|det| < 1e-8) fall back to the plain upper-3×3 so assets with a
  zero-scale axis degrade gracefully rather than blowing up.

---

## Scene graph + transforms

- [ ] **Full 4×4 transform API on SceneKit3D**
  - **Why:** `GltfScene::add_to` bakes transforms into vertices as a
    workaround. A `scene.set_transform(handle, Mat4)` upstream lets
    us preserve the scene graph for runtime re-posing (skins,
    animation, IK).
  - **How:** Extend `blinc_canvas_kit::scene3d::SceneObject` to
    optionally carry a raw `Mat4`; use it in `transform()` when
    present, fall back to TRS. Then rewrite `add_to` to push the
    world matrix per spawn.

- [ ] **Joint parenting through glue nodes**
  - Current skin parent derivation only works when glTF joints are
    direct children of each other. If a non-joint node sits between
    two joints (legal but rare), the deeper joint reads as a root.
    Fix: walk the full node parent chain during skin parse.

- [x] **Node names as a lookup table.**
  `GltfScene::node_by_name(&str) -> Option<usize>` — O(n) scan that
  returns the first match, consistent with how DCC tools name
  nodes. Downstream demos can replace manual `enumerate +
  filter_map` name scans (buster_drone's rotor subtree, lens
  emissive) with the one-liner once we use it in the demo.

---

## Loading ergonomics

- [ ] **Streaming / async loader**
  - Large glTFs blocking the UI thread is a real problem. Thread pool
    on native, `wasm-bindgen-futures` on web.

- [ ] **Progress callbacks**
  - Consumers of big assets want "loading 42%..." UI. Emit a callback
    with `{ meshes_loaded, total_meshes, textures_loaded, total_textures }`
    from the loader.

- [ ] **File-loader trait for external references**
  - `.gltf` JSON can reference external `.bin` / image files via
    relative URIs. `load_path` handles disk paths; wasm / network
    callers need a `trait FileLoader { fn load(&self, uri: &str) ->
    Result<Vec<u8>> }` abstraction.

---

## Non-goals

- **FBX / OBJ / Collada** support. One format per crate; other
  formats should live in their own `blinc_fbx` / `blinc_obj` crates.
- **glTF authoring / export.** Loader only. Editing / re-serializing
  is a much bigger surface and doesn't belong here.
- **Runtime animation sampling.** Parses animation data only; sampling
  lives in `blinc_skeleton`.
