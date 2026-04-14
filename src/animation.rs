//! glTF animation → typed keyframe data.
//!
//! Each animation channel targets one property (translation, rotation,
//! scale, or morph-target weights) on one node. The sampler holds the
//! keyframe times plus values, decoded into concrete `f32` arrays so
//! downstream (`blinc_skeleton`) can interpolate without re-parsing.
//!
//! **CubicSpline** values in glTF are stored as `[in_tangent, value,
//! out_tangent]` triples per keyframe; the sampler exposes them in
//! source order and flags `interpolation: Interpolation::CubicSpline`
//! so a consumer knows to read three values per time entry.

/// A parsed glTF animation clip.
#[derive(Debug, Clone)]
pub struct GltfAnimation {
    pub name: Option<String>,
    pub channels: Vec<AnimationChannel>,
}

/// One animation channel — a (target, sampler) pair.
#[derive(Debug, Clone)]
pub struct AnimationChannel {
    pub target: AnimationTarget,
    pub sampler: AnimationSampler,
}

/// What this channel animates.
#[derive(Debug, Clone, Copy)]
pub struct AnimationTarget {
    /// Index into `GltfScene::nodes`.
    pub node: usize,
    pub property: AnimatedProperty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimatedProperty {
    Translation,
    Rotation,
    Scale,
    /// Morph-target weights. When this is the property, the sampler's
    /// `KeyframeValues::Scalars` array holds `N × weight_count` floats
    /// (contiguous per-keyframe weight blocks).
    MorphWeights,
}

#[derive(Debug, Clone)]
pub struct AnimationSampler {
    /// Keyframe times in seconds.
    pub times: Vec<f32>,
    pub values: KeyframeValues,
    pub interpolation: Interpolation,
}

/// Keyframe values, typed to the target property.
#[derive(Debug, Clone)]
pub enum KeyframeValues {
    /// Per-keyframe `[x, y, z]`. Translation and scale use this.
    Vec3(Vec<[f32; 3]>),
    /// Per-keyframe `[x, y, z, w]` quaternion. Rotation uses this.
    Vec4(Vec<[f32; 4]>),
    /// Per-keyframe scalars — used for morph-target weights. Values
    /// are laid out contiguously: `times.len() * morph_weight_count`.
    Scalars(Vec<f32>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// Hold the previous keyframe value until the next one.
    Step,
    /// Linear interpolation between adjacent keyframes.
    Linear,
    /// Cubic Hermite interpolation with per-keyframe in/out tangents.
    /// The sampler's `values` hold `[tangent_in, value, tangent_out]`
    /// triples per keyframe time.
    CubicSpline,
}

pub(crate) fn parse_animation(
    anim: &gltf::Animation,
    buffers: &[gltf::buffer::Data],
) -> GltfAnimation {
    let channels = anim
        .channels()
        .map(|ch| parse_channel(&ch, buffers))
        .collect();
    GltfAnimation {
        name: anim.name().map(str::to_string),
        channels,
    }
}

fn parse_channel(ch: &gltf::animation::Channel, buffers: &[gltf::buffer::Data]) -> AnimationChannel {
    let reader = ch.reader(|b| Some(&buffers[b.index()].0));
    let times: Vec<f32> = reader
        .read_inputs()
        .map(|iter| iter.collect())
        .unwrap_or_default();

    let (values, property) = match reader.read_outputs() {
        Some(outputs) => match outputs {
            gltf::animation::util::ReadOutputs::Translations(iter) => {
                (KeyframeValues::Vec3(iter.collect()), AnimatedProperty::Translation)
            }
            gltf::animation::util::ReadOutputs::Rotations(r) => {
                // `r.into_f32()` yields per-keyframe `[x, y, z, w]`
                // quaternions regardless of the source component
                // type (u8/i8/u16/i16/f32 are all valid in glTF).
                (KeyframeValues::Vec4(r.into_f32().collect()), AnimatedProperty::Rotation)
            }
            gltf::animation::util::ReadOutputs::Scales(iter) => {
                (KeyframeValues::Vec3(iter.collect()), AnimatedProperty::Scale)
            }
            gltf::animation::util::ReadOutputs::MorphTargetWeights(w) => (
                KeyframeValues::Scalars(w.into_f32().collect()),
                AnimatedProperty::MorphWeights,
            ),
        },
        None => (KeyframeValues::Scalars(Vec::new()), AnimatedProperty::Translation),
    };

    let interpolation = match ch.sampler().interpolation() {
        gltf::animation::Interpolation::Step => Interpolation::Step,
        gltf::animation::Interpolation::Linear => Interpolation::Linear,
        gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
    };

    AnimationChannel {
        target: AnimationTarget {
            node: ch.target().node().index(),
            property,
        },
        sampler: AnimationSampler {
            times,
            values,
            interpolation,
        },
    }
}
