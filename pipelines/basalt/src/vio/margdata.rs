use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

use serde::{de::Error as _, Deserialize, Deserializer, Serialize, Serializer};

/// Versioned on-disk contract for the Rust Basalt marginalization artifact.
/// Versions 1--3 remain readable through serde defaults, while all newly
/// emitted records use version 4 and carry the complete mixed AOM/FEJ/target
/// trace plus the raw optical-flow input images needed by the NFR mapper.
///
/// Current on-wire emitter version.  Schema 3 remains readable through
/// [`MARGDATA_SCHEMA_VERSION_V3`] and serde defaults.  Newly emitted
/// estimator records carry the schema-4 FEJ sidecars and satisfy the strict
/// schema-4 contract enforced by [`MargData::validate_contract`].
pub const MARGDATA_SCHEMA_VERSION: u32 = 4;
pub const MARGDATA_PREVIOUS_SCHEMA_VERSION: u32 = 2;
pub const MARGDATA_LEGACY_SCHEMA_VERSION: u32 = 1;
pub const MARGDATA_SCHEMA_VERSION_V3: u32 = 3;
pub const MARGDATA_SCHEMA_VERSION_V4: u32 = MARGDATA_SCHEMA_VERSION;
pub const MARGDATA_SCHEMA_V4: u32 = MARGDATA_SCHEMA_VERSION;
const POSE_BLOCK_DOF: usize = 6;
const STATE_BLOCK_DOF: usize = 15;

/// Immutable identity of the pinned native mapper event used by a paired
/// schema-4 capture.
///
/// This metadata is never synthesized by the estimator. A diagnostic caller
/// must load it before replay and may attach it only to the matching emitted
/// mapper packet through
/// [`MargData::write_mapper_packet_json_with_native_companion_identity`].
/// Keeping the identity outside [`MargData`] preserves the upstream queue
/// packet unless the explicit paired-capture API is used.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeCompanionIdentity {
    pub run_uuid: String,
    pub event_ordinal: u64,
    pub event_state_timestamp_ns: i64,
    pub primary_kf_timestamp_ns: i64,
    pub packet_filename: String,
    pub packet_sha256: String,
    pub frame_map_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatrixData {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}
impl MatrixData {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Option<Self> {
        if data.len() != rows.checked_mul(cols)? {
            None
        } else {
            Some(Self { rows, cols, data })
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameStateData {
    pub frame_id: u64,
    pub timestamp_ns: i64,
    pub pose: [f64; 7],
    #[serde(default = "zero_vec3")]
    pub velocity: [f64; 3],
    #[serde(default = "zero_vec3")]
    pub gyro_bias: [f64; 3],
    #[serde(default = "zero_vec3")]
    pub accel_bias: [f64; 3],
    #[serde(default)]
    pub linearized: bool,
    pub is_keyframe: bool,
    pub is_latest: bool,
}

const fn zero_vec3() -> [f64; 3] {
    [0.0; 3]
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FramePoseData {
    pub frame_id: u64,
    pub timestamp_ns: i64,
    pub pose: [f64; 7],
    pub is_keyframe: bool,
}

/// The four navigation quantities serialized by Basalt's
/// `PoseVelBiasState`.  The pose uses the Rust MargData wire convention
/// `[tx, ty, tz, qw, qx, qy, qz]`; a native adapter must construct Sophus
/// explicitly instead of assuming this is Sophus cereal's field order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NavStateData {
    pub pose: [f64; 7],
    #[serde(default = "zero_vec3")]
    pub velocity: [f64; 3],
    #[serde(default = "zero_vec3")]
    pub gyro_bias: [f64; 3],
    #[serde(default = "zero_vec3")]
    pub accel_bias: [f64; 3],
}

/// JSON representation of upstream `PoseVelBiasStateWithLin`.
///
/// The timestamp is the key in `MargData::frame_states_fej`, matching the
/// upstream `aligned_map<int64_t, ...>` key and its
/// `state_linearized.t_ns` value.  `state_current` is intentionally retained
/// even when `linearized` is false: upstream's `applyInc` can leave that
/// private stored-current value unchanged while moving the linearized state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoseVelBiasStateWithLinData {
    pub state_linearized: NavStateData,
    pub state_current: NavStateData,
    #[serde(default = "zero_vec15")]
    pub delta: [f64; 15],
    #[serde(default)]
    pub linearized: bool,
}

/// JSON representation of upstream `PoseStateWithLin`.
///
/// The timestamp is the key in `MargData::frame_poses_fej`, matching
/// `pose_linearized.t_ns`.  The explicit current pose avoids reconstructing
/// an SO(3) increment with a different tangent or scalar convention.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoseStateWithLinData {
    pub pose_linearized: [f64; 7],
    pub pose_current: [f64; 7],
    #[serde(default = "zero_vec6")]
    pub delta: [f64; 6],
    #[serde(default)]
    pub linearized: bool,
}

const fn zero_vec6() -> [f64; 6] {
    [0.0; 6]
}

const fn zero_vec7() -> [f64; 7] {
    [0.0; 7]
}

const fn zero_vec15() -> [f64; 15] {
    [0.0; 15]
}

const fn is_false(value: &bool) -> bool {
    !*value
}

fn is_empty_map<K, V>(value: &BTreeMap<K, V>) -> bool {
    value.is_empty()
}

fn is_empty_map_ref<K, V>(value: &&BTreeMap<K, V>) -> bool {
    value.is_empty()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AomBlockData {
    pub frame_id: u64,
    pub offset: usize,
    pub dof: usize,
    /// `pose` for a retained pose-only keyframe, `state` for a full
    /// pose/velocity/bias block.  The prior-only `state_pose` kind is not a
    /// valid active AOM block and is carried by [`PriorData::block_kinds`].
    pub kind: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarginalizationTargets {
    pub poses_to_marg: Vec<u64>,
    pub states_to_marg_all: Vec<u64>,
    pub states_to_marg_vel_bias: Vec<u64>,
    pub lost_landmarks: Vec<u64>,
}

impl Default for MarginalizationTargets {
    fn default() -> Self {
        Self {
            poses_to_marg: Vec::new(),
            states_to_marg_all: Vec::new(),
            states_to_marg_vel_bias: Vec::new(),
            lost_landmarks: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriorData {
    pub frame_ids: Vec<u64>,
    /// Explicit per-block kind for schema 4.  Legacy schema-3 records may
    /// omit this field; the checked legacy path then retains its historical
    /// all-state interpretation, while schema 4 rejects the omission.  The
    /// explicit schema-4 values are `pose` (6 DoF from `frame_poses`),
    /// `state_pose` (the first 6 DoF of `frame_states`), and `state` (all 15
    /// DoF from `frame_states`).
    #[serde(default)]
    pub block_kinds: Vec<String>,
    pub jacobian: MatrixData,
    pub rhs: Vec<f64>,
    pub fej_point: Vec<f64>,
}

/// A bit-oriented representation of one diagnostic prior payload.
///
/// `PriorData` deliberately keeps the public JSON values as `f64` for
/// compatibility with the schema-4 mapper record.  The estimator's native
/// compatibility mode, however, materializes the prior in `f32` and widens
/// it only at the Rust API boundary.  Keeping the source `f32` words beside
/// the widened values lets an oracle distinguish a real arithmetic mismatch
/// from JSON parsing/formatting noise without feeding the sidecar back into
/// the solver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriorDiagnosticBits {
    pub jacobian: MatrixBits,
    pub rhs: Vec<String>,
    pub fej_point: Vec<String>,
}

/// A dense matrix represented in the same column-major order as nalgebra's
/// iterator and the native `Eigen::Matrix` cereal/diagnostic dumps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatrixBits {
    pub rows: usize,
    pub cols: usize,
    pub layout: String,
    pub bits: Vec<String>,
}

/// One schema-4 pose FEJ record plus the exact f32 words used by the
/// UpstreamF32 estimator boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoseFejDiagnostic {
    pub frame_id: u64,
    pub timestamp_ns: i64,
    pub raw: PoseStateWithLinData,
    /// Explicit aliases keep the four PoseStateWithLin branches independent
    /// in the diagnostic wire contract. `raw` remains for v1 compatibility.
    #[serde(default = "zero_vec7")]
    pub raw_linearized: [f64; 7],
    #[serde(default = "zero_vec7")]
    pub raw_current: [f64; 7],
    #[serde(default = "zero_vec6")]
    pub raw_delta: [f64; 6],
    /// Value exposed by native `PoseStateWithLin::getPose()`: non-linearized
    /// uses raw linearized; linearized applies delta at the f32 boundary.
    #[serde(default = "zero_vec7")]
    pub effective_get_pose: [f64; 7],
    #[serde(default)]
    pub effective_get_pose_f32_bits: Vec<String>,
    pub f32_bits: FejPoseBits,
}

/// One schema-4 navigation FEJ record plus the exact f32 words used by the
/// UpstreamF32 estimator boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateFejDiagnostic {
    pub frame_id: u64,
    pub timestamp_ns: i64,
    pub raw: PoseVelBiasStateWithLinData,
    #[serde(default)]
    pub raw_state_linearized: Option<NavStateData>,
    #[serde(default)]
    pub raw_state_current: Option<NavStateData>,
    #[serde(default = "zero_vec15")]
    pub raw_delta: [f64; 15],
    /// Pose-only value exposed by native `getPoseStateWithLin().getPose()`.
    #[serde(default = "zero_vec7")]
    pub effective_get_pose: [f64; 7],
    #[serde(default)]
    pub effective_get_pose_f32_bits: Vec<String>,
    pub f32_bits: FejStateBits,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FejPoseBits {
    pub pose_linearized: Vec<String>,
    pub pose_current: Vec<String>,
    pub delta: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FejStateBits {
    pub state_linearized: FejNavBits,
    pub state_current: FejNavBits,
    pub delta: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FejNavBits {
    pub pose: Vec<String>,
    pub velocity: Vec<String>,
    pub gyro_bias: Vec<String>,
    pub accel_bias: Vec<String>,
}

/// A diagnostic-only join record for one emitted mapper MargData event.
///
/// This is intentionally a separate JSONL payload rather than additional
/// fields on the queue-facing cereal-compatible packet.  It captures the
/// pre/post square-root prior and the raw FEJ current/linearized/delta
/// branches at the same pre-shift boundary as `MargData`, while retaining a
/// deterministic event identity (selected KFs, timestamp map, AOM order and
/// membership).  The normal estimator does not build this value unless the
/// explicit `VISLOC_BASALT_MARGDATA_SIDECAR` diagnostic path is configured.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MargDataDiagnosticSidecar {
    pub schema: String,
    pub event_ordinal: usize,
    pub event_frame_id: u64,
    pub event_timestamp_ns: i64,
    pub boundary_state: Option<FrameIdentity>,
    pub selected_keyframes: Vec<FrameIdentity>,
    pub kfs_all: Vec<FrameIdentity>,
    pub aom_order: Vec<AomBlockData>,
    pub pre_prior: Option<PriorData>,
    pub post_prior: Option<PriorData>,
    pub pre_prior_f32_bits: Option<PriorDiagnosticBits>,
    pub post_prior_f32_bits: Option<PriorDiagnosticBits>,
    pub pose_fej: Vec<PoseFejDiagnostic>,
    pub state_fej: Vec<StateFejDiagnostic>,
    pub scalar_mode: String,
    pub wire_order: String,
    pub normalization: String,
    pub marg_data_stable_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameIdentity {
    pub frame_id: u64,
    pub timestamp_ns: i64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OfObservationData {
    pub frame_id: u64,
    pub track_id: u64,
    pub camera_id: u16,
    /// Optical-flow keypoint center in source-image pixel coordinates.
    /// This is not a normalized camera bearing.
    pub x: f64,
    pub y: f64,
}

/// One raw image retained from Basalt's `OpticalFlowInput::img_data`.
///
/// The mapper consumes the input image itself, not a pyramid or an 8-bit
/// feature image.  Keeping the samples as `u16` therefore preserves the
/// EuRoC reader's exact `u8 << 8` promotion (and genuine 16-bit PNG values)
/// through a MargData round-trip.  The wire representation is a canonical
/// little-endian byte string encoded as base64; deserialization also accepts
/// the older JSON-array spelling so a hand-authored fixture can be upgraded
/// without loss.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OfImageData {
    /// Rust frame index corresponding to the upstream image timestamp.
    pub frame_id: u64,
    /// Camera timestamp used as the key in upstream `img_data`.
    pub timestamp_ns: i64,
    /// Upstream `TimeCamId::cam_id` (0 for cam0, 1 for cam1 in EuRoC).
    pub camera_id: u16,
    pub width: u32,
    pub height: u32,
    #[serde(
        serialize_with = "serialize_u16_samples",
        deserialize_with = "deserialize_u16_samples"
    )]
    pub data: Vec<u16>,
}

impl OfImageData {
    /// Creates a validated image record.  Keeping this constructor fallible
    /// prevents an accidental dimension truncation when adapting a
    /// platform-sized `RawU16Image`.
    pub fn new(
        frame_id: u64,
        timestamp_ns: i64,
        camera_id: u16,
        width: usize,
        height: usize,
        data: Vec<u16>,
    ) -> Option<Self> {
        Some(Self {
            frame_id,
            timestamp_ns,
            camera_id,
            width: u32::try_from(width).ok()?,
            height: u32::try_from(height).ok()?,
            data,
        })
        .filter(Self::is_valid)
    }

    /// Returns the FNV-1a hash of the exact little-endian sample stream.
    /// This is intentionally independent of the JSON/base64 representation
    /// and is useful for source-image parity checks.
    pub fn sample_hash(&self) -> u64 {
        self.data
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .fold(1469598103934665603u64, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(1099511628211)
            })
    }

    pub fn is_valid(&self) -> bool {
        self.width != 0
            && self.height != 0
            && self
                .width
                .checked_mul(self.height)
                .is_some_and(|count| usize::try_from(count).ok() == Some(self.data.len()))
    }
}

fn serialize_u16_samples<S>(samples: &[u16], serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let bytes = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    serializer.serialize_str(&BASE64.encode(bytes))
}

fn deserialize_u16_samples<'de, D>(deserializer: D) -> Result<Vec<u16>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Samples {
        /// Compact v3 spelling.
        Base64(String),
        /// Compatibility spelling accepted for manually authored fixtures.
        Json(Vec<u16>),
    }

    let samples = Samples::deserialize(deserializer)?;
    match samples {
        Samples::Json(values) => Ok(values),
        Samples::Base64(encoded) => {
            let bytes = BASE64
                .decode(encoded)
                .map_err(|error| D::Error::custom(format!("invalid u16 image base64: {error}")))?;
            let chunks = bytes.chunks_exact(2);
            if !chunks.remainder().is_empty() {
                return Err(D::Error::custom(
                    "u16 image base64 payload has an odd byte count",
                ));
            }
            Ok(chunks
                .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                .collect())
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MargData {
    pub schema_version: u32,
    pub aom_sqrt_jacobian: MatrixData,
    pub aom_sqrt_rhs: Vec<f64>,
    pub aom_abs_h: Option<MatrixData>,
    pub aom_abs_b: Option<Vec<f64>>,
    /// Explicit pose-only table. Upstream stores this separately from
    /// `frame_states`; deriving it from keyframe IDs loses timestamp/pose
    /// values and cannot reproduce the AOM order on a mapper round-trip.
    #[serde(default)]
    pub frame_poses: Vec<FramePoseData>,
    pub frame_states: Vec<FrameStateData>,
    pub keyframes: Vec<u64>,
    pub kf_to_marg: Vec<(u64, u64)>,
    /// Upstream names retained alongside the compatibility aliases above.
    #[serde(default)]
    pub kfs_all: Vec<u64>,
    #[serde(default)]
    pub kfs_to_marg: Vec<u64>,
    #[serde(default)]
    pub aom_order: Vec<AomBlockData>,
    #[serde(default)]
    pub marginalization: MarginalizationTargets,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior: Option<PriorData>,
    #[serde(default)]
    pub row_counts: [usize; 4],
    #[serde(default)]
    pub of_observations: Vec<OfObservationData>,
    /// Raw image payloads for the keyframe timestamps represented by this
    /// artifact.  Empty is valid for legacy/synthetic records that did not
    /// carry optical-flow input images.
    #[serde(default)]
    pub of_images: Vec<OfImageData>,
    /// Upstream's `frame_poses` map keyed by `pose_linearized.t_ns`.  These
    /// sidecars are omitted from schema-3 records and required for schema 4.
    #[serde(default, skip_serializing_if = "is_empty_map")]
    pub frame_poses_fej: BTreeMap<i64, PoseStateWithLinData>,
    /// Upstream's `frame_states` map keyed by `state_linearized.t_ns`.  These
    /// sidecars retain both stored-current and linearized values.
    #[serde(default, skip_serializing_if = "is_empty_map")]
    pub frame_states_fej: BTreeMap<i64, PoseVelBiasStateWithLinData>,
    /// Set only when both FEJ tables are complete and were populated from
    /// retained upstream sidecars.  Schema-3 defaults to false.
    #[serde(default, skip_serializing_if = "is_false")]
    pub fej_complete: bool,
    /// Optional poses, keyed by frame id, that the mapper uses instead of
    /// `frame_poses` and `frame_states` as the measurements of the factors it
    /// recovers from this packet. Outside the FEJ contract: the factor
    /// information still comes from the packet's own linearization. Empty
    /// keeps the packet poses.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub measurement_poses: Vec<FramePoseData>,
    pub used_imu: bool,
    pub provenance_version: String,
}

/// Borrowed queue-facing view used by the hot-path packet writer.
///
/// `to_mapper_packet` is intentionally retained as an owned compatibility
/// helper, but cloning a packet also clones all retained images and dense
/// matrices. The upstream queue packet differs from the diagnostic record
/// only by omitting `prior` and using its queue provenance string, so the
/// writer can serialize those fields directly from the diagnostic record.
#[derive(Serialize)]
struct MapperPacketRef<'a> {
    schema_version: u32,
    aom_sqrt_jacobian: &'a MatrixData,
    aom_sqrt_rhs: &'a Vec<f64>,
    aom_abs_h: Option<&'a MatrixData>,
    aom_abs_b: Option<&'a Vec<f64>>,
    frame_poses: &'a Vec<FramePoseData>,
    frame_states: &'a Vec<FrameStateData>,
    keyframes: &'a Vec<u64>,
    kf_to_marg: &'a Vec<(u64, u64)>,
    kfs_all: &'a Vec<u64>,
    kfs_to_marg: &'a Vec<u64>,
    aom_order: &'a Vec<AomBlockData>,
    marginalization: &'a MarginalizationTargets,
    row_counts: [usize; 4],
    of_observations: &'a Vec<OfObservationData>,
    of_images: &'a Vec<OfImageData>,
    #[serde(skip_serializing_if = "is_empty_map_ref")]
    frame_poses_fej: &'a BTreeMap<i64, PoseStateWithLinData>,
    #[serde(skip_serializing_if = "is_empty_map_ref")]
    frame_states_fej: &'a BTreeMap<i64, PoseVelBiasStateWithLinData>,
    #[serde(skip_serializing_if = "is_false")]
    fej_complete: bool,
    used_imu: bool,
    provenance_version: &'static str,
}

#[derive(Serialize)]
struct MapperPacketWithNativeIdentityRef<'a> {
    #[serde(flatten)]
    packet: MapperPacketRef<'a>,
    native_companion_identity: &'a NativeCompanionIdentity,
    #[serde(skip_serializing_if = "Option::is_none")]
    prior: Option<&'a PriorData>,
}

const MAPPER_PACKET_PROVENANCE_VERSION: &str = "basalt-0f3b2b52-mapper-packet-v1";

fn finite_array<const N: usize>(values: &[f64; N]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn finite_nav(value: &NavStateData) -> bool {
    finite_array(&value.pose)
        && finite_array(&value.velocity)
        && finite_array(&value.gyro_bias)
        && finite_array(&value.accel_bias)
}

impl MargData {
    const fn mapper_packet_ref(&self) -> MapperPacketRef<'_> {
        MapperPacketRef {
            schema_version: self.schema_version,
            aom_sqrt_jacobian: &self.aom_sqrt_jacobian,
            aom_sqrt_rhs: &self.aom_sqrt_rhs,
            aom_abs_h: self.aom_abs_h.as_ref(),
            aom_abs_b: self.aom_abs_b.as_ref(),
            frame_poses: &self.frame_poses,
            frame_states: &self.frame_states,
            keyframes: &self.keyframes,
            kf_to_marg: &self.kf_to_marg,
            kfs_all: &self.kfs_all,
            kfs_to_marg: &self.kfs_to_marg,
            aom_order: &self.aom_order,
            marginalization: &self.marginalization,
            row_counts: self.row_counts,
            of_observations: &self.of_observations,
            of_images: &self.of_images,
            frame_poses_fej: &self.frame_poses_fej,
            frame_states_fej: &self.frame_states_fej,
            fej_complete: self.fej_complete,
            used_imu: self.used_imu,
            provenance_version: MAPPER_PACKET_PROVENANCE_VERSION,
        }
    }

    /// Empty diagnostic placeholder used when a caller explicitly disables
    /// MargData retention.  The estimator's internal square-root prior is
    /// still updated; only the owned on-wire snapshot is omitted.
    pub(crate) fn empty() -> Self {
        Self {
            schema_version: MARGDATA_SCHEMA_VERSION_V3,
            aom_sqrt_jacobian: MatrixData {
                rows: 0,
                cols: 0,
                data: Vec::new(),
            },
            aom_sqrt_rhs: Vec::new(),
            aom_abs_h: None,
            aom_abs_b: None,
            frame_poses: Vec::new(),
            frame_states: Vec::new(),
            keyframes: Vec::new(),
            kf_to_marg: Vec::new(),
            kfs_all: Vec::new(),
            kfs_to_marg: Vec::new(),
            aom_order: Vec::new(),
            marginalization: MarginalizationTargets::default(),
            prior: None,
            row_counts: [0; 4],
            of_observations: Vec::new(),
            of_images: Vec::new(),
            frame_poses_fej: BTreeMap::new(),
            frame_states_fej: BTreeMap::new(),
            fej_complete: false,
            measurement_poses: Vec::new(),
            used_imu: false,
            provenance_version: "basalt-disabled".into(),
        }
    }

    /// Recover the mixed absolute order from the versioned artifact without
    /// adding a second parallel frame table: pose-only keyframes are the KF
    /// IDs absent from `frame_states`, and all emitted full states follow.
    /// This is the same `frame_poses`-then-`frame_states` order used by the
    /// upstream `AbsOrderMap`.
    pub fn aom_block_order(&self) -> Vec<(u64, usize)> {
        if !self.aom_order.is_empty() {
            return self
                .aom_order
                .iter()
                .map(|block| (block.frame_id, block.dof))
                .collect();
        }
        let state_ids = self
            .frame_states
            .iter()
            .map(|state| state.frame_id)
            .collect::<std::collections::BTreeSet<_>>();
        let mut order = self
            .keyframes
            .iter()
            .copied()
            .filter(|id| !state_ids.contains(id))
            .map(|id| (id, 6))
            .collect::<Vec<_>>();
        order.extend(self.frame_states.iter().map(|state| (state.frame_id, 15)));
        order
    }

    pub fn aom_dof(&self) -> usize {
        self.aom_block_order().iter().map(|(_, dof)| *dof).sum()
    }

    /// Whether this record is an upstream mapper-queue packet.
    ///
    /// Basalt computes marginalization on every eligible frame, but pushes a
    /// `MargData` record only when at least one keyframe is selected for
    /// removal (`kfs_to_marg` is non-empty).  The estimator still exposes a
    /// diagnostic record on state-only frames; callers writing the mapper
    /// stream must use this predicate to preserve that queue contract.
    pub fn is_mapper_packet(&self) -> bool {
        !self.kfs_to_marg.is_empty()
    }

    /// Returns this record when it has the exact upstream mapper-queue
    /// emission condition (`kfs_to_marg` is non-empty).
    ///
    /// The estimator deliberately keeps one complete [`MargData`] value for
    /// every processed frame so that per-frame diagnostics remain available.
    /// Callers that write the upstream mapper stream must use this view (or
    /// [`Self::is_mapper_packet`]); state-only marginalization records are not
    /// queue packets even though they contain a valid AOM snapshot.
    pub fn as_mapper_packet(&self) -> Option<&Self> {
        self.is_mapper_packet().then_some(self)
    }

    /// Materializes the queue-facing packet without Rust's diagnostic-only
    /// carried prior.  The pinned upstream `MargData` contains the absolute
    /// H/b snapshot but has no serialized square-root prior; Rust retains that
    /// prior on the per-frame diagnostic record so the next active window can
    /// still be audited.  This method is the owned boundary used by writers.
    pub fn to_mapper_packet(&self) -> Option<Self> {
        let mut packet = self.as_mapper_packet()?.clone();
        packet.prior = None;
        packet.provenance_version = MAPPER_PACKET_PROVENANCE_VERSION.into();
        Some(packet)
    }

    pub fn validate_contract(&self) -> Result<(), String> {
        if !matches!(
            self.schema_version,
            MARGDATA_SCHEMA_VERSION
                | MARGDATA_SCHEMA_VERSION_V3
                | MARGDATA_PREVIOUS_SCHEMA_VERSION
                | MARGDATA_LEGACY_SCHEMA_VERSION
        ) {
            return Err(format!(
                "unsupported schema_version {}",
                self.schema_version
            ));
        }

        let jacobian_size = self
            .aom_sqrt_jacobian
            .rows
            .checked_mul(self.aom_sqrt_jacobian.cols)
            .ok_or_else(|| "square-root Jacobian dimensions overflow".to_owned())?;
        if self.aom_sqrt_jacobian.data.len() != jacobian_size {
            return Err("square-root Jacobian data length does not match dimensions".to_owned());
        }
        if self.aom_sqrt_rhs.len() != self.aom_sqrt_jacobian.rows {
            return Err("square-root RHS length does not match Jacobian rows".to_owned());
        }
        if self
            .aom_sqrt_jacobian
            .data
            .iter()
            .chain(self.aom_sqrt_rhs.iter())
            .any(|value| !value.is_finite())
        {
            return Err("square-root Jacobian/RHS contains a non-finite value".to_owned());
        }

        if let Some(matrix) = &self.aom_abs_h {
            let size = matrix
                .rows
                .checked_mul(matrix.cols)
                .ok_or_else(|| "absolute H dimensions overflow".to_owned())?;
            if matrix.rows != matrix.cols
                || matrix.rows != self.aom_sqrt_jacobian.cols
                || matrix.data.len() != size
            {
                return Err("absolute H dimensions do not match the AOM".to_owned());
            }
            if matrix.data.iter().any(|value| !value.is_finite()) {
                return Err("absolute H contains a non-finite value".to_owned());
            }
        }
        if let Some(vector) = &self.aom_abs_b {
            if vector.len() != self.aom_sqrt_jacobian.cols {
                return Err("absolute b length does not match the AOM".to_owned());
            }
            if vector.iter().any(|value| !value.is_finite()) {
                return Err("absolute b contains a non-finite value".to_owned());
            }
        }

        // Upstream stores pose and state maps keyed by frame ID, with a unique
        // timestamp for each frame.  The Rust vectors are intentionally kept
        // as vectors for JSON compatibility, so enforce the map invariants at
        // the boundary instead of relying on callers to canonicalize them.
        let mut frame_ids = BTreeMap::<u64, (&str, i64)>::new();
        let mut frame_timestamps = BTreeMap::<i64, u64>::new();
        let mut pose_ids = BTreeSet::new();
        let mut state_ids = BTreeSet::new();
        for pose in &self.frame_poses {
            if pose.pose.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "pose {} contains a non-finite value",
                    pose.frame_id
                ));
            }
            if frame_ids
                .insert(pose.frame_id, ("pose", pose.timestamp_ns))
                .is_some()
            {
                return Err(format!("duplicate frame ID {}", pose.frame_id));
            }
            if frame_timestamps
                .insert(pose.timestamp_ns, pose.frame_id)
                .is_some()
            {
                return Err(format!("duplicate frame timestamp {}", pose.timestamp_ns));
            }
            pose_ids.insert(pose.frame_id);
        }
        for state in &self.frame_states {
            if state
                .pose
                .iter()
                .chain(state.velocity.iter())
                .chain(state.gyro_bias.iter())
                .chain(state.accel_bias.iter())
                .any(|value| !value.is_finite())
            {
                return Err(format!(
                    "state {} contains a non-finite value",
                    state.frame_id
                ));
            }
            if frame_ids
                .insert(state.frame_id, ("state", state.timestamp_ns))
                .is_some()
            {
                return Err(format!("duplicate frame ID {}", state.frame_id));
            }
            if frame_timestamps
                .insert(state.timestamp_ns, state.frame_id)
                .is_some()
            {
                return Err(format!("duplicate frame timestamp {}", state.timestamp_ns));
            }
            state_ids.insert(state.frame_id);
        }

        // A non-empty AOM is a self-describing order in schema 3.  Older
        // records may omit it, so retain the legacy permissive path when the
        // field is absent.  When present, every block must be a valid 6- or
        // 15-DoF block and must resolve to the corresponding frame table.
        if !self.aom_order.is_empty() {
            let mut aom_ids = BTreeSet::new();
            let mut expected_offset = 0usize;
            for block in &self.aom_order {
                let expected_dof = match block.kind.as_str() {
                    "pose" => POSE_BLOCK_DOF,
                    "state" => STATE_BLOCK_DOF,
                    other => return Err(format!("unknown AOM block kind {other:?}")),
                };
                if block.dof != expected_dof {
                    return Err(format!(
                        "AOM block {} has kind {} but dof {} (expected {})",
                        block.frame_id, block.kind, block.dof, expected_dof
                    ));
                }
                if block.offset != expected_offset {
                    return Err(format!(
                        "AOM block {} has non-contiguous offset {} (expected {})",
                        block.frame_id, block.offset, expected_offset
                    ));
                }
                if !aom_ids.insert(block.frame_id) {
                    return Err(format!("duplicate AOM frame ID {}", block.frame_id));
                }
                let table_present = match block.kind.as_str() {
                    "pose" => pose_ids.contains(&block.frame_id),
                    "state" => state_ids.contains(&block.frame_id),
                    _ => false,
                };
                if !table_present {
                    return Err(format!(
                        "AOM {} block {} is missing from its frame table",
                        block.kind, block.frame_id
                    ));
                }
                expected_offset = expected_offset
                    .checked_add(expected_dof)
                    .ok_or_else(|| "AOM offsets overflow".to_owned())?;
            }
            if expected_offset != self.aom_sqrt_jacobian.cols {
                return Err(format!(
                    "AOM block width {} does not match Jacobian columns {}",
                    expected_offset, self.aom_sqrt_jacobian.cols
                ));
            }
        }

        // Raw images are allowed for a keyframe selected for removal: that
        // frame is intentionally absent from the post-shift pose/state maps.
        // In that case kfs_all/kfs_to_marg (or the legacy kf_to_marg alias)
        // still supplies the identity needed to validate its timestamp.
        let mut image_frame_ids = frame_ids.keys().copied().collect::<BTreeSet<_>>();
        image_frame_ids.extend(self.keyframes.iter().copied());
        image_frame_ids.extend(self.kfs_all.iter().copied());
        image_frame_ids.extend(self.kfs_to_marg.iter().copied());
        image_frame_ids.extend(self.kf_to_marg.iter().flat_map(|(from, to)| [*from, *to]));
        image_frame_ids.extend(self.marginalization.poses_to_marg.iter().copied());
        let mut image_timestamps = BTreeMap::<u64, i64>::new();
        let mut image_timestamp_owners = BTreeMap::<i64, u64>::new();
        let mut image_keys = BTreeSet::<(u64, i64, u16)>::new();
        for image in &self.of_images {
            if !image.is_valid() {
                return Err(format!("image {} is malformed", image.frame_id));
            }
            if !image_frame_ids.contains(&image.frame_id) {
                return Err(format!(
                    "image {} has no matching frame/keyframe identity",
                    image.frame_id
                ));
            }
            if let Some((_, table_timestamp)) = frame_ids.get(&image.frame_id) {
                if *table_timestamp != image.timestamp_ns {
                    return Err(format!(
                        "image {} timestamp {} does not match frame timestamp {}",
                        image.frame_id, image.timestamp_ns, table_timestamp
                    ));
                }
            }
            if let Some(previous) = image_frame_ids
                .contains(&image.frame_id)
                .then(|| image_timestamps.insert(image.frame_id, image.timestamp_ns))
                .flatten()
            {
                if previous != image.timestamp_ns {
                    return Err(format!(
                        "frame {} has inconsistent image timestamps",
                        image.frame_id
                    ));
                }
            }
            if let Some(previous_frame) =
                image_timestamp_owners.insert(image.timestamp_ns, image.frame_id)
            {
                if previous_frame != image.frame_id {
                    return Err(format!(
                        "timestamp {} is paired with multiple image frames",
                        image.timestamp_ns
                    ));
                }
            }
            if !image_keys.insert((image.frame_id, image.timestamp_ns, image.camera_id)) {
                return Err(format!(
                    "duplicate image key ({}, {}, {})",
                    image.frame_id, image.timestamp_ns, image.camera_id
                ));
            }
        }
        if self.of_images.windows(2).any(|images| {
            (
                images[0].frame_id,
                images[0].timestamp_ns,
                images[0].camera_id,
            ) >= (
                images[1].frame_id,
                images[1].timestamp_ns,
                images[1].camera_id,
            )
        }) {
            return Err("images are not in canonical frame/timestamp/camera order".to_owned());
        }
        if self
            .of_observations
            .iter()
            .any(|observation| !observation.x.is_finite() || !observation.y.is_finite())
        {
            return Err("optical-flow observations contain a non-finite value".to_owned());
        }

        if self.row_counts.iter().any(|count| *count != 0) {
            let row_count_sum = self
                .row_counts
                .iter()
                .try_fold(0usize, |sum, count| sum.checked_add(*count))
                .ok_or_else(|| "row-count metadata overflows".to_owned())?;
            if row_count_sum != self.aom_sqrt_jacobian.rows {
                return Err(format!(
                    "row-count metadata sums to {}, Jacobian has {} rows",
                    row_count_sum, self.aom_sqrt_jacobian.rows
                ));
            }
        }

        if let Some(prior) = &self.prior {
            if prior.frame_ids.len() != prior.block_kinds.len() && !prior.block_kinds.is_empty() {
                return Err("prior frame_ids and block_kinds lengths differ".to_owned());
            }
            let mut prior_ids = BTreeSet::new();
            if prior
                .frame_ids
                .iter()
                .any(|frame_id| !prior_ids.insert(*frame_id))
            {
                return Err("prior contains duplicate frame IDs".to_owned());
            }
            let expected_prior_cols = if prior.block_kinds.is_empty() {
                prior
                    .frame_ids
                    .len()
                    .checked_mul(STATE_BLOCK_DOF)
                    .ok_or_else(|| "prior block dimensions overflow".to_owned())?
            } else {
                prior.block_kinds.iter().try_fold(0usize, |sum, kind| {
                    let dof = match kind.as_str() {
                        "pose" => POSE_BLOCK_DOF,
                        "state_pose" => POSE_BLOCK_DOF,
                        "state" => STATE_BLOCK_DOF,
                        other => return Err(format!("unknown prior block kind {other:?}")),
                    };
                    sum.checked_add(dof)
                        .ok_or_else(|| "prior block dimensions overflow".to_owned())
                })?
            };
            let prior_size = prior
                .jacobian
                .rows
                .checked_mul(prior.jacobian.cols)
                .ok_or_else(|| "prior Jacobian dimensions overflow".to_owned())?;
            if prior.jacobian.data.len() != prior_size || prior.jacobian.cols != expected_prior_cols
            {
                return Err(
                    "prior Jacobian dimensions do not match prior block metadata".to_owned(),
                );
            }
            if prior.rhs.len() != prior.jacobian.rows {
                return Err("prior RHS length does not match prior Jacobian rows".to_owned());
            }
            if prior.fej_point.len() != prior.jacobian.cols {
                return Err("prior FEJ point is missing or has the wrong dimension".to_owned());
            }
            if prior
                .jacobian
                .data
                .iter()
                .chain(prior.rhs.iter())
                .chain(prior.fej_point.iter())
                .any(|value| !value.is_finite())
            {
                return Err("prior contains a non-finite value".to_owned());
            }
        }

        if !self.provenance_version.starts_with("basalt-") {
            return Err("provenance_version must start with basalt-".to_owned());
        }
        if self.schema_version == MARGDATA_SCHEMA_VERSION {
            if !self.fej_complete {
                return Err("schema 4 requires fej_complete=true".to_owned());
            }
            self.validate_fej_tables()?;
            self.validate_schema4_fields()?;
        }
        Ok(())
    }

    /// Enforce fields that are optional for schema 1--3 compatibility but are
    /// required to replay a schema-4 marginalization record.  Keeping this
    /// check at the schema boundary is important: a native cereal packet may
    /// carry an `abs_H`/`abs_b` snapshot and raw FEJ anchors, but it does not
    /// contain Rust's square-root rows, prior order, or `PriorData.fej_point`.
    /// Such a packet must remain an explicitly incomplete adapter result
    /// rather than becoming a seemingly valid Rust record through defaults.
    fn validate_schema4_fields(&self) -> Result<(), String> {
        if self.aom_abs_h.is_none() != self.aom_abs_b.is_none() {
            return Err("schema 4 requires paired absolute H and b fields".to_owned());
        }
        if self.aom_abs_h.is_none() || self.aom_abs_b.is_none() {
            return Err("schema 4 requires absolute H and b fields".to_owned());
        }

        // A non-empty square-root system (or any retained frame table) needs
        // the explicit mixed pose/state order.  A zero-row/zero-column
        // no-op record may omit it; legacy records may omit it regardless.
        if (self.aom_sqrt_jacobian.cols != 0
            || !self.frame_poses.is_empty()
            || !self.frame_states.is_empty())
            && self.aom_order.is_empty()
        {
            return Err("schema 4 requires explicit AOM order metadata".to_owned());
        }

        let row_count_sum = self
            .row_counts
            .iter()
            .try_fold(0usize, |sum, count| sum.checked_add(*count))
            .ok_or_else(|| "schema-4 row-count metadata overflows".to_owned())?;
        if row_count_sum != self.aom_sqrt_jacobian.rows {
            return Err(format!(
                "schema-4 row-count metadata sums to {}, Jacobian has {} rows",
                row_count_sum, self.aom_sqrt_jacobian.rows
            ));
        }

        for (name, values) in [
            ("keyframes", self.keyframes.as_slice()),
            ("kfs_all", self.kfs_all.as_slice()),
            ("kfs_to_marg", self.kfs_to_marg.as_slice()),
        ] {
            let mut ids = BTreeSet::new();
            if values.iter().any(|frame_id| !ids.insert(*frame_id)) {
                return Err(format!("schema 4 {name} contains duplicate frame IDs"));
            }
        }

        if let Some(prior) = &self.prior {
            if prior.block_kinds.is_empty() || prior.block_kinds.len() != prior.frame_ids.len() {
                return Err(
                    "schema 4 prior requires one explicit block kind per frame ID".to_owned(),
                );
            }

            let table_kind = |frame_id: u64| {
                if self
                    .frame_poses
                    .iter()
                    .any(|pose| pose.frame_id == frame_id)
                {
                    Some("pose")
                } else if self
                    .frame_states
                    .iter()
                    .any(|state| state.frame_id == frame_id)
                {
                    Some("state")
                } else {
                    None
                }
            };
            for (frame_id, kind) in prior.frame_ids.iter().zip(&prior.block_kinds) {
                let Some(actual_kind) = table_kind(*frame_id) else {
                    return Err(format!(
                        "schema 4 prior frame {} is missing from pose/state tables",
                        frame_id
                    ));
                };
                let kind_matches_table = match kind.as_str() {
                    "pose" => actual_kind == "pose",
                    "state_pose" | "state" => actual_kind == "state",
                    _ => false,
                };
                if !kind_matches_table {
                    return Err(format!(
                        "schema 4 prior frame {} kind {} does not match table kind {}",
                        frame_id, kind, actual_kind
                    ));
                }
            }
        }

        Ok(())
    }

    fn validate_fej_tables(&self) -> Result<(), String> {
        if self.frame_poses_fej.len() != self.frame_poses.len() {
            return Err(format!(
                "schema 4 has {} pose FEJ entries for {} frame poses",
                self.frame_poses_fej.len(),
                self.frame_poses.len()
            ));
        }
        if self.frame_states_fej.len() != self.frame_states.len() {
            return Err(format!(
                "schema 4 has {} state FEJ entries for {} frame states",
                self.frame_states_fej.len(),
                self.frame_states.len()
            ));
        }

        for pose in &self.frame_poses {
            let sidecar = self
                .frame_poses_fej
                .get(&pose.timestamp_ns)
                .ok_or_else(|| {
                    format!(
                        "schema 4 is missing pose FEJ timestamp {} (frame {})",
                        pose.timestamp_ns, pose.frame_id
                    )
                })?;
            if !finite_array(&sidecar.pose_linearized)
                || !finite_array(&sidecar.pose_current)
                || !finite_array(&sidecar.delta)
            {
                return Err(format!(
                    "pose FEJ timestamp {} contains a non-finite value",
                    pose.timestamp_ns
                ));
            }
            if !sidecar.linearized && sidecar.delta.iter().any(|value| *value != 0.0) {
                return Err(format!(
                    "pose FEJ timestamp {} has a non-zero delta while not linearized",
                    pose.timestamp_ns
                ));
            }
            let effective = if sidecar.linearized {
                &sidecar.pose_current
            } else {
                &sidecar.pose_linearized
            };
            if pose.pose != *effective {
                return Err(format!(
                    "pose {} does not match its effective FEJ branch",
                    pose.frame_id
                ));
            }
        }

        for state in &self.frame_states {
            let sidecar = self
                .frame_states_fej
                .get(&state.timestamp_ns)
                .ok_or_else(|| {
                    format!(
                        "schema 4 is missing state FEJ timestamp {} (frame {})",
                        state.timestamp_ns, state.frame_id
                    )
                })?;
            if !finite_nav(&sidecar.state_linearized)
                || !finite_nav(&sidecar.state_current)
                || !finite_array(&sidecar.delta)
            {
                return Err(format!(
                    "state FEJ timestamp {} contains a non-finite value",
                    state.timestamp_ns
                ));
            }
            if sidecar.linearized != state.linearized {
                return Err(format!(
                    "state {} FEJ flag does not match the compatibility flag",
                    state.frame_id
                ));
            }
            if !sidecar.linearized && sidecar.delta.iter().any(|value| *value != 0.0) {
                return Err(format!(
                    "state FEJ timestamp {} has a non-zero delta while not linearized",
                    state.timestamp_ns
                ));
            }
            let effective = if sidecar.linearized {
                &sidecar.state_current
            } else {
                &sidecar.state_linearized
            };
            if state.pose != effective.pose
                || state.velocity != effective.velocity
                || state.gyro_bias != effective.gyro_bias
                || state.accel_bias != effective.accel_bias
            {
                return Err(format!(
                    "state {} does not match its effective FEJ branch",
                    state.frame_id
                ));
            }
        }

        Ok(())
    }

    pub fn validate(&self) -> bool {
        self.validate_contract().is_ok()
    }
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Streams the same compact JSON representation produced by [`Self::to_json`]
    /// without first allocating the complete document as a `String`.
    pub fn write_json<W: Write>(&self, writer: W) -> Result<(), serde_json::Error> {
        serde_json::to_writer(writer, self)
    }

    /// Streams the queue-facing packet without cloning its images/matrices.
    ///
    /// The diagnostic-only `prior` field is deliberately absent, matching
    /// [`Self::to_mapper_packet`]. Calling this on a state-only record is a
    /// no-op, just as there is no queue packet to write for that record.
    pub fn write_mapper_packet_json<W: Write>(&self, writer: W) -> Result<(), serde_json::Error> {
        if !self.is_mapper_packet() {
            return Ok(());
        }
        serde_json::to_writer(writer, &self.mapper_packet_ref())
    }

    /// Streams a queue packet with a pre-bound native companion identity.
    ///
    /// Validation happens before a byte is written, so a stale identity cannot
    /// create a partially labelled packet.
    pub fn write_mapper_packet_json_with_native_companion_identity<W: Write>(
        &self,
        writer: W,
        identity: &NativeCompanionIdentity,
        packet_ordinal: u64,
    ) -> Result<(), String> {
        self.validate_native_companion_identity(identity, packet_ordinal)?;
        let packet = MapperPacketWithNativeIdentityRef {
            packet: self.mapper_packet_ref(),
            native_companion_identity: identity,
            prior: self.prior.as_ref(),
        };
        serde_json::to_writer(writer, &packet).map_err(|error| error.to_string())
    }

    pub fn validate_native_companion_identity(
        &self,
        identity: &NativeCompanionIdentity,
        packet_ordinal: u64,
    ) -> Result<(), String> {
        if !self.is_mapper_packet() {
            return Err("native companion identity requires a mapper packet".into());
        }
        if identity.event_ordinal != packet_ordinal {
            return Err(format!(
                "native companion event ordinal {} does not match Rust packet ordinal {}",
                identity.event_ordinal, packet_ordinal
            ));
        }
        if identity.run_uuid.is_empty()
            || identity.run_uuid.len() > 128
            || !identity
                .run_uuid
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        {
            return Err("native companion run UUID is empty or malformed".into());
        }
        for (name, digest) in [
            ("packet_sha256", identity.packet_sha256.as_str()),
            ("frame_map_sha256", identity.frame_map_sha256.as_str()),
        ] {
            if digest.len() != 64
                || !digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(format!("native companion {name} is not lowercase SHA-256"));
            }
        }
        let expected_filename = format!("{}.cereal", identity.primary_kf_timestamp_ns);
        if identity.packet_filename != expected_filename {
            return Err(format!(
                "native companion packet filename {:?} does not match {:?}",
                identity.packet_filename, expected_filename
            ));
        }

        let selected_frames = self.kfs_to_marg.iter().copied().collect::<BTreeSet<_>>();
        let selected_timestamps = self
            .of_images
            .iter()
            .filter(|image| selected_frames.contains(&image.frame_id))
            .map(|image| image.timestamp_ns)
            .collect::<BTreeSet<_>>();
        if selected_timestamps.len() != 1
            || !selected_timestamps.contains(&identity.primary_kf_timestamp_ns)
        {
            return Err(format!(
                "native companion primary keyframe timestamp {} does not match Rust selected-keyframe images {:?}",
                identity.primary_kf_timestamp_ns, selected_timestamps
            ));
        }
        let newest_state_timestamp = self
            .frame_states
            .iter()
            .map(|state| state.timestamp_ns)
            .max()
            .ok_or_else(|| {
                "native companion identity requires a Rust state timestamp".to_owned()
            })?;
        if identity.event_state_timestamp_ns != newest_state_timestamp {
            return Err(format!(
                "native companion event state timestamp {} does not match newest Rust state {}",
                identity.event_state_timestamp_ns, newest_state_timestamp
            ));
        }
        Ok(())
    }

    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// Deserialize and validate one complete Rust MargData record.  The
    /// ordinary [`Self::from_json`] function remains parse-only for callers
    /// that need to inspect a legacy or intentionally incomplete adapter
    /// payload; replay/load boundaries should use this checked form.
    pub fn from_json_checked(s: &str) -> Result<Self, String> {
        let value = Self::from_json(s).map_err(|error| error.to_string())?;
        value.validate_contract()?;
        Ok(value)
    }
    pub fn stable_hash(&self) -> u64 {
        self.to_json()
            .expect("serializable")
            .bytes()
            .fold(1469598103934665603u64, |h, b| {
                (h ^ (b as u64)).wrapping_mul(1099511628211)
            })
    }

    /// Materialize the opt-in boundary sidecar without changing the queue
    /// packet or any solver-owned value.  The caller supplies the newly
    /// reduced prior immediately before the native f32 RHS shift; `self.prior`
    /// is the same post-reduction prior after that shift, already attached to
    /// this pre-shift MargData record.
    pub fn diagnostic_sidecar(
        &self,
        event_ordinal: usize,
        event_frame_id: u64,
        event_timestamp_ns: i64,
        pre_prior: Option<PriorData>,
    ) -> MargDataDiagnosticSidecar {
        self.diagnostic_sidecar_with_effective(
            event_ordinal,
            event_frame_id,
            event_timestamp_ns,
            pre_prior,
            &BTreeMap::new(),
            &BTreeMap::new(),
        )
    }

    /// Variant used by the estimator when the native `getPose()` value has
    /// been materialized by the same UpstreamF32 helper as the visual factor
    /// path.  Keeping the effective tables as arguments avoids putting any
    /// diagnostic-only state on queue-facing `MargData` or changing its
    /// stable hash/serialization.
    pub fn diagnostic_sidecar_with_effective(
        &self,
        event_ordinal: usize,
        event_frame_id: u64,
        event_timestamp_ns: i64,
        pre_prior: Option<PriorData>,
        effective_pose_fej: &BTreeMap<i64, [f64; 7]>,
        effective_state_fej: &BTreeMap<i64, [f64; 7]>,
    ) -> MargDataDiagnosticSidecar {
        let identity_for = |frame_id: u64| {
            self.frame_poses
                .iter()
                .find(|pose| pose.frame_id == frame_id)
                .map(|pose| FrameIdentity {
                    frame_id,
                    timestamp_ns: pose.timestamp_ns,
                })
                .or_else(|| {
                    self.frame_states
                        .iter()
                        .find(|state| state.frame_id == frame_id)
                        .map(|state| FrameIdentity {
                            frame_id,
                            timestamp_ns: state.timestamp_ns,
                        })
                })
        };
        let kfs_all_ids = if self.kfs_all.is_empty() {
            self.keyframes.clone()
        } else {
            self.kfs_all.clone()
        };
        let selected_keyframes = self
            .kfs_to_marg
            .iter()
            .filter_map(|frame_id| identity_for(*frame_id))
            .collect::<Vec<_>>();
        let kfs_all = kfs_all_ids
            .iter()
            .filter_map(|frame_id| identity_for(*frame_id))
            .collect::<Vec<_>>();
        let boundary_state = self
            .aom_order
            .iter()
            .rev()
            .find(|block| block.kind == "state")
            .and_then(|block| identity_for(block.frame_id));
        let pose_fej = self
            .frame_poses
            .iter()
            .filter_map(|pose| {
                self.frame_poses_fej.get(&pose.timestamp_ns).map(|raw| {
                    let effective_get_pose = effective_pose_fej
                        .get(&pose.timestamp_ns)
                        .copied()
                        .unwrap_or_else(|| {
                            if raw.linearized {
                                raw.pose_current
                            } else {
                                raw.pose_linearized
                            }
                        });
                    PoseFejDiagnostic {
                        frame_id: pose.frame_id,
                        timestamp_ns: pose.timestamp_ns,
                        raw: raw.clone(),
                        raw_linearized: raw.pose_linearized,
                        raw_current: raw.pose_current,
                        raw_delta: raw.delta,
                        effective_get_pose,
                        effective_get_pose_f32_bits: f32_bits_vec(effective_get_pose),
                        f32_bits: fej_pose_bits(raw),
                    }
                })
            })
            .collect::<Vec<_>>();
        let state_fej = self
            .frame_states
            .iter()
            .filter_map(|state| {
                self.frame_states_fej.get(&state.timestamp_ns).map(|raw| {
                    let effective_get_pose = effective_state_fej
                        .get(&state.timestamp_ns)
                        .copied()
                        .unwrap_or_else(|| {
                            if raw.linearized {
                                raw.state_current.pose
                            } else {
                                raw.state_linearized.pose
                            }
                        });
                    StateFejDiagnostic {
                        frame_id: state.frame_id,
                        timestamp_ns: state.timestamp_ns,
                        raw: raw.clone(),
                        raw_state_linearized: Some(raw.state_linearized.clone()),
                        raw_state_current: Some(raw.state_current.clone()),
                        raw_delta: raw.delta,
                        effective_get_pose,
                        effective_get_pose_f32_bits: f32_bits_vec(effective_get_pose),
                        f32_bits: fej_state_bits(raw),
                    }
                })
            })
            .collect::<Vec<_>>();
        let post_prior = self.prior.clone();
        MargDataDiagnosticSidecar {
            schema: "visloc.basalt.margdata_boundary.v2".into(),
            event_ordinal,
            event_frame_id,
            event_timestamp_ns,
            boundary_state,
            selected_keyframes,
            kfs_all,
            aom_order: self.aom_order.clone(),
            pre_prior_f32_bits: pre_prior.as_ref().map(prior_bits),
            post_prior_f32_bits: post_prior.as_ref().map(prior_bits),
            pre_prior,
            post_prior,
            pose_fej,
            state_fej,
            scalar_mode: "UpstreamF32".into(),
            wire_order: "[tx,ty,tz,qw,qx,qy,qz]".into(),
            normalization: "raw branches are direct stored Sophus values; effective_get_pose applies native f32 delta semantics".into(),
            marg_data_stable_hash: self.stable_hash(),
        }
    }
}

impl MargDataDiagnosticSidecar {
    /// Streams one complete sidecar object.  The estimator appends a newline
    /// when writing JSONL so multiple marginalization events can share one
    /// diagnostic path without changing normal MargData serialization.
    pub fn write_json<W: Write>(&self, writer: W) -> Result<(), serde_json::Error> {
        serde_json::to_writer(writer, self)
    }
}

fn f32_bits(value: f64) -> String {
    format!("{:08x}", (value as f32).to_bits())
}

fn f32_bits_vec(values: impl IntoIterator<Item = f64>) -> Vec<String> {
    values.into_iter().map(f32_bits).collect()
}

fn matrix_bits(matrix: &MatrixData) -> MatrixBits {
    MatrixBits {
        rows: matrix.rows,
        cols: matrix.cols,
        layout: "column_major".into(),
        bits: f32_bits_vec(matrix.data.iter().copied()),
    }
}

fn prior_bits(prior: &PriorData) -> PriorDiagnosticBits {
    PriorDiagnosticBits {
        jacobian: matrix_bits(&prior.jacobian),
        rhs: f32_bits_vec(prior.rhs.iter().copied()),
        fej_point: f32_bits_vec(prior.fej_point.iter().copied()),
    }
}

fn fej_pose_bits(value: &PoseStateWithLinData) -> FejPoseBits {
    FejPoseBits {
        pose_linearized: f32_bits_vec(value.pose_linearized),
        pose_current: f32_bits_vec(value.pose_current),
        delta: f32_bits_vec(value.delta),
    }
}

fn fej_nav_bits(value: &NavStateData) -> FejNavBits {
    FejNavBits {
        pose: f32_bits_vec(value.pose),
        velocity: f32_bits_vec(value.velocity),
        gyro_bias: f32_bits_vec(value.gyro_bias),
        accel_bias: f32_bits_vec(value.accel_bias),
    }
}

fn fej_state_bits(value: &PoseVelBiasStateWithLinData) -> FejStateBits {
    FejStateBits {
        state_linearized: fej_nav_bits(&value.state_linearized),
        state_current: fej_nav_bits(&value.state_current),
        delta: f32_bits_vec(value.delta),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowEntry {
    pub frame_id: u64,
    pub is_keyframe: bool,
    pub is_latest: bool,
    pub connected_ratio: f64,
    pub lost_landmarks: u32,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowPolicy {
    pub max_states: usize,
    pub max_kfs: usize,
    pub min_feature_ratio: f64,
}
impl Default for WindowPolicy {
    fn default() -> Self {
        Self {
            max_states: 3,
            max_kfs: 7,
            min_feature_ratio: 0.1,
        }
    }
}
pub fn select_marginalization_targets(entries: &[WindowEntry], policy: WindowPolicy) -> Vec<u64> {
    let mut states = entries.to_vec();
    states.sort_by_key(|e| e.frame_id);
    if policy.max_states == 0 || states.len() < policy.max_states {
        return Vec::new();
    }

    // Pinned upstream `sqrt_keypoint_vio.cpp:559-608` enters at
    // `frame_states.size() >= max_states`, chooses index
    // `size - max_states + 1` as the retained boundary, and removes every
    // older full state. Upstream separately converts an older keyframe to a
    // pose-only variable; the current Rust solver does not yet have that
    // variable type. Connectivity and lost landmarks do not choose
    // navigation-state targets.
    let states_to_remove = states.len() - policy.max_states + 1;
    states
        .into_iter()
        .take(states_to_remove)
        .filter(|entry| !entry.is_latest)
        .map(|entry| entry.frame_id)
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    fn md() -> MargData {
        MargData {
            schema_version: MARGDATA_SCHEMA_VERSION_V3,
            aom_sqrt_jacobian: MatrixData::new(2, 2, vec![1., 2., 3., 4.]).unwrap(),
            aom_sqrt_rhs: vec![5., 6.],
            aom_abs_h: Some(MatrixData::new(2, 2, vec![7., 0., 0., 8.]).unwrap()),
            aom_abs_b: Some(vec![1., 2.]),
            frame_poses: Vec::new(),
            frame_states: vec![FrameStateData {
                frame_id: 1,
                timestamp_ns: 10,
                pose: [0.; 7],
                velocity: [0.; 3],
                gyro_bias: [0.; 3],
                accel_bias: [0.; 3],
                linearized: false,
                is_keyframe: true,
                is_latest: false,
            }],
            keyframes: vec![1],
            kf_to_marg: vec![(1, 2)],
            kfs_all: Vec::new(),
            kfs_to_marg: Vec::new(),
            aom_order: Vec::new(),
            marginalization: MarginalizationTargets::default(),
            prior: None,
            row_counts: [0; 4],
            of_observations: vec![OfObservationData {
                frame_id: 1,
                track_id: 4,
                camera_id: 0,
                x: 1.,
                y: 2.,
            }],
            of_images: Vec::new(),
            frame_poses_fej: BTreeMap::new(),
            frame_states_fej: BTreeMap::new(),
            fej_complete: false,
            measurement_poses: Vec::new(),
            used_imu: true,
            provenance_version: "basalt-0f3b2b52".into(),
        }
    }
    #[test]
    fn roundtrip_hash() {
        let x = md();
        assert!(x.validate());
        let y = MargData::from_json(&x.to_json().unwrap()).unwrap();
        assert_eq!(x, y);
        assert_eq!(x.stable_hash(), y.stable_hash());
    }

    #[test]
    fn streaming_writer_matches_to_json_bytes() {
        let x = md();
        let expected = x.to_json().unwrap();
        let mut streamed = Vec::new();
        x.write_json(&mut streamed).unwrap();
        assert_eq!(streamed, expected.as_bytes());
    }

    #[test]
    fn mapper_packet_streaming_writer_matches_owned_packet_bytes() {
        let mut x = schema4_record(true);
        x.kfs_to_marg.push(1);

        let expected = x.to_mapper_packet().unwrap().to_json().unwrap();
        let mut streamed = Vec::new();
        x.write_mapper_packet_json(&mut streamed).unwrap();
        assert_eq!(streamed, expected.as_bytes());
    }

    #[test]
    fn native_companion_identity_is_bound_before_streaming() {
        let mut x = md();
        x.kfs_to_marg = vec![1];
        x.of_images = vec![
            OfImageData::new(1, 10, 0, 1, 1, vec![7]).unwrap(),
            OfImageData::new(1, 10, 1, 1, 1, vec![8]).unwrap(),
        ];
        let identity = NativeCompanionIdentity {
            run_uuid: "native-run-1".into(),
            event_ordinal: 0,
            event_state_timestamp_ns: 10,
            primary_kf_timestamp_ns: 10,
            packet_filename: "10.cereal".into(),
            packet_sha256: "1".repeat(64),
            frame_map_sha256: "a".repeat(64),
        };
        let mut bytes = Vec::new();
        x.write_mapper_packet_json_with_native_companion_identity(&mut bytes, &identity, 0)
            .unwrap();
        let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            wire["native_companion_identity"]["run_uuid"],
            "native-run-1"
        );
        assert!(wire.get("prior").is_none());

        x.prior = Some(PriorData {
            frame_ids: vec![1],
            block_kinds: vec!["state".into()],
            jacobian: MatrixData::new(1, 1, vec![1.]).unwrap(),
            rhs: vec![2.],
            fej_point: vec![3.],
        });
        let mut diagnostic_bytes = Vec::new();
        x.write_mapper_packet_json_with_native_companion_identity(
            &mut diagnostic_bytes,
            &identity,
            0,
        )
        .unwrap();
        let diagnostic_wire: serde_json::Value = serde_json::from_slice(&diagnostic_bytes).unwrap();
        assert!(diagnostic_wire.get("prior").is_some());
        let mut ordinary_bytes = Vec::new();
        x.write_mapper_packet_json(&mut ordinary_bytes).unwrap();
        let ordinary_wire: serde_json::Value = serde_json::from_slice(&ordinary_bytes).unwrap();
        assert!(ordinary_wire.get("prior").is_none());

        let mut stale = identity.clone();
        stale.event_state_timestamp_ns += 1;
        let mut rejected = Vec::new();
        assert!(x
            .write_mapper_packet_json_with_native_companion_identity(&mut rejected, &stale, 0)
            .is_err());
        assert!(rejected.is_empty(), "validation must precede all output");
    }

    #[test]
    fn native_companion_identity_rejects_stale_ordinal_and_digest() {
        let mut x = md();
        x.kfs_to_marg = vec![1];
        x.of_images = vec![OfImageData::new(1, 10, 0, 1, 1, vec![7]).unwrap()];
        let mut identity = NativeCompanionIdentity {
            run_uuid: "native-run-1".into(),
            event_ordinal: 1,
            event_state_timestamp_ns: 10,
            primary_kf_timestamp_ns: 10,
            packet_filename: "10.cereal".into(),
            packet_sha256: "1".repeat(64),
            frame_map_sha256: "a".repeat(64),
        };
        assert!(x
            .write_mapper_packet_json_with_native_companion_identity(Vec::new(), &identity, 0)
            .is_err());
        identity.event_ordinal = 0;
        identity.packet_sha256 = "A".repeat(64);
        assert!(x
            .write_mapper_packet_json_with_native_companion_identity(Vec::new(), &identity, 0)
            .is_err());
    }

    #[test]
    fn mapper_packet_requires_keyframe_removal() {
        let mut x = md();
        x.prior = Some(PriorData {
            frame_ids: vec![1],
            block_kinds: vec!["state".into()],
            jacobian: MatrixData::new(1, 1, vec![1.]).unwrap(),
            rhs: vec![2.],
            fej_point: vec![3.],
        });
        assert!(!x.is_mapper_packet());
        assert!(x.as_mapper_packet().is_none());
        x.kfs_to_marg.push(1);
        assert!(x.is_mapper_packet());
        assert!(x.as_mapper_packet().is_some());
        let packet = x.to_mapper_packet().expect("queue packet");
        assert!(packet.prior.is_none());
        assert_eq!(packet.kfs_to_marg, vec![1]);
        assert!(x.prior.is_some(), "diagnostic record remains unchanged");
        assert!(packet.validate());
        let packet_wire: serde_json::Value =
            serde_json::from_str(&packet.to_json().unwrap()).unwrap();
        assert!(packet_wire.get("prior").is_none());
        let diagnostic_wire: serde_json::Value =
            serde_json::from_str(&x.to_json().unwrap()).unwrap();
        assert!(diagnostic_wire.get("prior").is_some());
    }

    #[test]
    fn diagnostic_record_remains_available_when_packet_view_is_absent() {
        let x = md();
        assert_eq!(x.as_mapper_packet(), None);
        assert_eq!(x.of_observations.len(), 1);
        assert_eq!(x.aom_abs_b.as_ref().map(Vec::len), Some(2));
    }

    #[test]
    fn absolute_rhs_dimension_is_validated() {
        let mut x = md();
        x.aom_abs_b = Some(vec![1.0]);
        assert!(!x.validate());
    }

    #[test]
    fn raw_u16_image_payload_is_lossless_compact_and_deterministic() {
        let mut x = md();
        x.of_images = vec![
            OfImageData::new(1, 10, 0, 2, 2, vec![0, 1, 0x1234, 0xffff]).unwrap(),
            OfImageData::new(1, 10, 1, 2, 2, vec![0x8000, 0x00ff, 7, 9]).unwrap(),
        ];
        assert!(x.validate());
        let json = x.to_json().unwrap();
        // v3 uses 4/3 base64 bytes rather than a multi-megabyte decimal JSON
        // array, while preserving the public Vec<u16> API.
        let wire: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(wire["of_images"][0]["data"].is_string());
        let y = MargData::from_json(&json).unwrap();
        assert_eq!(x, y);
        assert_eq!(x.stable_hash(), y.stable_hash());
        assert_eq!(x.of_images[0].sample_hash(), y.of_images[0].sample_hash());
    }

    #[test]
    fn old_margdata_without_images_remains_readable() {
        let mut value: serde_json::Value = serde_json::from_str(&md().to_json().unwrap()).unwrap();
        value["schema_version"] = serde_json::Value::from(MARGDATA_PREVIOUS_SCHEMA_VERSION);
        value.as_object_mut().unwrap().remove("of_images");
        let old = MargData::from_json(&value.to_string()).unwrap();
        assert!(old.validate());
        assert!(old.of_images.is_empty());
    }

    #[test]
    fn malformed_raw_image_payload_is_rejected() {
        let mut x = md();
        x.of_images = vec![OfImageData {
            frame_id: 1,
            timestamp_ns: 10,
            camera_id: 0,
            width: 2,
            height: 2,
            data: vec![1, 2, 3],
        }];
        assert!(!x.validate());
        x.of_images = vec![
            OfImageData::new(1, 10, 1, 1, 1, vec![1]).unwrap(),
            OfImageData::new(1, 10, 0, 1, 1, vec![2]).unwrap(),
        ];
        assert!(!x.validate(), "image order must be canonical");
    }
    #[test]
    fn policy_matches_upstream_state_boundary_independent_of_kf_or_track_loss() {
        let e = (0..5)
            .map(|i| WindowEntry {
                frame_id: i,
                is_keyframe: i < 2,
                is_latest: i == 4,
                connected_ratio: if i == 3 { 0.2 } else { 1. },
                lost_landmarks: if i == 2 { 1 } else { 0 },
            })
            .collect::<Vec<_>>();
        let out = select_marginalization_targets(&e, WindowPolicy::default());
        assert_eq!(out, vec![0, 1, 2]);
    }
    #[test]
    fn upstream_first_ten_frame_state_pose_schedule_is_reproducible() {
        let keyframes = [0_u64, 7];
        let mut states = Vec::<WindowEntry>::new();
        let mut poses = Vec::<u64>::new();
        let mut counts = Vec::new();

        for frame_id in 0_u64..=10 {
            states.push(WindowEntry {
                frame_id,
                is_keyframe: keyframes.contains(&frame_id),
                is_latest: true,
                connected_ratio: 0.0,
                lost_landmarks: 0,
            });
            for state in &mut states {
                state.is_latest = state.frame_id == frame_id;
            }
            // `opt_started` gates marginalization off for frames 0..3.
            if frame_id >= 4 {
                let dropped = select_marginalization_targets(&states, WindowPolicy::default());
                for state in states
                    .iter()
                    .filter(|state| dropped.contains(&state.frame_id))
                {
                    if state.is_keyframe {
                        poses.push(state.frame_id);
                    }
                }
                states.retain(|state| !dropped.contains(&state.frame_id));
            }
            counts.push((states.len(), poses.len()));
        }

        assert_eq!(
            counts,
            vec![
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (2, 1),
                (2, 1),
                (2, 1),
                (2, 1),
                (2, 1),
                (2, 2),
                (2, 2),
            ]
        );
    }
    #[test]
    fn malformed_schema() {
        let mut x = md();
        x.schema_version = 4;
        assert!(!x.validate());
    }

    #[test]
    fn mixed_aom_order_puts_pose_only_keyframes_first() {
        let mut x = md();
        x.keyframes = vec![1, 7];
        x.frame_states[0].frame_id = 2;
        x.frame_states.push(FrameStateData {
            frame_id: 3,
            timestamp_ns: 12,
            pose: [0.; 7],
            velocity: [0.; 3],
            gyro_bias: [0.; 3],
            accel_bias: [0.; 3],
            linearized: false,
            is_keyframe: false,
            is_latest: true,
        });
        assert_eq!(x.aom_block_order(), vec![(1, 6), (7, 6), (2, 15), (3, 15)]);
        assert_eq!(x.aom_dof(), 42);
    }

    fn nav_state(seed: f64) -> NavStateData {
        NavStateData {
            pose: [seed; 7],
            velocity: [seed + 1.0; 3],
            gyro_bias: [seed + 2.0; 3],
            accel_bias: [seed + 3.0; 3],
        }
    }

    fn schema4_record(linearized: bool) -> MargData {
        let mut base = md();
        base.schema_version = MARGDATA_SCHEMA_VERSION_V4;
        // Keep this fixture schema-4-complete: schema 3 deliberately uses a
        // tiny compatibility matrix, while schema 4 must expose the paired
        // absolute system, explicit AOM order, and exact row categories.
        base.aom_sqrt_jacobian =
            MatrixData::new(2, STATE_BLOCK_DOF, vec![0.; 2 * STATE_BLOCK_DOF]).unwrap();
        base.aom_abs_h = Some(
            MatrixData::new(
                STATE_BLOCK_DOF,
                STATE_BLOCK_DOF,
                vec![0.; STATE_BLOCK_DOF * STATE_BLOCK_DOF],
            )
            .unwrap(),
        );
        base.aom_abs_b = Some(vec![0.; STATE_BLOCK_DOF]);
        base.aom_order = vec![AomBlockData {
            frame_id: 1,
            offset: 0,
            dof: STATE_BLOCK_DOF,
            kind: "state".into(),
        }];
        base.row_counts = [2, 0, 0, 0];
        base.prior = Some(PriorData {
            frame_ids: vec![1],
            block_kinds: vec!["state".into()],
            jacobian: MatrixData::new(1, STATE_BLOCK_DOF, vec![1.; STATE_BLOCK_DOF]).unwrap(),
            rhs: vec![2.],
            fej_point: vec![3.; STATE_BLOCK_DOF],
        });

        let linearized_state = nav_state(1.0);
        let current_state = nav_state(10.0);
        let effective_state = if linearized {
            &current_state
        } else {
            &linearized_state
        };
        base.frame_states[0].pose = effective_state.pose;
        base.frame_states[0].velocity = effective_state.velocity;
        base.frame_states[0].gyro_bias = effective_state.gyro_bias;
        base.frame_states[0].accel_bias = effective_state.accel_bias;
        base.frame_states[0].linearized = linearized;

        let sidecar = PoseVelBiasStateWithLinData {
            state_linearized: linearized_state,
            state_current: current_state,
            delta: if linearized {
                [0.25; STATE_BLOCK_DOF]
            } else {
                [0.; STATE_BLOCK_DOF]
            },
            linearized,
        };
        base.frame_poses_fej = BTreeMap::new();
        base.frame_states_fej = [(10, sidecar)].into_iter().collect();
        base.fej_complete = true;
        base
    }

    #[test]
    fn schema4_roundtrip_preserves_fej_sidecars_and_prior_dimensions() {
        let x = schema4_record(true);
        assert!(x.validate());
        let json = x.to_json().unwrap();
        let wire: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(wire["schema_version"], MARGDATA_SCHEMA_VERSION_V4);
        assert!(wire.get("frame_states_fej").is_some());
        assert!(wire.get("fej_complete").is_some());

        let y = MargData::from_json(&json).unwrap();
        assert_eq!(x, y);
        assert_eq!(x.stable_hash(), y.stable_hash());
        let prior = y.prior.as_ref().unwrap();
        assert_eq!(prior.fej_point.len(), prior.jacobian.cols);
    }

    #[test]
    fn schema4_state_pose_prior_is_explicit_six_dof_and_roundtrips() {
        let mut x = schema4_record(false);
        let prior = x.prior.as_mut().unwrap();
        prior.block_kinds = vec!["state_pose".into()];
        prior.jacobian = MatrixData::new(1, POSE_BLOCK_DOF, vec![1.; POSE_BLOCK_DOF]).unwrap();
        prior.rhs = vec![2.];
        prior.fej_point = vec![3.; POSE_BLOCK_DOF];

        assert!(
            x.validate(),
            "state_pose resolves through the frame_states table"
        );
        let json = x.to_json().unwrap();
        let wire: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            wire["prior"]["block_kinds"],
            serde_json::json!(["state_pose"])
        );
        let roundtrip = MargData::from_json_checked(&json).unwrap();
        assert_eq!(
            roundtrip.prior.as_ref().unwrap().jacobian.cols,
            POSE_BLOCK_DOF
        );
        assert_eq!(roundtrip, x);
    }

    #[test]
    fn schema4_state_pose_rejects_pose_table_or_full_state_dimensions() {
        let mut x = schema4_record(false);
        let prior = x.prior.as_mut().unwrap();
        prior.block_kinds = vec!["state_pose".into()];
        prior.jacobian = MatrixData::new(1, STATE_BLOCK_DOF, vec![1.; STATE_BLOCK_DOF]).unwrap();
        prior.fej_point = vec![3.; STATE_BLOCK_DOF];
        assert!(
            !x.validate(),
            "state_pose must retain exactly six prior columns"
        );

        let mut x = schema4_record(false);
        x.prior.as_mut().unwrap().block_kinds[0] = "pose".into();
        assert!(
            !x.validate(),
            "state_pose cannot be relabeled as a pose-table block"
        );
    }

    #[test]
    fn diagnostic_boundary_sidecar_roundtrips_identity_prior_and_raw_f32_bits() {
        let mut x = schema4_record(true);
        x.kfs_all = vec![1];
        x.kfs_to_marg = vec![1];
        x.aom_order = vec![AomBlockData {
            frame_id: 1,
            offset: 0,
            dof: STATE_BLOCK_DOF,
            kind: "state".into(),
        }];
        let pre = x.prior.clone();
        let sidecar = x.diagnostic_sidecar(3, 9, 99, pre.clone());
        assert_eq!(sidecar.schema, "visloc.basalt.margdata_boundary.v2");
        assert_eq!(sidecar.event_ordinal, 3);
        assert_eq!(sidecar.event_frame_id, 9);
        assert_eq!(sidecar.event_timestamp_ns, 99);
        assert_eq!(
            sidecar.selected_keyframes,
            vec![FrameIdentity {
                frame_id: 1,
                timestamp_ns: 10
            }]
        );
        assert_eq!(sidecar.kfs_all, sidecar.selected_keyframes);
        assert_eq!(sidecar.pre_prior, pre);
        assert_eq!(sidecar.post_prior, x.prior);
        assert_eq!(sidecar.state_fej.len(), 1);
        assert_eq!(
            sidecar.state_fej[0].f32_bits.state_current.pose[0],
            "41200000"
        );
        assert_eq!(sidecar.state_fej[0].f32_bits.delta[0], "3e800000");
        assert_eq!(
            sidecar.state_fej[0]
                .raw_state_current
                .as_ref()
                .unwrap()
                .pose[0],
            10.0
        );
        assert_eq!(
            sidecar.state_fej[0].effective_get_pose,
            sidecar.state_fej[0]
                .raw_state_current
                .as_ref()
                .unwrap()
                .pose
        );
        assert_eq!(
            sidecar.state_fej[0]
                .raw_state_linearized
                .as_ref()
                .unwrap()
                .pose[0],
            1.0
        );
        assert_eq!(
            sidecar.state_fej[0]
                .raw_state_current
                .as_ref()
                .unwrap()
                .pose[0],
            10.0
        );
        assert_eq!(sidecar.state_fej[0].raw_delta, [0.25; STATE_BLOCK_DOF]);
        assert_eq!(
            sidecar.state_fej[0].effective_get_pose_f32_bits[0],
            "41200000"
        );
        assert_eq!(sidecar.marg_data_stable_hash, x.stable_hash());

        let encoded = serde_json::to_string(&sidecar).unwrap();
        let decoded: MargDataDiagnosticSidecar = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, sidecar);
    }

    #[test]
    fn diagnostic_boundary_sidecar_non_linearized_uses_raw_linearized_effective_pose() {
        let sidecar = schema4_record(false).diagnostic_sidecar(0, 2, 20, None);
        let state = &sidecar.state_fej[0];
        assert!(!state.raw.linearized);
        assert_eq!(state.raw_state_linearized.as_ref().unwrap().pose[0], 1.0);
        assert_eq!(state.raw_state_current.as_ref().unwrap().pose[0], 10.0);
        assert_eq!(state.raw_delta, [0.0; STATE_BLOCK_DOF]);
        assert_eq!(state.effective_get_pose[0], 1.0);
        assert_eq!(state.effective_get_pose_f32_bits[0], "3f800000");
    }

    #[test]
    fn diagnostic_boundary_sidecar_is_out_of_band_from_mapper_packet() {
        let mut x = schema4_record(true);
        x.kfs_to_marg.push(1);
        let packet_hash = x.to_mapper_packet().unwrap().stable_hash();
        let sidecar = x.diagnostic_sidecar(0, 2, 20, None);
        assert_eq!(x.to_mapper_packet().unwrap().stable_hash(), packet_hash);
        assert_eq!(sidecar.event_frame_id, 2);
        assert_eq!(sidecar.post_prior, x.prior);
    }

    #[test]
    fn schema3_fixture_accepts_without_inventing_fej_values() {
        let legacy_json = md().to_json().unwrap();
        let legacy = MargData::from_json(&legacy_json).unwrap();
        assert_eq!(legacy.schema_version, MARGDATA_SCHEMA_VERSION_V3);
        assert!(legacy.frame_poses_fej.is_empty());
        assert!(legacy.frame_states_fej.is_empty());
        assert!(!legacy.fej_complete);
        assert!(legacy.validate());

        let upgraded_wire: serde_json::Value =
            serde_json::from_str(&legacy.to_json().unwrap()).unwrap();
        assert!(upgraded_wire.get("frame_poses_fej").is_none());
        assert!(upgraded_wire.get("frame_states_fej").is_none());
        assert!(upgraded_wire.get("fej_complete").is_none());
    }

    #[test]
    fn schema3_prior_without_block_kinds_keeps_legacy_default_without_inventing_kind() {
        let mut value = md();
        value.prior = Some(PriorData {
            frame_ids: vec![1],
            block_kinds: vec!["state".into()],
            jacobian: MatrixData::new(1, STATE_BLOCK_DOF, vec![1.; STATE_BLOCK_DOF]).unwrap(),
            rhs: vec![2.],
            fej_point: vec![3.; STATE_BLOCK_DOF],
        });
        let mut wire: serde_json::Value = serde_json::from_str(&value.to_json().unwrap()).unwrap();
        wire["prior"].as_object_mut().unwrap().remove("block_kinds");
        let legacy = MargData::from_json(&wire.to_string()).unwrap();
        assert!(legacy.validate());
        assert!(legacy.prior.unwrap().block_kinds.is_empty());
    }

    #[test]
    fn schema4_requires_complete_fej_tables() {
        let mut x = {
            let mut base = md();
            base.schema_version = MARGDATA_SCHEMA_VERSION_V4;
            base
        };
        assert!(!x.validate());

        x = schema4_record(false);
        x.frame_states_fej.clear();
        assert!(!x.validate());
    }

    #[test]
    fn schema4_false_linearized_uses_linearized_value_not_stored_current() {
        let x = schema4_record(false);
        assert!(x.validate());
        let sidecar = x.frame_states_fej.get(&10).unwrap();
        assert_ne!(sidecar.state_linearized.pose, sidecar.state_current.pose);
        assert_eq!(x.frame_states[0].pose, sidecar.state_linearized.pose);
        assert!(sidecar.delta.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn schema4_rejects_bad_prior_fej_dimension() {
        let mut x = schema4_record(true);
        x.prior.as_mut().unwrap().fej_point.pop();
        assert!(!x.validate());
    }

    #[test]
    fn schema4_rejects_legacy_optional_system_fields() {
        let mut x = schema4_record(false);
        x.aom_abs_h = None;
        assert!(!x.validate(), "schema 4 must reject missing absolute H");

        let mut x = schema4_record(false);
        x.aom_abs_b = None;
        assert!(!x.validate(), "schema 4 must reject missing absolute b");

        let mut x = schema4_record(false);
        x.aom_order.clear();
        assert!(!x.validate(), "schema 4 must reject omitted AOM order");

        let mut x = schema4_record(false);
        x.row_counts[0] -= 1;
        assert!(
            !x.validate(),
            "schema 4 row categories must sum to sqrt rows"
        );
    }

    #[test]
    fn schema4_rejects_underspecified_or_mismatched_prior_blocks() {
        let mut x = schema4_record(false);
        x.prior.as_mut().unwrap().block_kinds.clear();
        assert!(
            !x.validate(),
            "prior block kinds may not be inferred in schema 4"
        );

        let mut x = schema4_record(false);
        x.prior.as_mut().unwrap().frame_ids[0] = 999;
        assert!(
            !x.validate(),
            "prior frame IDs must resolve to state/pose tables"
        );

        let mut x = schema4_record(false);
        x.prior.as_mut().unwrap().block_kinds[0] = "pose".into();
        assert!(!x.validate(), "prior block kind must match its frame table");
    }

    #[test]
    fn checked_json_keeps_schema4_unavailable_adapter_payload_fail_closed() {
        let mut wire: serde_json::Value =
            serde_json::from_str(&schema4_record(false).to_json().unwrap()).unwrap();
        wire["aom_sqrt_jacobian"] = serde_json::json!({
            "available": false,
            "reason": "native cereal packet does not contain Rust sqrt rows"
        });
        let error = MargData::from_json_checked(&wire.to_string()).unwrap_err();
        assert!(error.contains("rows") || error.contains("expected"));
    }

    #[test]
    fn checked_json_accepts_schema3_without_inventing_fej_or_strict_fields() {
        let json = md().to_json().unwrap();
        let checked = MargData::from_json_checked(&json).unwrap();
        assert_eq!(checked.schema_version, MARGDATA_SCHEMA_VERSION_V3);
        assert!(!checked.fej_complete);
        assert!(checked.frame_poses_fej.is_empty());
        assert!(checked.frame_states_fej.is_empty());
    }
}
