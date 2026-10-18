//! Multi-view stereo for visloc-rs: per-view depth and normal maps by
//! PatchMatch ([`patch_match`]), fused into a point cloud ([`fusion`]).
//!
//! The input is a COLMAP sparse model with pinhole cameras (undistorted
//! images) and its images ([`model`], [`workspace`]). Depth estimation is
//! ACMH with ACMP's planar prior ([`mod@planar_prior`]) and ACMM's geometric
//! consistency, on an image pyramid in ACMMP's order ([`pipeline`]; Xu and
//! Tao, CVPR 2019 and AAAI 2020; Xu et al., TPAMI 2022); fusion keeps the
//! points that enough source views agree on.

pub mod fusion;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod model;
pub mod patch_match;
pub mod pipeline;
pub mod planar_prior;
pub mod ply;
pub mod workspace;

pub use fusion::{fuse, FusedPoint, FusionOptions};
#[cfg(feature = "gpu")]
pub use gpu::GpuPatchMatch;
pub use model::{read_text_model, SparseModel};
pub use patch_match::{
    estimate, estimate_from, estimate_with_prior, photometric_score, DepthMap, DetailRestorer,
    PatchMatchOptions, Sampling, ViewSelection,
};
pub use pipeline::{
    geometric_pass, multi_scale_depth_maps, planar_prior_pass, Estimator, MultiScaleOptions,
    PriorEstimator, StartEstimator,
};
pub use planar_prior::{planar_prior, PlanarPrior, PlanarPriorOptions};
pub use ply::write_ply;
pub use workspace::{View, Workspace, WorkspaceOptions};
