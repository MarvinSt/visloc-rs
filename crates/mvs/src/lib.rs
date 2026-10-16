//! Multi-view stereo for visloc-rs: per-view depth and normal maps by
//! PatchMatch ([`patch_match`]), fused into a point cloud ([`fusion`]).
//!
//! The input is a COLMAP sparse model with pinhole cameras (undistorted
//! images) and its images ([`model`], [`workspace`]). Depth estimation is
//! ACMH with ACMM's geometric consistency (Xu and Tao, CVPR 2019); fusion
//! keeps the points that enough source views agree on.

pub mod fusion;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod model;
pub mod patch_match;
pub mod pipeline;
pub mod ply;
pub mod workspace;

pub use fusion::{fuse, FusedPoint, FusionOptions};
#[cfg(feature = "gpu")]
pub use gpu::GpuPatchMatch;
pub use model::{read_text_model, SparseModel};
pub use patch_match::{estimate, DepthMap, PatchMatchOptions, ViewSelection};
pub use pipeline::{multi_scale_depth_maps, MultiScaleOptions};
pub use ply::write_ply;
pub use workspace::{View, Workspace, WorkspaceOptions};
