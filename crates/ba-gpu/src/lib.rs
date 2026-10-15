//! wgpu bundle adjustment for visloc-rs SfM.
//!
//! `GpuBundleAdjuster` runs the same Levenberg-Marquardt loop as
//! visloc-slam's CPU `BundleAdjustment::optimize` for monocular pinhole
//! problems (same damping schedule, acceptance test, non-projectable gate
//! and convergence tests; costs evaluated on the CPU in f64), but builds and
//! solves each linear system on the GPU in f32: per-observation Jacobians
//! with the Huber IRLS weight, per-pose / per-landmark reductions, and an
//! implicit Schur complement solved by block-Jacobi PCG, then landmark
//! back-substitution. The f32 step is only a proposal; the f64 cost test
//! decides, so the result is validated like any LM step.
//!
//! Register it with `visloc_slam::set_ba_accelerator` to make the
//! incremental SfM use it for its global bundle adjustments (local windows
//! are opt-in, see `GpuBaSettings::local_windows`).
//!
//! `GpuReducedSolver` instead serves the COLMAP-port mapper: it runs the
//! rig solver's preconditioned conjugate gradients on the reduced camera
//! system of large bundle adjustments (`register_reduced_pcg`).

#[cfg(feature = "gpu")]
mod reduced;
#[cfg(feature = "gpu")]
mod solver;

#[cfg(feature = "gpu")]
pub use reduced::GpuReducedSolver;

#[cfg(feature = "gpu")]
pub use solver::{GpuBaError, GpuBaSettings, GpuBundleAdjuster};
#[cfg(feature = "gpu")]
pub use visloc_gsplat_render::{try_context, GpuContext, GpuError};

/// Create a GPU context and register a [`GpuReducedSolver`] as the
/// COLMAP-port rig solver's reduced-camera solve; returns the adapter name.
#[cfg(feature = "gpu")]
pub fn register_reduced_pcg() -> Result<String, GpuError> {
    let ctx = GpuContext::new()?;
    let name = format!("{} ({:?})", ctx.adapter_info.name, ctx.adapter_info.backend);
    visloc_slam::set_reduced_solve_accelerator(Box::new(GpuReducedSolver::new(ctx)));
    Ok(name)
}
