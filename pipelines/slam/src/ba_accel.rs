//! Optional accelerator for the incremental SfM's bundle adjustments.
//!
//! visloc-slam stays GPU-free: a backend crate (e.g. `visloc-ba-gpu`)
//! implements [`BaAccelerator`] and registers it once with
//! [`set_ba_accelerator`]. The SfM then offers every unweighted,
//! free-rotation global BA and every local BA to it; returning `None` falls
//! back to the CPU solver for that call.

use std::sync::OnceLock;

use crate::bundle::{BaConfig, BaError, BaResult, BundleAdjustment};

/// Which SfM solve is being offered to the accelerator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaScope {
    /// Whole-model refinement (periodic, growth and final global BA).
    Global,
    /// The per-registration window around a newly added image.
    Local,
}

/// A replacement solver for SfM bundle adjustment.
pub trait BaAccelerator: Send + Sync {
    /// Optimize `ba` in place with `config`'s LM settings, or return `None`
    /// when the problem is outside what the accelerator supports.
    fn optimize(
        &self,
        ba: &mut BundleAdjustment,
        config: &BaConfig,
        scope: BaScope,
    ) -> Option<Result<BaResult, BaError>>;
}

static ACCELERATOR: OnceLock<Box<dyn BaAccelerator>> = OnceLock::new();

/// Register the process-wide accelerator. Returns `false` if one was
/// already registered (the first registration wins).
pub fn set_ba_accelerator(accelerator: Box<dyn BaAccelerator>) -> bool {
    ACCELERATOR.set(accelerator).is_ok()
}

pub(crate) fn ba_accelerator() -> Option<&'static dyn BaAccelerator> {
    ACCELERATOR.get().map(|a| a.as_ref())
}

/// The reduced camera system of the COLMAP-port rig solver's iterative
/// branch (`colmap_incremental::rig_ba_solver`, systems of at least 1,000
/// free frames), offered to a [`ReducedSolveAccelerator`]. The CPU solver
/// runs block-preconditioned conjugate gradients on it; an accelerator runs
/// the same iteration with the same preconditioner and stopping rule.
pub struct ReducedCameraSystem<'a> {
    /// Damped 6x6 diagonal blocks, one per free node.
    pub diag: &'a [nalgebra::Matrix6<f64>],
    /// Edge `e` couples nodes `edges[e] = (a, b)`, `a < b`.
    pub edges: &'a [(u32, u32)],
    /// `offdiag[36 e..36 e + 36]`: the (row b, column a) block of edge `e`,
    /// column-major.
    pub offdiag: &'a [f32],
    pub rhs: &'a [nalgebra::Vector6<f64>],
    /// Dense inverses of the system restricted to runs of consecutive nodes,
    /// as `(first node, inverse)`; `None`: block-Jacobi.
    pub clusters: Option<&'a [(usize, nalgebra::DMatrix<f64>)]>,
    /// Additive coarse level: `(nodes per aggregate, leading nodes covered,
    /// inverse of the aggregated matrix)`.
    pub coarse: Option<(usize, usize, &'a nalgebra::DMatrix<f64>)>,
    /// Stop once `|r| <= tol |rhs|`.
    pub tol: f64,
    pub max_iterations: usize,
}

/// A replacement for the COLMAP-port rig solver's iterative reduced-camera
/// solve (e.g. on a GPU). Registered once with
/// [`set_reduced_solve_accelerator`]; returning `None` falls back to the CPU
/// solve for that call.
pub trait ReducedSolveAccelerator: Send + Sync {
    /// The camera step (one 6-vector per node) and the iteration count.
    fn solve(
        &self,
        system: &ReducedCameraSystem<'_>,
    ) -> Option<(Vec<nalgebra::Vector6<f64>>, usize)>;
}

static REDUCED_SOLVE: OnceLock<Box<dyn ReducedSolveAccelerator>> = OnceLock::new();

/// Register the process-wide reduced-camera solve accelerator. Returns
/// `false` if one was already registered (the first registration wins).
pub fn set_reduced_solve_accelerator(accelerator: Box<dyn ReducedSolveAccelerator>) -> bool {
    REDUCED_SOLVE.set(accelerator).is_ok()
}

pub(crate) fn reduced_solve_accelerator() -> Option<&'static dyn ReducedSolveAccelerator> {
    REDUCED_SOLVE.get().map(|a| a.as_ref())
}
