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
    /// Identifies `edges` across calls (0: unknown): while it stays the same
    /// an accelerator may reuse what it derived from them.
    pub structure_id: u64,
    /// Damped 6x6 diagonal blocks, one per free node.
    pub diag: &'a [nalgebra::Matrix6<f64>],
    /// Edge `e` couples nodes `edges[e] = (a, b)`, `a < b`; sorted.
    pub edges: &'a [(u32, u32)],
    /// `offdiag[36 e..36 e + 36]`: the (row b, column a) block of edge `e`,
    /// column-major. Empty: the blocks the accelerator's last
    /// [`ReducedSolveAccelerator::eliminate`] of this structure formed.
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

/// The point elimination that forms a [`ReducedCameraSystem`]'s
/// off-diagonal blocks, for an accelerator that forms them itself: the block
/// of edge `(s, t)` is `-Σ hcp[p, j] V_p⁻¹ hcp[p, k]ᵀ` over the points `p`
/// whose `k`-th and `j`-th nodes are `s` and `t`, plus any direct coupling
/// in `extra`. The diagonal blocks and the right-hand side stay with the
/// caller.
pub struct PointElimination<'a> {
    /// As [`ReducedCameraSystem::structure_id`]; covers every field but
    /// `hcp`, `point_inverse`, `scale` and `extra`.
    pub structure_id: u64,
    /// Changes whenever `hcp` does.
    pub linearization_id: u64,
    /// Sorted `(s, t)`, `s < t`, as in the system.
    pub edges: &'a [(u32, u32)],
    /// `edges[row_off[s]..row_off[s + 1]]` are the edges `(s, _)`.
    pub row_off: &'a [usize],
    /// `slots[slot_off[s]..slot_off[s + 1]]`: `(point, k)` for each point
    /// observing node `s`, ascending by point.
    pub slot_off: &'a [usize],
    pub slots: &'a [[u32; 2]],
    /// Point `p`'s blocks are `hcp[point_off[p]..point_off[p + 1]]`, their
    /// nodes `point_nodes[...]`, ascending.
    pub point_off: &'a [usize],
    pub point_nodes: &'a [u32],
    /// `J_nodeᵀ J_point` per (point, node).
    pub hcp: &'a [nalgebra::SMatrix<f64, 6, 3>],
    /// Damped point-block inverses; `None` leaves the point out.
    pub point_inverse: &'a [Option<nalgebra::Matrix3<f64>>],
    /// Per node, the square roots of the diagonal of `Σ hcp V⁻¹ hcpᵀ` over
    /// its points: by Cauchy-Schwarz, `|block(t, s)[r][c]| <=
    /// scale[t][r] * scale[s][c]`.
    pub scale: &'a [nalgebra::Vector6<f64>],
    /// Direct couplings `(edge, block)` added to the eliminated blocks.
    pub extra: &'a [(u32, nalgebra::Matrix6<f64>)],
    /// Edges whose final blocks to return (those inside preconditioner
    /// runs).
    pub run_edges: &'a [u32],
    /// `(nodes per aggregate, leading nodes covered)`: also return, for each
    /// aggregate pair `G <= H`, the sum of the blocks of edges from `G` to
    /// `H`.
    pub coarse: Option<(usize, usize)>,
}

/// What [`ReducedSolveAccelerator::eliminate`] returns to the caller.
pub struct EliminatedBlocks {
    /// The blocks of [`PointElimination::run_edges`], in that order.
    pub run_blocks: Vec<nalgebra::Matrix6<f64>>,
    /// `coarse_sums[g * m + h]` for `g <= h` (`m` aggregates), when
    /// [`PointElimination::coarse`] asked for them; empty otherwise.
    pub coarse_sums: Vec<nalgebra::Matrix6<f64>>,
}

/// A replacement for the COLMAP-port rig solver's iterative reduced-camera
/// solve (e.g. on a GPU). Registered once with
/// [`set_reduced_solve_accelerator`]; returning `None` falls back to the CPU
/// for that call.
pub trait ReducedSolveAccelerator: Send + Sync {
    /// The camera step (one 6-vector per node) and the iteration count.
    fn solve(
        &self,
        system: &ReducedCameraSystem<'_>,
    ) -> Option<(Vec<nalgebra::Vector6<f64>>, usize)>;

    /// Form the off-diagonal blocks itself, for the next
    /// [`solve`](Self::solve) of the same structure with empty `offdiag`.
    /// The default declines, and the caller eliminates on the CPU.
    fn eliminate(&self, elimination: &PointElimination<'_>) -> Option<EliminatedBlocks> {
        let _ = elimination;
        None
    }
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
