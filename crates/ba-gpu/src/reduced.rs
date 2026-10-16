//! GPU solve of the COLMAP-port rig solver's reduced camera system
//! ([`visloc_slam::ReducedSolveAccelerator`]).
//!
//! - **Conjugate gradients** (`shaders/reduced_pcg.wgsl`): the CPU solver's
//!   preconditioned iteration (dense inverses over runs of consecutive
//!   frames, the optional coarse level, the same stopping rule) in f32.
//! - **Point elimination** (`shaders/reduced_elim.wgsl`): the system's
//!   off-diagonal blocks, formed on the device and kept there for the
//!   solve, summed exactly in 64-bit fixed point. The CPU keeps the diagonal
//!   blocks and the right-hand side (f64) and factors the preconditioner
//!   from the blocks the device returns.
//!
//! Every reduction sums in a fixed order and the elimination's sums are
//! exact, so a solve is deterministic on a given device; its f32 arithmetic
//! makes the step match the CPU's to about the PCG tolerance, not
//! bit-for-bit. Structure-only uploads (the edge index, the elimination
//! index) are reused while the caller's structure id stays the same, the
//! camera-point blocks while its linearization id does.

use std::sync::Mutex;

use nalgebra::{Matrix6, Vector6};
use rayon::prelude::*;
use visloc_gsplat_render::GpuContext;
use visloc_slam::{
    EliminatedBlocks, PointElimination, ReducedCameraSystem, ReducedSolveAccelerator,
};

const ENTRIES: [&str; 14] = [
    "init",
    "init_b",
    "precond_cluster",
    "coarse_restrict",
    "coarse_apply",
    "precond_finish",
    "init_rz",
    "init_p",
    "matvec",
    "alpha",
    "update_xr",
    "check",
    "beta",
    "update_p",
];
const INIT: usize = 0;
const INIT_B: usize = 1;
const PRECOND_CLUSTER: usize = 2;
const COARSE_RESTRICT: usize = 3;
const COARSE_APPLY: usize = 4;
const PRECOND_FINISH: usize = 5;
const INIT_RZ: usize = 6;
const INIT_P: usize = 7;
const MATVEC: usize = 8;
const ALPHA: usize = 9;
const UPDATE_XR: usize = 10;
const CHECK: usize = 11;
const BETA: usize = 12;
const UPDATE_P: usize = 13;

/// PCG bindings: 0 is the uniform, 1-9 read-only storage, 10-18 read-write.
const NUM_BINDINGS: usize = 19;
const B_BLOCKS: usize = 1;
const B_DIAG: usize = 2;
const B_ROW_PTR: usize = 3;
const B_ROW_ENT: usize = 4;
const B_ROW_COL: usize = 5;
const B_CL_MAT: usize = 6;
const B_CL_OFF: usize = 7;
const B_COARSE: usize = 8;
const B_RHS: usize = 9;
const B_X: usize = 10;
const B_PARTIALS: usize = 15;
const B_RC: usize = 16;
const B_YC: usize = 17;
const B_SCAL: usize = 18;
/// `scal` slots read back by the host (see the shader).
const S_DONE: usize = 6;
const S_ITERS: usize = 7;
const S_FAILED: usize = 8;
const N_SCAL: usize = 16;

const ELIM_ENTRIES: [&str; 4] = ["eliminate", "add_extra", "gather_runs", "coarse_sums"];
const ELIMINATE: usize = 0;
const ADD_EXTRA: usize = 1;
const GATHER_RUNS: usize = 2;
const COARSE_SUMS: usize = 3;
/// Elimination bindings (`shaders/reduced_elim.wgsl`); 10 is the PCG's
/// block buffer.
const NUM_ELIM_BINDINGS: usize = 17;
const E_TASKS: usize = 1;
const E_EDGE_B: usize = 2;
const E_SLOT_OFF: usize = 3;
const E_SLOTS: usize = 4;
const E_POINT_OFF: usize = 5;
const E_POINT_NODES: usize = 6;
const E_HCP: usize = 7;
const E_VINV: usize = 8;
const E_SCALE: usize = 9;
const E_BLOCKS: usize = 10;
const E_EXTRA_EDGE: usize = 11;
const E_EXTRA_BLK: usize = 12;
const E_RUN_EDGES: usize = 13;
const E_RUN_OUT: usize = 14;
const E_ROW_OFF: usize = 15;
const E_COARSE_OUT: usize = 16;
const ELIM_WRITABLE: [usize; 3] = [E_BLOCKS, E_RUN_OUT, E_COARSE_OUT];
/// Most edges one elimination workgroup accumulates (its fixed-point sums
/// take 288 bytes of workgroup memory per edge).
const MAX_ELIM_CHUNK: u32 = 160;

/// Workgroups per dispatch dimension.
const MAX_GROUPS: u32 = 65_535;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    n_wg: u32,
    n_clusters: u32,
    cluster_k: u32,
    coarse_m: u32,
    coarse_group: u32,
    coarse_nodes: u32,
    max_iter: u32,
    tol2: f32,
    rows_x: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ElimParams {
    n_tasks: u32,
    tasks_x: u32,
    n_extra: u32,
    n_run_edges: u32,
    coarse_m: u32,
    coarse_group: u32,
    coarse_nodes: u32,
    _pad: u32,
}

/// One storage buffer per binding, grown on demand; a buffer keeps its
/// contents while it is large enough.
struct BufferSet {
    label: &'static str,
    bufs: Vec<Option<wgpu::Buffer>>,
    caps: Vec<u64>,
}

impl BufferSet {
    fn new(label: &'static str, n: usize) -> Self {
        Self {
            label,
            bufs: (0..n).map(|_| None).collect(),
            caps: vec![0; n],
        }
    }

    /// At least `size` bytes in binding `i`; `true` when (re)allocated,
    /// which loses the contents.
    fn ensure(&mut self, dev: &wgpu::Device, limit: u64, i: usize, size: u64) -> bool {
        let size = size.max(16);
        if self.bufs[i].is_some() && self.caps[i] >= size {
            return false;
        }
        // Grow geometrically so a slowly growing model does not reallocate
        // on every solve.
        let cap = (size + size / 4).min(limit / 4 * 4).max(size).div_ceil(4) * 4;
        self.bufs[i] = Some(dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{}-{i}", self.label)),
            size: cap,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        self.caps[i] = cap;
        true
    }

    fn get(&self, i: usize) -> &wgpu::Buffer {
        self.bufs[i].as_ref().expect("buffer ensured")
    }
}

struct State {
    pcg: BufferSet,
    elim: BufferSet,
    pcg_params: wgpu::Buffer,
    elim_params: wgpu::Buffer,
    /// Structure whose edge index (`row_*`) the PCG buffers hold.
    csr_of: Option<u64>,
    /// Structure whose elimination index the elimination buffers hold, and
    /// its task count.
    elim_of: Option<(u64, u32)>,
    /// (structure, linearization) whose camera-point blocks are uploaded.
    hcp_of: Option<(u64, u64)>,
    /// Structure whose blocks the last elimination left in the PCG's block
    /// buffer.
    resident: Option<u64>,
}

/// The reduced-camera solve on the GPU; register it with
/// [`visloc_slam::set_reduced_solve_accelerator`].
pub struct GpuReducedSolver {
    ctx: GpuContext,
    layout: wgpu::BindGroupLayout,
    pipes: Vec<wgpu::ComputePipeline>,
    elim_layout: wgpu::BindGroupLayout,
    elim_pipes: Vec<wgpu::ComputePipeline>,
    /// Edges per elimination workgroup.
    elim_chunk: u32,
    state: Mutex<State>,
    /// Iterations encoded per submission between reads of the done flag.
    chunk: u32,
}

/// Row-kernel grid for `count` workgroups and its x extent.
fn grid(count: u32) -> ((u32, u32), u32) {
    if count <= MAX_GROUPS {
        ((count.max(1), 1), count.max(1))
    } else {
        ((MAX_GROUPS, count.div_ceil(MAX_GROUPS)), MAX_GROUPS)
    }
}

fn layout_and_pipelines(
    dev: &wgpu::Device,
    label: &str,
    source: &str,
    entries: &[&str],
    bindings: usize,
    writable: impl Fn(usize) -> bool,
    constants: &[(&str, f64)],
) -> (wgpu::BindGroupLayout, Vec<wgpu::ComputePipeline>) {
    let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..bindings)
        .map(|b| wgpu::BindGroupLayoutEntry {
            binding: b as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if b == 0 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage {
                        read_only: !writable(b),
                    }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &layout_entries,
    });
    let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipes = entries
        .iter()
        .map(|e| {
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(e),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(e),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants,
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            })
        })
        .collect();
    (layout, pipes)
}

fn uniform(dev: &wgpu::Device, label: &str, size: usize) -> wgpu::Buffer {
    dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl GpuReducedSolver {
    pub fn new(ctx: GpuContext) -> Self {
        let dev = &ctx.device;
        let (layout, pipes) = layout_and_pipelines(
            dev,
            "reduced-pcg",
            include_str!("shaders/reduced_pcg.wgsl"),
            &ENTRIES,
            NUM_BINDINGS,
            |b| b > B_RHS,
            &[],
        );
        let shared = ctx.limits.max_compute_workgroup_storage_size;
        let elim_chunk = (shared.saturating_sub(2048) / 288).clamp(1, MAX_ELIM_CHUNK);
        let (elim_layout, elim_pipes) = layout_and_pipelines(
            dev,
            "reduced-elim",
            include_str!("shaders/reduced_elim.wgsl"),
            &ELIM_ENTRIES,
            NUM_ELIM_BINDINGS,
            |b| ELIM_WRITABLE.contains(&b),
            &[("CHUNK", elim_chunk as f64)],
        );
        let chunk = std::env::var("VISLOC_GPU_PCG_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&c: &u32| c > 0)
            .unwrap_or(32);
        let state = State {
            pcg: BufferSet::new("reduced-pcg", NUM_BINDINGS),
            elim: BufferSet::new("reduced-elim", NUM_ELIM_BINDINGS),
            pcg_params: uniform(dev, "reduced-pcg-params", std::mem::size_of::<Params>()),
            elim_params: uniform(
                dev,
                "reduced-elim-params",
                std::mem::size_of::<ElimParams>(),
            ),
            csr_of: None,
            elim_of: None,
            hcp_of: None,
            resident: None,
        };
        Self {
            ctx,
            layout,
            pipes,
            elim_layout,
            elim_pipes,
            elim_chunk,
            state: Mutex::new(state),
            chunk,
        }
    }

    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }

    /// Largest buffer a storage binding can cover.
    fn buffer_limit(&self) -> u64 {
        self.ctx
            .limits
            .max_storage_buffer_binding_size
            .min(self.ctx.limits.max_buffer_size)
    }

    fn read_f32(&self, buffer: &wgpu::Buffer, len: usize) -> Vec<f32> {
        if len == 0 {
            return Vec::new();
        }
        let dev = &self.ctx.device;
        let bytes = (len * 4) as u64;
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reduced-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("reduced-read"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
        self.ctx.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        dev.poll(wgpu::PollType::wait_indefinitely()).ok();
        let _ = rx.recv();
        let out: Vec<f32> =
            bytemuck::cast_slice(&slice.get_mapped_range().expect("map range")).to_vec();
        staging.unmap();
        out
    }

    /// The device blocks of the edges `0..n_edges` (tests).
    #[cfg(test)]
    fn read_blocks(&self, n_edges: usize) -> Vec<f32> {
        let st = self.state.lock().expect("state");
        self.read_f32(st.pcg.get(B_BLOCKS), 36 * n_edges)
    }

    fn eliminate_blocks(&self, e: &PointElimination<'_>) -> Option<EliminatedBlocks> {
        let n = e.row_off.len().checked_sub(1)?;
        let n_points = e.point_off.len().checked_sub(1)?;
        let n_edges = e.edges.len();
        let n_obs = *e.point_off.last()?;
        if e.structure_id == 0
            || n == 0
            || e.slot_off.len() != n + 1
            || e.scale.len() != n
            || e.point_inverse.len() != n_points
            || e.hcp.len() != n_obs
            || e.point_nodes.len() != n_obs
            || e.row_off[n] != n_edges
            || *e.slot_off.last()? != e.slots.len()
            || n_obs >= 1 << 32
            || n_edges >= 1 << 31
        {
            return None;
        }
        let coarse = match e.coarse {
            Some((group, nodes)) if group > 0 && nodes <= n => Some((group, nodes)),
            Some(_) => return None,
            None => None,
        };
        let coarse_m = coarse.map_or(0, |(g, nodes)| nodes.div_ceil(g));
        let dev = &self.ctx.device;
        let queue = &self.ctx.queue;
        let limit = self.buffer_limit();
        let f = |count: usize| (count * 4) as u64;
        let mut guard = self.state.lock().ok()?;
        let st = &mut *guard;

        // The structure: tasks of at most `elim_chunk` edges of one row, the
        // edges' far nodes, the per-node and per-point indexes.
        let mut sizes = [0u64; NUM_ELIM_BINDINGS];
        let task_bound = n_edges.div_ceil(self.elim_chunk as usize) + n;
        sizes[E_TASKS] = f(4 * task_bound);
        sizes[E_EDGE_B] = f(n_edges);
        sizes[E_SLOT_OFF] = f(n + 1);
        sizes[E_SLOTS] = f(2 * e.slots.len());
        sizes[E_POINT_OFF] = f(n_points + 1);
        sizes[E_POINT_NODES] = f(n_obs);
        sizes[E_HCP] = f(18 * n_obs);
        sizes[E_VINV] = f(9 * n_points);
        sizes[E_SCALE] = f(6 * n);
        sizes[E_EXTRA_EDGE] = f(e.extra.len());
        sizes[E_EXTRA_BLK] = f(36 * e.extra.len());
        sizes[E_RUN_EDGES] = f(e.run_edges.len());
        sizes[E_RUN_OUT] = f(36 * e.run_edges.len());
        sizes[E_ROW_OFF] = f(n + 1);
        sizes[E_COARSE_OUT] = f(36 * coarse_m * coarse_m);
        if sizes.iter().any(|&s| s > limit) || f(36 * n_edges) > limit {
            return None;
        }
        let mut lost_structure = false;
        let mut lost_hcp = false;
        for (b, &size) in sizes.iter().enumerate().skip(1) {
            if b == E_BLOCKS {
                continue;
            }
            let fresh = st.elim.ensure(dev, limit, b, size);
            lost_structure |= fresh && b != E_HCP && b < E_VINV;
            lost_structure |= fresh && b == E_ROW_OFF;
            lost_hcp |= fresh && b == E_HCP;
        }
        if lost_structure || st.elim_of.map(|(id, _)| id) != Some(e.structure_id) {
            let mut tasks: Vec<u32> = Vec::with_capacity(4 * task_bound);
            for s in 0..n {
                let (mut lo, hi) = (e.row_off[s], e.row_off[s + 1]);
                while lo < hi {
                    let end = hi.min(lo + self.elim_chunk as usize);
                    tasks.extend_from_slice(&[s as u32, lo as u32, end as u32, 0]);
                    lo = end;
                }
            }
            let to_u32 = |v: &[usize]| v.iter().map(|&x| x as u32).collect::<Vec<u32>>();
            let edge_b: Vec<u32> = e.edges.iter().map(|&(_, b)| b).collect();
            let (slot_off, point_off, row_off) =
                (to_u32(e.slot_off), to_u32(e.point_off), to_u32(e.row_off));
            let uploads: [(usize, &[u8]); 7] = [
                (E_TASKS, bytemuck::cast_slice(&tasks)),
                (E_EDGE_B, bytemuck::cast_slice(&edge_b)),
                (E_SLOT_OFF, bytemuck::cast_slice(&slot_off)),
                (E_SLOTS, bytemuck::cast_slice(e.slots)),
                (E_POINT_OFF, bytemuck::cast_slice(&point_off)),
                (E_POINT_NODES, bytemuck::cast_slice(e.point_nodes)),
                (E_ROW_OFF, bytemuck::cast_slice(&row_off)),
            ];
            for (b, bytes) in uploads {
                if !bytes.is_empty() {
                    queue.write_buffer(st.elim.get(b), 0, bytes);
                }
            }
            st.elim_of = Some((e.structure_id, (tasks.len() / 4) as u32));
            st.hcp_of = None;
        }
        let n_tasks = st.elim_of.map_or(0, |(_, t)| t);
        if lost_hcp || st.hcp_of != Some((e.structure_id, e.linearization_id)) {
            let mut hcp = vec![0.0f32; 18 * n_obs];
            hcp.par_chunks_mut(18)
                .zip(e.hcp.par_iter())
                .for_each(|(out, m)| {
                    for (o, &v) in out.iter_mut().zip(m.as_slice()) {
                        *o = v as f32;
                    }
                });
            queue.write_buffer(st.elim.get(E_HCP), 0, bytemuck::cast_slice(&hcp));
            st.hcp_of = Some((e.structure_id, e.linearization_id));
        }
        // Per call: the damped point inverses (zero for a skipped point), the
        // bounds, the couplings.
        let mut vinv = vec![0.0f32; 9 * n_points];
        vinv.par_chunks_mut(9)
            .zip(e.point_inverse.par_iter())
            .for_each(|(out, m)| {
                if let Some(m) = m {
                    for (o, &v) in out.iter_mut().zip(m.as_slice()) {
                        *o = v as f32;
                    }
                }
            });
        // A zero bound has no terms to bound; keep the division finite.
        let scale: Vec<f32> = e
            .scale
            .iter()
            .flat_map(|v| v.iter().map(|&x| (x as f32).max(1e-30)))
            .collect();
        let extra_edge: Vec<u32> = e.extra.iter().map(|&(edge, _)| edge).collect();
        let extra_blk: Vec<f32> = e
            .extra
            .iter()
            .flat_map(|(_, m)| m.as_slice().iter().map(|&v| v as f32))
            .collect();
        for (b, bytes) in [
            (E_VINV, bytemuck::cast_slice::<f32, u8>(&vinv)),
            (E_SCALE, bytemuck::cast_slice(&scale)),
            (E_EXTRA_EDGE, bytemuck::cast_slice(&extra_edge)),
            (E_EXTRA_BLK, bytemuck::cast_slice(&extra_blk)),
            (E_RUN_EDGES, bytemuck::cast_slice(e.run_edges)),
        ] {
            if !bytes.is_empty() {
                queue.write_buffer(st.elim.get(b), 0, bytes);
            }
        }
        // The blocks land in the PCG's block buffer; its old contents go.
        st.pcg.ensure(dev, limit, B_BLOCKS, f(36 * n_edges));
        st.resident = None;
        let ((tx, ty), tasks_x) = grid(n_tasks);
        let params = ElimParams {
            n_tasks,
            tasks_x,
            n_extra: e.extra.len() as u32,
            n_run_edges: e.run_edges.len() as u32,
            coarse_m: coarse_m as u32,
            coarse_group: coarse.map_or(1, |(g, _)| g as u32),
            coarse_nodes: coarse.map_or(0, |(_, nodes)| nodes as u32),
            _pad: 0,
        };
        queue.write_buffer(&st.elim_params, 0, bytemuck::bytes_of(&params));
        let entries: Vec<wgpu::BindGroupEntry> = (0..NUM_ELIM_BINDINGS)
            .map(|b| wgpu::BindGroupEntry {
                binding: b as u32,
                resource: match b {
                    0 => st.elim_params.as_entire_binding(),
                    E_BLOCKS => st.pcg.get(B_BLOCKS).as_entire_binding(),
                    _ => st.elim.get(b).as_entire_binding(),
                },
            })
            .collect();
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reduced-elim"),
            layout: &self.elim_layout,
            entries: &entries,
        });
        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("reduced-elim"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduced-elim"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bind, &[]);
            if n_tasks > 0 {
                pass.set_pipeline(&self.elim_pipes[ELIMINATE]);
                pass.dispatch_workgroups(tx, ty, 1);
            }
            for (k, count) in [
                (ADD_EXTRA, 36 * e.extra.len()),
                (GATHER_RUNS, 36 * e.run_edges.len()),
            ] {
                if count > 0 {
                    pass.set_pipeline(&self.elim_pipes[k]);
                    pass.dispatch_workgroups(count.div_ceil(64) as u32, 1, 1);
                }
            }
            if coarse_m > 0 {
                pass.set_pipeline(&self.elim_pipes[COARSE_SUMS]);
                pass.dispatch_workgroups(coarse_m as u32, 1, 1);
            }
        }
        queue.submit(Some(encoder.finish()));
        let run = self.read_f32(st.elim.get(E_RUN_OUT), 36 * e.run_edges.len());
        let sums = self.read_f32(st.elim.get(E_COARSE_OUT), 36 * coarse_m * coarse_m);
        let block = |c: &[f32]| Matrix6::from_iterator(c.iter().map(|&v| v as f64));
        let run_blocks: Vec<Matrix6<f64>> = run.chunks_exact(36).map(block).collect();
        let coarse_sums: Vec<Matrix6<f64>> = sums.chunks_exact(36).map(block).collect();
        if !run_blocks
            .iter()
            .chain(&coarse_sums)
            .all(|m| m.iter().all(|v| v.is_finite()))
        {
            return None;
        }
        st.resident = Some(e.structure_id);
        Some(EliminatedBlocks {
            run_blocks,
            coarse_sums,
        })
    }

    fn solve_system(&self, sys: &ReducedCameraSystem<'_>) -> Option<(Vec<Vector6<f64>>, usize)> {
        let n = sys.diag.len();
        let n_edges = sys.edges.len();
        let resident = sys.offdiag.is_empty() && n_edges > 0;
        if n == 0
            || sys.rhs.len() != n
            || (!resident && sys.offdiag.len() != n_edges * 36)
            || n_edges >= 1 << 31
            || 6 * n > (MAX_GROUPS as usize) * 256
        {
            return None;
        }
        // Scale the right-hand side to unit max norm (the stopping rule is
        // scale-free) so f32 dot products stay far from overflow.
        let scale = sys
            .rhs
            .iter()
            .flat_map(|v| v.iter())
            .fold(0.0f64, |m, v| m.max(v.abs()));
        if !scale.is_finite() {
            return None;
        }
        if scale == 0.0 {
            return Some((vec![Vector6::zeros(); n], 0));
        }
        let rhs: Vec<f32> = sys
            .rhs
            .iter()
            .flat_map(|v| v.iter().map(move |&b| (b / scale) as f32))
            .collect();
        let diag: Vec<f32> = sys
            .diag
            .iter()
            .flat_map(|d| d.as_slice().iter().map(|&v| v as f32))
            .collect();

        // Preconditioner runs: the CPU's cluster inverses, or block-Jacobi
        // (runs of one node).
        let (cluster_k, cl_off, cl_mat) = match sys.clusters {
            Some(clusters) => {
                let k = clusters.first()?.1.nrows() / 6;
                let mut off = Vec::with_capacity(clusters.len() + 1);
                let mut total = 0usize;
                for (c, (start, inv)) in clusters.iter().enumerate() {
                    let m = k.min(n.checked_sub(c * k)?);
                    if *start != c * k || inv.nrows() != 6 * m || inv.ncols() != 6 * m {
                        return None;
                    }
                    off.push(total as u32);
                    total += 36 * m * m;
                }
                if clusters.len() != n.div_ceil(k) {
                    return None;
                }
                off.push(total as u32);
                let mat: Vec<f32> = clusters
                    .iter()
                    .flat_map(|(_, inv)| inv.as_slice().iter().map(|&v| v as f32))
                    .collect();
                (k, off, mat)
            }
            None => {
                let inverses: Vec<Matrix6<f64>> = sys
                    .diag
                    .par_iter()
                    .map(|d| d.cholesky().map(|c| c.inverse()))
                    .collect::<Option<Vec<_>>>()?;
                let off: Vec<u32> = (0..=n as u32).map(|i| 36 * i).collect();
                let mat: Vec<f32> = inverses
                    .iter()
                    .flat_map(|m| m.as_slice().iter().map(|&v| v as f32))
                    .collect();
                (1, off, mat)
            }
        };
        let n_clusters = n.div_ceil(cluster_k);
        let (coarse_m, coarse_group, coarse_nodes, coarse_inv) = match sys.coarse {
            Some((group, nodes, inv)) => {
                let m = inv.nrows() / 6;
                if group == 0 || nodes > n || m == 0 || m != nodes.div_ceil(group) {
                    return None;
                }
                let mat: Vec<f32> = inv.as_slice().iter().map(|&v| v as f32).collect();
                (m, group, nodes, mat)
            }
            None => (0, 1, 0, vec![0.0f32]),
        };

        let n6 = 6 * n;
        let n_wg = n6.div_ceil(256);
        // Both row grids index workgroups as `x + y * rows_x`: the cluster
        // grid needs a second dimension only when the node grid does
        // (n_clusters <= n), and then both are MAX_GROUPS wide.
        let ((mx, my), rows_x) = grid(n as u32);
        let ((cx, cy), _) = grid(n_clusters as u32);
        let params = Params {
            n: n as u32,
            n_wg: n_wg as u32,
            n_clusters: n_clusters as u32,
            cluster_k: cluster_k as u32,
            coarse_m: coarse_m as u32,
            coarse_group: coarse_group as u32,
            coarse_nodes: coarse_nodes as u32,
            max_iter: sys.max_iterations as u32,
            tol2: (sys.tol * sys.tol) as f32,
            rows_x,
            _pad: [0; 2],
        };
        let f = |count: usize| (count * 4) as u64;
        let mut sizes = [0u64; NUM_BINDINGS];
        sizes[B_BLOCKS] = f(36 * n_edges);
        sizes[B_DIAG] = f(diag.len());
        sizes[B_ROW_PTR] = f(n + 1);
        sizes[B_ROW_ENT] = f(2 * n_edges);
        sizes[B_ROW_COL] = f(2 * n_edges);
        sizes[B_CL_MAT] = f(cl_mat.len());
        sizes[B_CL_OFF] = f(cl_off.len());
        sizes[B_COARSE] = f(coarse_inv.len());
        sizes[B_RHS] = f(n6);
        for s in &mut sizes[B_X..B_X + 5] {
            *s = f(n6);
        }
        sizes[B_PARTIALS] = f(n.max(n_wg));
        sizes[B_RC] = f((6 * coarse_m).max(1));
        sizes[B_YC] = f((6 * coarse_m).max(1));
        sizes[B_SCAL] = f(N_SCAL);
        let limit = self.buffer_limit();
        if sizes.iter().any(|&s| s > limit) {
            return None;
        }

        let dev = &self.ctx.device;
        let queue = &self.ctx.queue;
        let mut guard = self.state.lock().ok()?;
        let st = &mut *guard;
        if resident && (sys.structure_id == 0 || st.resident != Some(sys.structure_id)) {
            return None;
        }
        let mut lost_csr = false;
        for (b, &size) in sizes.iter().enumerate().skip(1) {
            let fresh = st.pcg.ensure(dev, limit, b, size);
            lost_csr |= fresh && (B_ROW_PTR..=B_ROW_COL).contains(&b);
            if fresh && b == B_BLOCKS && resident {
                return None;
            }
        }
        if !resident {
            queue.write_buffer(st.pcg.get(B_BLOCKS), 0, bytemuck::cast_slice(sys.offdiag));
            st.resident = None;
        }
        if lost_csr || sys.structure_id == 0 || st.csr_of != Some(sys.structure_id) {
            // Both block rows of every edge: row b reads the block as
            // stored, row a its transpose (bit 31), each in edge order.
            let mut row_ptr = vec![0u32; n + 1];
            for &(a, b) in sys.edges {
                if a as usize >= n || b as usize >= n || a == b {
                    return None;
                }
                row_ptr[a as usize + 1] += 1;
                row_ptr[b as usize + 1] += 1;
            }
            for i in 0..n {
                row_ptr[i + 1] += row_ptr[i];
            }
            let mut cursor: Vec<u32> = row_ptr[..n].to_vec();
            let mut row_ent = vec![0u32; 2 * n_edges];
            let mut row_col = vec![0u32; 2 * n_edges];
            for (e, &(a, b)) in sys.edges.iter().enumerate() {
                let ia = cursor[a as usize] as usize;
                row_ent[ia] = e as u32 | 1 << 31;
                row_col[ia] = b;
                cursor[a as usize] += 1;
                let ib = cursor[b as usize] as usize;
                row_ent[ib] = e as u32;
                row_col[ib] = a;
                cursor[b as usize] += 1;
            }
            for (b, data) in [
                (B_ROW_PTR, &row_ptr),
                (B_ROW_ENT, &row_ent),
                (B_ROW_COL, &row_col),
            ] {
                if !data.is_empty() {
                    queue.write_buffer(st.pcg.get(b), 0, bytemuck::cast_slice(data));
                }
            }
            st.csr_of = (sys.structure_id != 0).then_some(sys.structure_id);
        }
        queue.write_buffer(&st.pcg_params, 0, bytemuck::bytes_of(&params));
        for (b, bytes) in [
            (B_DIAG, bytemuck::cast_slice::<f32, u8>(&diag)),
            (B_CL_MAT, bytemuck::cast_slice(&cl_mat)),
            (B_CL_OFF, bytemuck::cast_slice(&cl_off)),
            (B_COARSE, bytemuck::cast_slice(&coarse_inv)),
            (B_RHS, bytemuck::cast_slice(&rhs)),
        ] {
            if !bytes.is_empty() {
                queue.write_buffer(st.pcg.get(b), 0, bytes);
            }
        }
        let entries: Vec<wgpu::BindGroupEntry> = (0..NUM_BINDINGS)
            .map(|b| wgpu::BindGroupEntry {
                binding: b as u32,
                resource: if b == 0 {
                    st.pcg_params.as_entire_binding()
                } else {
                    st.pcg.get(b).as_entire_binding()
                },
            })
            .collect();
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reduced-pcg"),
            layout: &self.layout,
            entries: &entries,
        });

        let elem = n_wg as u32;
        let coarse_wg = (6 * coarse_m).div_ceil(256) as u32;
        let encode = |pass: &mut wgpu::ComputePass<'_>, k: usize| {
            pass.set_pipeline(&self.pipes[k]);
            match k {
                MATVEC => pass.dispatch_workgroups(mx, my, 1),
                PRECOND_CLUSTER => pass.dispatch_workgroups(cx, cy, 1),
                COARSE_RESTRICT | COARSE_APPLY => pass.dispatch_workgroups(coarse_wg, 1, 1),
                INIT_B | INIT_RZ | ALPHA | CHECK | BETA => pass.dispatch_workgroups(1, 1, 1),
                _ => pass.dispatch_workgroups(elem, 1, 1),
            }
        };
        let precond = |pass: &mut wgpu::ComputePass<'_>| {
            encode(pass, PRECOND_CLUSTER);
            if coarse_m > 0 {
                encode(pass, COARSE_RESTRICT);
                encode(pass, COARSE_APPLY);
            }
            encode(pass, PRECOND_FINISH);
        };
        let submit = |with_init: bool, iterations: u32| {
            let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reduced-pcg"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("reduced-pcg"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &bind, &[]);
                if with_init {
                    encode(&mut pass, INIT);
                    encode(&mut pass, INIT_B);
                    precond(&mut pass);
                    encode(&mut pass, INIT_RZ);
                    encode(&mut pass, INIT_P);
                }
                for _ in 0..iterations {
                    encode(&mut pass, MATVEC);
                    encode(&mut pass, ALPHA);
                    encode(&mut pass, UPDATE_XR);
                    encode(&mut pass, CHECK);
                    precond(&mut pass);
                    encode(&mut pass, BETA);
                    encode(&mut pass, UPDATE_P);
                }
            }
            queue.submit(Some(encoder.finish()));
        };

        let max = sys.max_iterations as u32;
        let mut issued = self.chunk.min(max);
        submit(true, issued);
        loop {
            let scal = self.read_f32(st.pcg.get(B_SCAL), N_SCAL);
            if scal[S_DONE] != 0.0 || issued >= max {
                break;
            }
            let more = self.chunk.min(max - issued);
            submit(false, more);
            issued += more;
        }
        let scal = self.read_f32(st.pcg.get(B_SCAL), N_SCAL);
        if scal[S_FAILED] != 0.0 {
            return None;
        }
        let x = self.read_f32(st.pcg.get(B_X), n6);
        if !x.iter().all(|v| v.is_finite()) {
            return None;
        }
        let steps = x
            .chunks_exact(6)
            .map(|c| Vector6::from_fn(|k, _| c[k] as f64 * scale))
            .collect();
        Some((steps, scal[S_ITERS] as usize))
    }
}

impl ReducedSolveAccelerator for GpuReducedSolver {
    fn solve(&self, system: &ReducedCameraSystem<'_>) -> Option<(Vec<Vector6<f64>>, usize)> {
        self.solve_system(system)
    }

    fn eliminate(&self, elimination: &PointElimination<'_>) -> Option<EliminatedBlocks> {
        self.eliminate_blocks(elimination)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    struct Lcg(u64);

    impl Lcg {
        /// Uniform in [-1, 1).
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        }
    }

    struct System {
        diag: Vec<Matrix6<f64>>,
        edges: Vec<(u32, u32)>,
        offdiag: Vec<f32>,
        rhs: Vec<Vector6<f64>>,
    }

    /// `JᵀJ` of random relative constraints along a capture-ordered chain
    /// (four neighbours each, plus a few long-range edges), with node scales
    /// spanning two decades and a weak prior: SPD and Laplacian-like, so
    /// conjugate gradients take many iterations, like a long camera system.
    fn random_system(n: usize, seed: u64) -> System {
        let mut rng = Lcg(seed);
        let mut edges = Vec::new();
        for b in 1..n {
            for d in 1..=4.min(b) {
                edges.push(((b - d) as u32, b as u32));
            }
        }
        for _ in 0..n / 4 {
            let a = (rng.next().abs() * n as f64) as usize % n;
            let b = (rng.next().abs() * n as f64) as usize % n;
            if a + 4 < b {
                edges.push((a as u32, b as u32));
            }
        }
        let scale: Vec<f64> = (0..n).map(|_| 10f64.powf(rng.next() + 1.0)).collect();
        let mut diag: Vec<Matrix6<f64>> = (0..n)
            .map(|i| Matrix6::identity() * 1e-3 * scale[i] * scale[i])
            .collect();
        let mut offdiag = Vec::with_capacity(36 * edges.len());
        for &(a, b) in &edges {
            let (a, b) = (a as usize, b as usize);
            // A relative constraint `M (x_b - x_a)` (in scaled variables),
            // which makes the system Laplacian-like.
            let m = Matrix6::from_fn(|_, _| rng.next()) + Matrix6::identity() * 2.0;
            let fa = -m * scale[a];
            let fb = m * scale[b];
            diag[a] += fa.transpose() * fa;
            diag[b] += fb.transpose() * fb;
            // The (row b, column a) block.
            let blk = fb.transpose() * fa;
            offdiag.extend(blk.as_slice().iter().map(|&v| v as f32));
        }
        let rhs = (0..n)
            .map(|i| Vector6::from_fn(|_, _| rng.next()) * scale[i] * 1e3)
            .collect();
        System {
            diag,
            edges,
            offdiag,
            rhs,
        }
    }

    fn dense(sys: &System) -> DMatrix<f64> {
        let n = sys.diag.len();
        let mut a = DMatrix::<f64>::zeros(6 * n, 6 * n);
        for (i, d) in sys.diag.iter().enumerate() {
            a.view_mut((6 * i, 6 * i), (6, 6)).copy_from(d);
        }
        for (e, &(ea, eb)) in sys.edges.iter().enumerate() {
            let blk = Matrix6::from_column_slice(
                &sys.offdiag[36 * e..36 * e + 36]
                    .iter()
                    .map(|&v| v as f64)
                    .collect::<Vec<_>>(),
            );
            let (ea, eb) = (6 * ea as usize, 6 * eb as usize);
            a.view_mut((eb, ea), (6, 6)).copy_from(&blk);
            a.view_mut((ea, eb), (6, 6)).copy_from(&blk.transpose());
        }
        a
    }

    fn cluster_inverses(a: &DMatrix<f64>, n: usize, k: usize) -> Vec<(usize, DMatrix<f64>)> {
        (0..n)
            .step_by(k)
            .map(|s| {
                let m = k.min(n - s);
                let sub = a.view((6 * s, 6 * s), (6 * m, 6 * m)).clone_owned();
                (s, sub.cholesky().expect("SPD run").inverse())
            })
            .collect()
    }

    fn coarse_matrix(a: &DMatrix<f64>, nodes: usize, group: usize) -> DMatrix<f64> {
        let m = nodes.div_ceil(group);
        let mut p = DMatrix::<f64>::zeros(a.nrows(), 6 * m);
        for i in 0..nodes {
            for k in 0..6 {
                p[(6 * i + k, 6 * (i / group) + k)] = 1.0;
            }
        }
        p.transpose() * a * p
    }

    /// The CPU solver's iteration (`pcg_core`) in f64 on a dense matrix.
    fn reference(
        a: &DMatrix<f64>,
        b: &DVector<f64>,
        clusters: &[(usize, DMatrix<f64>)],
        coarse: Option<(usize, usize, &DMatrix<f64>)>,
        tol: f64,
    ) -> (DVector<f64>, usize) {
        let apply = |r: &DVector<f64>| {
            let mut z = DVector::<f64>::zeros(r.len());
            for (s, inv) in clusters {
                let rows = inv.nrows();
                let zc = inv * r.rows(6 * s, rows);
                z.rows_mut(6 * s, rows).copy_from(&zc);
            }
            if let Some((group, nodes, inv)) = coarse {
                let mut rc = DVector::<f64>::zeros(inv.nrows());
                for i in 0..nodes {
                    for k in 0..6 {
                        rc[6 * (i / group) + k] += r[6 * i + k];
                    }
                }
                let yc = inv * rc;
                for i in 0..nodes {
                    for k in 0..6 {
                        z[6 * i + k] += yc[6 * (i / group) + k];
                    }
                }
            }
            z
        };
        let mut x = DVector::<f64>::zeros(b.len());
        let mut r = b.clone();
        let mut z = apply(&r);
        let mut p = z.clone();
        let mut rz = r.dot(&z);
        let mut it = 0;
        for _ in 0..500 {
            it += 1;
            let ap = a * &p;
            let alpha = rz / p.dot(&ap);
            x += alpha * &p;
            r -= alpha * &ap;
            if r.norm() <= tol * b.norm() {
                break;
            }
            z = apply(&r);
            let rz_next = r.dot(&z);
            p = &z + (rz_next / rz) * &p;
            rz = rz_next;
        }
        (x, it)
    }

    #[test]
    fn gpu_pcg_matches_the_cpu_iteration() {
        let Some(ctx) = visloc_gsplat_render::try_context() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let solver = GpuReducedSolver::new(ctx);
        // (nodes, run length, coarse aggregate size); runs of one node are
        // the block-Jacobi fallback.
        for (n, k, group) in [
            (300, 32, Some(16)),
            (150, 1, None),
            (77, 32, None),
            (61, 8, Some(7)),
        ] {
            let sys = random_system(n, 7 + n as u64);
            let a = dense(&sys);
            let clusters = cluster_inverses(&a, n, k);
            let coarse = group.map(|g| {
                let c = coarse_matrix(&a, n, g);
                (g, c.cholesky().expect("SPD coarse").inverse())
            });
            let b = DVector::from_iterator(6 * n, sys.rhs.iter().flat_map(|v| v.iter().copied()));
            let tol = 1e-2;
            let coarse_ref = coarse.as_ref().map(|(g, inv)| (*g, n, inv));
            let (x_ref, it_ref) = reference(&a, &b, &clusters, coarse_ref, tol);
            let system = ReducedCameraSystem {
                structure_id: 0,
                diag: &sys.diag,
                edges: &sys.edges,
                offdiag: &sys.offdiag,
                rhs: &sys.rhs,
                clusters: (k > 1).then_some(&clusters[..]),
                coarse: coarse_ref,
                tol,
                max_iterations: 500,
            };
            let (steps, it) = solver.solve(&system).expect("GPU solve");
            let x = DVector::from_iterator(6 * n, steps.iter().flat_map(|v| v.iter().copied()));
            let residual = (&b - &a * &x).norm() / b.norm();
            let diff = (&x - &x_ref).norm() / x_ref.norm();
            eprintln!(
                "n={n} k={k} group={group:?}: {it} iterations (f64 {it_ref}), \
                 residual {residual:.2e}, |x - x_f64| / |x_f64| = {diff:.2e}"
            );
            assert!(it_ref > 3, "the system should take a few iterations");
            assert!(
                it.abs_diff(it_ref) <= 1 + it_ref / 10,
                "{it} vs {it_ref} iterations"
            );
            assert!(residual <= 1.2 * tol, "residual {residual}");
            assert!(diff < 1e-3, "step differs by {diff}");
        }
    }

    /// A synthetic point elimination: tracks of 2-12 nodes in a window
    /// along a chain, random camera-point blocks, SPD point blocks, a few
    /// direct couplings.
    struct Elimination {
        n: usize,
        edges: Vec<(u32, u32)>,
        row_off: Vec<usize>,
        slot_off: Vec<usize>,
        slots: Vec<[u32; 2]>,
        point_off: Vec<usize>,
        point_nodes: Vec<u32>,
        hcp: Vec<nalgebra::SMatrix<f64, 6, 3>>,
        inverse: Vec<Option<nalgebra::Matrix3<f64>>>,
        scale: Vec<Vector6<f64>>,
        extra: Vec<(u32, Matrix6<f64>)>,
        /// The blocks in f64 (`-Σ hcp_j V⁻¹ hcp_kᵀ` plus the couplings).
        blocks: Vec<Matrix6<f64>>,
    }

    fn random_elimination(n: usize, n_points: usize, seed: u64) -> Elimination {
        let mut rng = Lcg(seed);
        let mut point_off = vec![0usize];
        let mut point_nodes = Vec::new();
        let mut hcp = Vec::new();
        let mut inverse = Vec::new();
        for p in 0..n_points {
            let len = 2 + (rng.next().abs() * 11.0) as usize;
            let start = ((rng.next().abs() * n as f64) as usize).min(n - 2);
            let mut nodes: Vec<u32> = (0..len)
                .map(|_| (start + (rng.next().abs() * 16.0) as usize).min(n - 1) as u32)
                .collect();
            nodes.sort_unstable();
            nodes.dedup();
            if nodes.len() < 2 {
                nodes = vec![start as u32, start as u32 + 1];
            }
            let blocks: Vec<nalgebra::SMatrix<f64, 6, 3>> = nodes
                .iter()
                .map(|_| nalgebra::SMatrix::<f64, 6, 3>::from_fn(|_, _| rng.next()) * 30.0)
                .collect();
            let v = blocks
                .iter()
                .fold(nalgebra::Matrix3::<f64>::identity(), |acc, b| {
                    acc + b.transpose() * b
                });
            // Every 37th point is left out, like a degenerate point.
            inverse.push((p % 37 != 5).then(|| v.try_inverse().expect("SPD")));
            point_nodes.extend_from_slice(&nodes);
            hcp.extend(blocks);
            point_off.push(point_nodes.len());
        }
        let mut edges: Vec<(u32, u32)> = Vec::new();
        for p in 0..n_points {
            let nodes = &point_nodes[point_off[p]..point_off[p + 1]];
            for k in 0..nodes.len() {
                for j in k + 1..nodes.len() {
                    edges.push((nodes[k], nodes[j]));
                }
            }
        }
        edges.sort_unstable();
        edges.dedup();
        let row_off: Vec<usize> = (0..=n)
            .map(|s| edges.partition_point(|&(a, _)| (a as usize) < s))
            .collect();
        let mut per_node: Vec<Vec<[u32; 2]>> = vec![Vec::new(); n];
        for p in 0..n_points {
            for (k, &s) in point_nodes[point_off[p]..point_off[p + 1]]
                .iter()
                .enumerate()
            {
                per_node[s as usize].push([p as u32, k as u32]);
            }
        }
        let mut slot_off = vec![0usize];
        let mut slots = Vec::new();
        for list in &per_node {
            slots.extend_from_slice(list);
            slot_off.push(slots.len());
        }
        let mut blocks = vec![Matrix6::<f64>::zeros(); edges.len()];
        let mut d = vec![Matrix6::<f64>::zeros(); n];
        for p in 0..n_points {
            let Some(vinv) = inverse[p] else {
                continue;
            };
            let nodes = &point_nodes[point_off[p]..point_off[p + 1]];
            let h = &hcp[point_off[p]..point_off[p + 1]];
            for k in 0..nodes.len() {
                let scaled = h[k] * vinv;
                d[nodes[k] as usize] += scaled * h[k].transpose();
                for j in k + 1..nodes.len() {
                    let e = edges.binary_search(&(nodes[k], nodes[j])).expect("edge");
                    blocks[e] -= h[j] * scaled.transpose();
                }
            }
        }
        let scale = d
            .iter()
            .map(|m| Vector6::from_fn(|i, _| m[(i, i)].max(0.0).sqrt()))
            .collect();
        let extra: Vec<(u32, Matrix6<f64>)> = (0..edges.len() as u32)
            .step_by(97)
            .map(|e| (e, Matrix6::from_fn(|_, _| rng.next())))
            .collect();
        for (e, m) in &extra {
            blocks[*e as usize] += m;
        }
        Elimination {
            n,
            edges,
            row_off,
            slot_off,
            slots,
            point_off,
            point_nodes,
            hcp,
            inverse,
            scale,
            extra,
            blocks,
        }
    }

    #[test]
    fn gpu_elimination_matches_the_f64_sums() {
        let Some(ctx) = visloc_gsplat_render::try_context() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let solver = GpuReducedSolver::new(ctx);
        let el = random_elimination(150, 4000, 11);
        let k = 8;
        let (group, m) = (10usize, 15usize);
        let run_edges: Vec<u32> = (0..el.edges.len() as u32)
            .filter(|&e| {
                let (a, b) = el.edges[e as usize];
                a as usize / k == b as usize / k
            })
            .collect();
        let input = PointElimination {
            structure_id: 1,
            linearization_id: 1,
            edges: &el.edges,
            row_off: &el.row_off,
            slot_off: &el.slot_off,
            slots: &el.slots,
            point_off: &el.point_off,
            point_nodes: &el.point_nodes,
            hcp: &el.hcp,
            point_inverse: &el.inverse,
            scale: &el.scale,
            extra: &el.extra,
            run_edges: &run_edges,
            coarse: Some((group, el.n)),
        };
        let out = solver.eliminate(&input).expect("GPU elimination");
        let device = solver.read_blocks(el.edges.len());
        // Errors relative to each entry's Cauchy-Schwarz bound.
        let mut worst = 0.0f64;
        for (e, &(a, b)) in el.edges.iter().enumerate() {
            for c in 0..6 {
                for r in 0..6 {
                    let bound = el.scale[b as usize][r] * el.scale[a as usize][c] + 1.0;
                    let err = (device[36 * e + c * 6 + r] as f64 - el.blocks[e][(r, c)]).abs();
                    worst = worst.max(err / bound);
                }
            }
        }
        eprintln!("{} edges; worst error / bound {worst:.2e}", el.edges.len());
        assert!(worst < 1e-5, "blocks differ: {worst}");
        assert_eq!(out.run_blocks.len(), run_edges.len());
        for (&e, blk) in run_edges.iter().zip(&out.run_blocks) {
            let i = e as usize * 36;
            for q in 0..36 {
                assert_eq!(blk.as_slice()[q], device[i + q] as f64);
            }
        }
        assert_eq!(out.coarse_sums.len(), m * m);
        for g in 0..m {
            for h in g..m {
                let mut want = Matrix6::<f64>::zeros();
                for (e, &(a, b)) in el.edges.iter().enumerate() {
                    if a as usize / group == g && b as usize / group == h {
                        want += el.blocks[e];
                    }
                }
                let got = &out.coarse_sums[g * m + h];
                let err = (got - want).abs().max();
                assert!(
                    err <= 1e-3 * (1.0 + want.abs().max()),
                    "coarse ({g}, {h}): {err}"
                );
            }
        }
        // The same elimination again (cached structure and blocks) gives the
        // same blocks: the fixed-point sums do not depend on scheduling.
        solver.eliminate(&input).expect("GPU elimination");
        assert_eq!(device, solver.read_blocks(el.edges.len()));
    }

    #[test]
    fn resident_blocks_solve_like_uploaded_ones() {
        let Some(ctx) = visloc_gsplat_render::try_context() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let solver = GpuReducedSolver::new(ctx);
        let el = random_elimination(120, 3000, 5);
        let input = PointElimination {
            structure_id: 7,
            linearization_id: 3,
            edges: &el.edges,
            row_off: &el.row_off,
            slot_off: &el.slot_off,
            slots: &el.slots,
            point_off: &el.point_off,
            point_nodes: &el.point_nodes,
            hcp: &el.hcp,
            point_inverse: &el.inverse,
            scale: &el.scale,
            extra: &el.extra,
            run_edges: &[],
            coarse: None,
        };
        // A diagonal that dominates the eliminated blocks.
        let diag: Vec<Matrix6<f64>> = (0..el.n)
            .map(|i| {
                let row: f64 = el
                    .edges
                    .iter()
                    .zip(&el.blocks)
                    .filter(|(&(a, b), _)| a as usize == i || b as usize == i)
                    .map(|(_, m)| m.abs().max())
                    .sum();
                Matrix6::identity() * (6.0 * row + 1.0)
            })
            .collect();
        let rhs: Vec<Vector6<f64>> = (0..el.n)
            .map(|i| Vector6::from_fn(|r, _| ((i * 6 + r) % 7) as f64 - 3.0))
            .collect();
        let offdiag: Vec<f32> = el
            .blocks
            .iter()
            .flat_map(|m| m.as_slice().iter().map(|&v| v as f32))
            .collect();
        fn system<'a>(
            diag: &'a [Matrix6<f64>],
            edges: &'a [(u32, u32)],
            offdiag: &'a [f32],
            rhs: &'a [Vector6<f64>],
        ) -> ReducedCameraSystem<'a> {
            ReducedCameraSystem {
                structure_id: 7,
                diag,
                edges,
                offdiag,
                rhs,
                clusters: None,
                coarse: None,
                tol: 1e-6,
                max_iterations: 500,
            }
        }
        let (uploaded, it_up) = solver
            .solve(&system(&diag, &el.edges, &offdiag, &rhs))
            .expect("uploaded solve");
        // Without an elimination there are no resident blocks.
        assert!(solver.solve(&system(&diag, &el.edges, &[], &rhs)).is_none());
        solver.eliminate(&input).expect("GPU elimination");
        let (resident, it_res) = solver
            .solve(&system(&diag, &el.edges, &[], &rhs))
            .expect("resident solve");
        let diff = uploaded
            .iter()
            .zip(&resident)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0f64, f64::max);
        let size = uploaded.iter().map(|v| v.norm()).fold(0.0f64, f64::max);
        eprintln!("iterations {it_up} vs {it_res}; max step difference {diff:.2e} of {size:.2e}");
        assert!(it_up.abs_diff(it_res) <= 1);
        assert!(diff <= 1e-4 * size);
    }

    #[test]
    fn zero_rhs_gives_a_zero_step() {
        let Some(ctx) = visloc_gsplat_render::try_context() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let solver = GpuReducedSolver::new(ctx);
        let mut sys = random_system(40, 3);
        sys.rhs.iter_mut().for_each(|v| *v = Vector6::zeros());
        let system = ReducedCameraSystem {
            structure_id: 0,
            diag: &sys.diag,
            edges: &sys.edges,
            offdiag: &sys.offdiag,
            rhs: &sys.rhs,
            clusters: None,
            coarse: None,
            tol: 1e-2,
            max_iterations: 500,
        };
        let (steps, it) = solver.solve(&system).expect("GPU solve");
        assert_eq!(it, 0);
        assert!(steps.iter().all(|v| v.iter().all(|&c| c == 0.0)));
    }
}
