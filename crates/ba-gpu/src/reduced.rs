//! GPU conjugate gradients for the COLMAP-port rig solver's reduced camera
//! system ([`visloc_slam::ReducedSolveAccelerator`]): the CPU solver's
//! preconditioned iteration (dense inverses over runs of consecutive frames,
//! the optional coarse level, the same stopping rule) in f32 on the GPU,
//! kernels in `shaders/reduced_pcg.wgsl`.
//!
//! The CPU still eliminates the points and builds the preconditioner; each
//! solve uploads the system, iterates on the device, and reads back the
//! camera step. Reductions sum in a fixed order, so a solve is deterministic
//! on a given device, but its f32 arithmetic differs from the CPU solve's
//! f64 vectors: the step matches it to about the PCG tolerance, not
//! bit-for-bit.

use std::sync::Mutex;

use nalgebra::{Matrix6, Vector6};
use rayon::prelude::*;
use visloc_gsplat_render::GpuContext;
use visloc_slam::{ReducedCameraSystem, ReducedSolveAccelerator};

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

/// Binding 0 is the uniform, 1-9 read-only storage, 10-18 read-write.
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
const B_SCAL: usize = 18;
/// `scal` slots read back by the host (see the shader).
const S_DONE: usize = 6;
const S_ITERS: usize = 7;
const S_FAILED: usize = 8;
const N_SCAL: usize = 16;
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

/// Device buffers, reused across solves and grown on demand.
struct Buffers {
    params: wgpu::Buffer,
    storage: Vec<wgpu::Buffer>,
    caps: Vec<u64>,
}

/// The reduced-camera PCG on the GPU; register it with
/// [`visloc_slam::set_reduced_solve_accelerator`].
pub struct GpuReducedSolver {
    ctx: GpuContext,
    layout: wgpu::BindGroupLayout,
    pipes: Vec<wgpu::ComputePipeline>,
    buffers: Mutex<Option<Buffers>>,
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

impl GpuReducedSolver {
    pub fn new(ctx: GpuContext) -> Self {
        let dev = &ctx.device;
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reduced-pcg"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/reduced_pcg.wgsl").into()),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..NUM_BINDINGS as u32)
            .map(|b| wgpu::BindGroupLayoutEntry {
                binding: b,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if b == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: b <= B_RHS as u32,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reduced-pcg"),
            entries: &entries,
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reduced-pcg"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipes = ENTRIES
            .iter()
            .map(|e| {
                dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(e),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: Some(e),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        let chunk = std::env::var("VISLOC_GPU_PCG_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&c: &u32| c > 0)
            .unwrap_or(32);
        Self {
            ctx,
            layout,
            pipes,
            buffers: Mutex::new(None),
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

    /// Storage buffers with at least `sizes[b]` bytes each (reused when
    /// large enough).
    fn ensure_buffers<'a>(&self, slot: &'a mut Option<Buffers>, sizes: &[u64]) -> &'a Buffers {
        let dev = &self.ctx.device;
        let limit = self.buffer_limit();
        let fits = slot
            .as_ref()
            .is_some_and(|b| b.caps.iter().zip(sizes).all(|(c, s)| c >= s));
        if !fits {
            let old = slot.take();
            let caps: Vec<u64> = sizes
                .iter()
                .enumerate()
                .map(|(i, &s)| {
                    // Grow geometrically to avoid reallocating every solve.
                    let prev = old.as_ref().map_or(0, |b| b.caps[i]);
                    if s <= prev {
                        prev
                    } else {
                        ((s + s / 4).min(limit / 4 * 4).max(s).max(16)).div_ceil(4) * 4
                    }
                })
                .collect();
            let storage = caps
                .iter()
                .enumerate()
                .map(|(i, &c)| {
                    dev.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("reduced-pcg-{i}")),
                        size: c,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    })
                })
                .collect();
            let params = dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("reduced-pcg-params"),
                size: std::mem::size_of::<Params>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            *slot = Some(Buffers {
                params,
                storage,
                caps,
            });
        }
        slot.as_ref().expect("buffers")
    }

    fn read_f32(&self, buffer: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let dev = &self.ctx.device;
        let bytes = (len * 4) as u64;
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reduced-pcg-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("reduced-pcg-read"),
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

    fn solve_system(&self, sys: &ReducedCameraSystem<'_>) -> Option<(Vec<Vector6<f64>>, usize)> {
        let n = sys.diag.len();
        let n_edges = sys.edges.len();
        if n == 0
            || sys.rhs.len() != n
            || sys.offdiag.len() != n_edges * 36
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

        // Both block rows of every edge: row b reads the block as stored,
        // row a its transpose (bit 31), each in edge order.
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
        let mut sizes = vec![0u64; NUM_BINDINGS];
        sizes[B_BLOCKS] = f(sys.offdiag.len());
        sizes[B_DIAG] = f(diag.len());
        sizes[B_ROW_PTR] = f(row_ptr.len());
        sizes[B_ROW_ENT] = f(row_ent.len());
        sizes[B_ROW_COL] = f(row_col.len());
        sizes[B_CL_MAT] = f(cl_mat.len());
        sizes[B_CL_OFF] = f(cl_off.len());
        sizes[B_COARSE] = f(coarse_inv.len());
        sizes[B_RHS] = f(n6);
        for s in &mut sizes[B_X..B_X + 5] {
            *s = f(n6);
        }
        sizes[B_PARTIALS] = f(n.max(n_wg));
        sizes[16] = f((6 * coarse_m).max(1));
        sizes[17] = f((6 * coarse_m).max(1));
        sizes[B_SCAL] = f(N_SCAL);
        // Binding 0 is the uniform; give its storage slot a token size.
        sizes[0] = 16;
        let limit = self.buffer_limit();
        if sizes.iter().any(|&s| s > limit) {
            return None;
        }

        let mut guard = self.buffers.lock().ok()?;
        let bufs = self.ensure_buffers(&mut guard, &sizes);
        let queue = &self.ctx.queue;
        queue.write_buffer(&bufs.params, 0, bytemuck::bytes_of(&params));
        let uploads: [(usize, &[u8]); 9] = [
            (B_BLOCKS, bytemuck::cast_slice(sys.offdiag)),
            (B_DIAG, bytemuck::cast_slice(&diag)),
            (B_ROW_PTR, bytemuck::cast_slice(&row_ptr)),
            (B_ROW_ENT, bytemuck::cast_slice(&row_ent)),
            (B_ROW_COL, bytemuck::cast_slice(&row_col)),
            (B_CL_MAT, bytemuck::cast_slice(&cl_mat)),
            (B_CL_OFF, bytemuck::cast_slice(&cl_off)),
            (B_COARSE, bytemuck::cast_slice(&coarse_inv)),
            (B_RHS, bytemuck::cast_slice(&rhs)),
        ];
        for (b, bytes) in uploads {
            if !bytes.is_empty() {
                queue.write_buffer(&bufs.storage[b], 0, bytes);
            }
        }
        let dev = &self.ctx.device;
        let entries: Vec<wgpu::BindGroupEntry> = (0..NUM_BINDINGS)
            .map(|b| wgpu::BindGroupEntry {
                binding: b as u32,
                resource: if b == 0 {
                    bufs.params.as_entire_binding()
                } else {
                    bufs.storage[b].as_entire_binding()
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
            let scal = self.read_f32(&bufs.storage[B_SCAL], N_SCAL);
            if scal[S_DONE] != 0.0 || issued >= max {
                break;
            }
            let more = self.chunk.min(max - issued);
            submit(false, more);
            issued += more;
        }
        let scal = self.read_f32(&bufs.storage[B_SCAL], N_SCAL);
        if scal[S_FAILED] != 0.0 {
            return None;
        }
        let x = self.read_f32(&bufs.storage[B_X], n6);
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
