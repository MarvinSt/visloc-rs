// Preconditioned conjugate gradients on the COLMAP-port rig solver's reduced
// camera system (`reduced.rs`): the CPU solver's iteration (`pcg_core` in
// pipelines/slam/src/colmap_incremental/rig_ba_solver.rs) in f32.
//
// The matrix is n 6x6 diagonal blocks plus one stored block B per edge
// (a, b), the (row b, column a) block, with the edges sorted by a. The
// product reads every block once: `matvec_out` takes the edges of one node a
// per workgroup, a contiguous run of blocks, adds B^T p_b into row a and
// writes u = B p_a per edge; `matvec_in` gathers the u of the edges ending at
// each node b. The preconditioner applies dense inverses over runs of
// consecutive nodes (a run of one node is block-Jacobi), plus an optional
// additive coarse level (one shared 6-vector per aggregate of consecutive
// nodes).
//
// Every reduction sums in a fixed order (per-row or per-workgroup partials,
// then one workgroup), so a solve is deterministic on a given device. Once
// `scal[DONE]` is set every kernel returns at entry, so iterations dispatched
// past convergence change nothing.

struct Params {
    n: u32,              // nodes (6 unknowns each)
    n_wg: u32,           // workgroups of the 256-wide element kernels
    n_clusters: u32,
    cluster_k: u32,      // nodes per run (the last run may be shorter)
    coarse_m: u32,       // aggregates; 0 turns the coarse level off
    coarse_group: u32,   // nodes per aggregate
    coarse_nodes: u32,   // leading nodes the coarse level covers
    max_iter: u32,
    tol2: f32,           // stop once |r|^2 <= tol2 |b|^2
    rows_x: u32,         // x extent of the row-kernel dispatch grid
    _pad0: u32,
    _pad1: u32,
}

const RZ: u32 = 0u;
const PAP: u32 = 1u;
const ALPHA: u32 = 2u;
const BETA: u32 = 3u;
const RR: u32 = 4u;
const BNORM2: u32 = 5u;
const DONE: u32 = 6u;
const ITERS: u32 = 7u;
const FAILED: u32 = 8u;

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> blocks: array<f32>;     // 36 per edge, column-major
@group(0) @binding(2) var<storage, read> diag: array<f32>;       // 36 per node
@group(0) @binding(3) var<storage, read> out_off: array<u32>;    // n + 1: node a's edges
@group(0) @binding(4) var<storage, read> edge_b: array<u32>;     // b of each edge
@group(0) @binding(5) var<storage, read> in_off: array<u32>;     // n + 1: into in_edge
@group(0) @binding(6) var<storage, read> in_edge: array<u32>;    // edges ending at each node, ascending
@group(0) @binding(7) var<storage, read> cl_mat: array<f32>;     // run inverses, column-major
@group(0) @binding(8) var<storage, read> cl_off: array<u32>;     // n_clusters + 1
@group(0) @binding(9) var<storage, read> coarse_inv: array<f32>; // (6 coarse_m)^2, row-major
@group(0) @binding(10) var<storage, read> rhs: array<f32>;
@group(0) @binding(11) var<storage, read_write> x: array<f32>;
@group(0) @binding(12) var<storage, read_write> r: array<f32>;
@group(0) @binding(13) var<storage, read_write> z: array<f32>;
@group(0) @binding(14) var<storage, read_write> p: array<f32>;
@group(0) @binding(15) var<storage, read_write> ap: array<f32>;
@group(0) @binding(16) var<storage, read_write> u: array<f32>;   // B p_a, 6 per edge
@group(0) @binding(17) var<storage, read_write> partials: array<f32>;
@group(0) @binding(18) var<storage, read_write> rc: array<f32>;
@group(0) @binding(19) var<storage, read_write> yc: array<f32>;
@group(0) @binding(20) var<storage, read_write> scal: array<f32>;

var<workgroup> red: array<f32, 256>;
var<workgroup> red192: array<f32, 1152>;
var<workgroup> seg: array<f32, 256>;
var<workgroup> flag: f32;

// `scal[DONE]` as a workgroup-uniform value (all invocations must call it).
fn finished(lid: u32) -> bool {
    if (lid == 0u) {
        flag = scal[DONE];
    }
    return workgroupUniformLoad(&flag) != 0.0;
}

// Sum of `v` over the 256 invocations, in a fixed order; valid in every
// invocation.
fn block_sum(lid: u32, v: f32) -> f32 {
    red[lid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            red[lid] = red[lid] + red[lid + s];
        }
        workgroupBarrier();
    }
    let total = red[0];
    workgroupBarrier();
    return total;
}

// One workgroup sums partials[0..count) in a fixed order.
fn sum_partials(lid: u32, count: u32) -> f32 {
    var acc = 0.0;
    for (var i = lid; i < count; i = i + 256u) {
        acc = acc + partials[i];
    }
    return block_sum(lid, acc);
}

// x = 0, r = b, and per-workgroup partials of |b|^2.
@compute @workgroup_size(256)
fn init(@builtin(local_invocation_index) lid: u32,
        @builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wg: vec3<u32>) {
    let i = gid.x;
    var v = 0.0;
    if (i < 6u * P.n) {
        x[i] = 0.0;
        let b = rhs[i];
        r[i] = b;
        v = b * b;
    }
    let s = block_sum(lid, v);
    if (lid == 0u) {
        partials[wg.x] = s;
    }
}

@compute @workgroup_size(256)
fn init_b(@builtin(local_invocation_index) lid: u32) {
    let b2 = sum_partials(lid, P.n_wg);
    if (lid == 0u) {
        scal[BNORM2] = b2;
        scal[ITERS] = 0.0;
        scal[FAILED] = 0.0;
        scal[DONE] = select(0.0, 1.0, b2 == 0.0);
        scal[RZ] = 0.0;
        scal[ALPHA] = 0.0;
        scal[BETA] = 0.0;
    }
}

// z = M^-1 r over one run of consecutive nodes per workgroup.
@compute @workgroup_size(256)
fn precond_cluster(@builtin(local_invocation_index) lid: u32,
                   @builtin(workgroup_id) wg: vec3<u32>) {
    if (finished(lid)) {
        return;
    }
    let c = wg.x + wg.y * P.rows_x;
    if (c >= P.n_clusters) {
        return;
    }
    let first = c * P.cluster_k;
    let rows = 6u * (min(first + P.cluster_k, P.n) - first);
    let base = 6u * first;
    let off = cl_off[c];
    // Rows in chunks of 256; the run's residual is staged through `seg`.
    for (var r0 = 0u; r0 < rows; r0 = r0 + 256u) {
        let row = r0 + lid;
        var acc = 0.0;
        for (var c0 = 0u; c0 < rows; c0 = c0 + 256u) {
            workgroupBarrier();
            if (c0 + lid < rows) {
                seg[lid] = r[base + c0 + lid];
            }
            workgroupBarrier();
            let cn = min(256u, rows - c0);
            if (row < rows) {
                for (var k = 0u; k < cn; k = k + 1u) {
                    acc = acc + cl_mat[off + (c0 + k) * rows + row] * seg[k];
                }
            }
        }
        if (row < rows) {
            z[base + row] = acc;
        }
    }
}

// rc = P^T r: per aggregate and component, the sum over its nodes.
@compute @workgroup_size(256)
fn coarse_restrict(@builtin(local_invocation_index) lid: u32,
                   @builtin(global_invocation_id) gid: vec3<u32>) {
    if (finished(lid)) {
        return;
    }
    let j = gid.x;
    if (j >= 6u * P.coarse_m) {
        return;
    }
    let g = j / 6u;
    let k = j % 6u;
    let end = min((g + 1u) * P.coarse_group, P.coarse_nodes);
    var acc = 0.0;
    for (var i = g * P.coarse_group; i < end; i = i + 1u) {
        acc = acc + r[6u * i + k];
    }
    rc[j] = acc;
}

// yc = (P^T A P)^-1 rc, one row per workgroup (a contiguous read).
@compute @workgroup_size(256)
fn coarse_apply(@builtin(local_invocation_index) lid: u32,
                @builtin(workgroup_id) wg: vec3<u32>) {
    if (finished(lid)) {
        return;
    }
    let m = 6u * P.coarse_m;
    let t = wg.x;
    if (t >= m) {
        return;
    }
    var acc = 0.0;
    for (var c = lid; c < m; c = c + 256u) {
        acc = acc + coarse_inv[t * m + c] * rc[c];
    }
    let s = block_sum(lid, acc);
    if (lid == 0u) {
        yc[t] = s;
    }
}

// z += P yc (when the coarse level is on), and per-workgroup partials of r.z.
@compute @workgroup_size(256)
fn precond_finish(@builtin(local_invocation_index) lid: u32,
                  @builtin(global_invocation_id) gid: vec3<u32>,
                  @builtin(workgroup_id) wg: vec3<u32>) {
    if (finished(lid)) {
        return;
    }
    let i = gid.x;
    var v = 0.0;
    if (i < 6u * P.n) {
        var zi = z[i];
        let node = i / 6u;
        if (P.coarse_m > 0u && node < P.coarse_nodes) {
            zi = zi + yc[6u * (node / P.coarse_group) + i % 6u];
            z[i] = zi;
        }
        v = r[i] * zi;
    }
    let s = block_sum(lid, v);
    if (lid == 0u) {
        partials[wg.x] = s;
    }
}

// rz = r.z of the initial residual, p = z.
@compute @workgroup_size(256)
fn init_rz(@builtin(local_invocation_index) lid: u32) {
    if (finished(lid)) {
        return;
    }
    let rz = sum_partials(lid, P.n_wg);
    if (lid == 0u) {
        scal[RZ] = rz;
    }
}

@compute @workgroup_size(256)
fn init_p(@builtin(local_invocation_index) lid: u32,
          @builtin(global_invocation_id) gid: vec3<u32>) {
    if (finished(lid)) {
        return;
    }
    let i = gid.x;
    if (i < 6u * P.n) {
        p[i] = z[i];
    }
}

// The product's first half, one node a per workgroup over its edges (a, b),
// whose blocks are contiguous: ap_a = sum of B^T p_b, and u_e = B p_a for
// `matvec_in`. Invocation 6 j + k takes component k of the edges j, j + 32,
// ... of the node, so consecutive invocations read consecutive floats.
@compute @workgroup_size(192)
fn matvec_out(@builtin(local_invocation_index) lid: u32,
              @builtin(workgroup_id) wg: vec3<u32>) {
    if (finished(lid)) {
        return;
    }
    let a = wg.x + wg.y * P.rows_x;
    if (a >= P.n) {
        return;
    }
    let k = lid % 6u;
    let j = lid / 6u;
    var pa: array<f32, 6>;
    for (var c = 0u; c < 6u; c = c + 1u) {
        pa[c] = p[6u * a + c];
    }
    var acc = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let end = out_off[a + 1u];
    for (var e = out_off[a] + j; e < end; e = e + 32u) {
        let base = 36u * e;
        let pb = p[6u * edge_b[e] + k];
        var uk = 0.0;
        for (var c = 0u; c < 6u; c = c + 1u) {
            // B[k][c]: row k (node b's component), column c (node a's).
            let v = blocks[base + 6u * c + k];
            uk = uk + v * pa[c];
            acc[c] = acc[c] + v * pb;
        }
        u[6u * e + k] = uk;
    }
    // Sum the 192 invocations' 6-vectors in a fixed order.
    for (var c = 0u; c < 6u; c = c + 1u) {
        red192[lid * 6u + c] = acc[c];
    }
    workgroupBarrier();
    if (lid < 64u) {
        for (var c = 0u; c < 6u; c = c + 1u) {
            red192[lid * 6u + c] = red192[lid * 6u + c] + red192[(lid + 128u) * 6u + c];
        }
    }
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            for (var c = 0u; c < 6u; c = c + 1u) {
                red192[lid * 6u + c] = red192[lid * 6u + c] + red192[(lid + s) * 6u + c];
            }
        }
        workgroupBarrier();
    }
    if (lid < 6u) {
        ap[6u * a + lid] = red192[lid];
    }
}

// The second half, one node b per workgroup: ap_b = D_b p_b + (the first
// half's B^T terms) + the u of the edges ending at b, and partials[b] =
// p_b . ap_b. Invocation 6 j + k sums component k of the edges j, j + 32,
// ... of the node's list.
@compute @workgroup_size(192)
fn matvec_in(@builtin(local_invocation_index) lid: u32,
             @builtin(workgroup_id) wg: vec3<u32>) {
    if (finished(lid)) {
        return;
    }
    let b = wg.x + wg.y * P.rows_x;
    if (b >= P.n) {
        return;
    }
    let k = lid % 6u;
    let j = lid / 6u;
    var acc = 0.0;
    let end = in_off[b + 1u];
    for (var i = in_off[b] + j; i < end; i = i + 32u) {
        acc = acc + u[6u * in_edge[i] + k];
    }
    red192[lid] = acc;
    workgroupBarrier();
    for (var s = 16u; s > 0u; s = s >> 1u) {
        if (j < s) {
            red192[lid] = red192[lid] + red192[lid + 6u * s];
        }
        workgroupBarrier();
    }
    var y = 0.0;
    if (lid < 6u) {
        let d = b * 36u;
        for (var c = 0u; c < 6u; c = c + 1u) {
            y = y + diag[d + c * 6u + lid] * p[6u * b + c];
        }
        y = y + ap[6u * b + lid] + red192[lid];
    }
    workgroupBarrier();
    if (lid < 6u) {
        ap[6u * b + lid] = y;
        red192[lid] = p[6u * b + lid] * y;
    }
    workgroupBarrier();
    if (lid == 0u) {
        partials[b] = red192[0] + red192[1] + red192[2] + red192[3] + red192[4] + red192[5];
    }
}

// alpha = rz / p.Ap; a non-positive or non-finite curvature fails the solve.
@compute @workgroup_size(256)
fn alpha(@builtin(local_invocation_index) lid: u32) {
    if (finished(lid)) {
        return;
    }
    let pap = sum_partials(lid, P.n);
    if (lid == 0u) {
        scal[PAP] = pap;
        // `pap > 0` is false for NaN; the upper bound rejects infinity.
        if (pap > 0.0 && pap <= 3.0e38) {
            scal[ALPHA] = scal[RZ] / pap;
        } else {
            scal[FAILED] = 1.0;
            scal[DONE] = 1.0;
        }
    }
}

// x += alpha p, r -= alpha Ap, and per-workgroup partials of |r|^2.
@compute @workgroup_size(256)
fn update_xr(@builtin(local_invocation_index) lid: u32,
             @builtin(global_invocation_id) gid: vec3<u32>,
             @builtin(workgroup_id) wg: vec3<u32>) {
    if (finished(lid)) {
        return;
    }
    let i = gid.x;
    var v = 0.0;
    if (i < 6u * P.n) {
        let a = scal[ALPHA];
        x[i] = x[i] + a * p[i];
        let ri = r[i] - a * ap[i];
        r[i] = ri;
        v = ri * ri;
    }
    let s = block_sum(lid, v);
    if (lid == 0u) {
        partials[wg.x] = s;
    }
}

// Count the iteration; stop at the tolerance or the iteration cap.
@compute @workgroup_size(256)
fn check(@builtin(local_invocation_index) lid: u32) {
    if (finished(lid)) {
        return;
    }
    let rr = sum_partials(lid, P.n_wg);
    if (lid == 0u) {
        scal[RR] = rr;
        let it = scal[ITERS] + 1.0;
        scal[ITERS] = it;
        if (rr <= P.tol2 * scal[BNORM2] || it >= f32(P.max_iter)) {
            scal[DONE] = 1.0;
        }
    }
}

// beta = r.z / previous r.z.
@compute @workgroup_size(256)
fn beta(@builtin(local_invocation_index) lid: u32) {
    if (finished(lid)) {
        return;
    }
    let rz = sum_partials(lid, P.n_wg);
    if (lid == 0u) {
        scal[BETA] = rz / scal[RZ];
        scal[RZ] = rz;
    }
}

// p = z + beta p.
@compute @workgroup_size(256)
fn update_p(@builtin(local_invocation_index) lid: u32,
            @builtin(global_invocation_id) gid: vec3<u32>) {
    if (finished(lid)) {
        return;
    }
    let i = gid.x;
    if (i < 6u * P.n) {
        p[i] = z[i] + scal[BETA] * p[i];
    }
}
