// Point elimination for the COLMAP-port rig solver's reduced camera system
// (`reduced.rs`): the off-diagonal blocks
//
//     A_ts -= sum over points p seen by both s < t of  W_tp V_p^-1 W_sp^T
//
// (W the 6x3 camera-point blocks, V_p^-1 the damped point inverse), which
// the CPU forms in `eliminate_by_output`. The diagonal blocks and the
// right-hand side stay on the CPU, in f64.
//
// One workgroup per task: a run of at most CHUNK consecutive edges of one
// row s (edges are sorted by (s, t)). Each invocation takes points observing
// s and adds the pairs that land in the task's edges. The sums are exact:
// every term is divided by the Cauchy-Schwarz bound sqrt(D_t[r][r] D_s[c][c])
// (D the per-node elimination diagonal, `scale` its square root), which
// bounds the whole sum by one, and accumulated as a 64-bit fixed-point
// integer with 2^61 units, carried over two 32-bit workgroup atomics.
// Integer addition is associative, so the result does not depend on the
// order the terms arrive in.

override CHUNK: u32 = 160u;

struct ElimParams {
    n_tasks: u32,
    tasks_x: u32,
    n_extra: u32,       // direct couplings to add (`add_extra`)
    n_run_edges: u32,   // edges inside preconditioner runs (`gather_runs`)
    coarse_m: u32,      // aggregates of `coarse_sums` (0: none)
    coarse_group: u32,  // nodes per aggregate
    coarse_nodes: u32,  // leading nodes the aggregates cover
    _pad: u32,
}

@group(0) @binding(0) var<uniform> E: ElimParams;
@group(0) @binding(1) var<storage, read> tasks: array<vec4<u32>>;   // (s, first edge, end edge, -)
@group(0) @binding(2) var<storage, read> edge_b: array<u32>;        // t of each edge
@group(0) @binding(3) var<storage, read> slot_off: array<u32>;      // per node, into `slots`
@group(0) @binding(4) var<storage, read> slots: array<vec2<u32>>;   // (point, k), ascending point
@group(0) @binding(5) var<storage, read> point_off: array<u32>;     // per point, into hcp / point_nodes
@group(0) @binding(6) var<storage, read> point_nodes: array<u32>;   // node of (point, k), ascending in k
@group(0) @binding(7) var<storage, read> hcp: array<f32>;           // 18 per (point, k), 6x3 column-major
@group(0) @binding(8) var<storage, read> vinv: array<f32>;          // 9 per point, 3x3 column-major
@group(0) @binding(9) var<storage, read> scale: array<f32>;         // 6 per node
@group(0) @binding(10) var<storage, read_write> blocks: array<f32>; // 36 per edge, column-major
@group(0) @binding(11) var<storage, read> extra_edge: array<u32>;   // edge of each direct coupling
@group(0) @binding(12) var<storage, read> extra_blk: array<f32>;    // 36 per direct coupling
@group(0) @binding(13) var<storage, read> run_edges: array<u32>;    // edges inside preconditioner runs
@group(0) @binding(14) var<storage, read_write> run_out: array<f32>; // 36 per run edge
@group(0) @binding(15) var<storage, read> row_off: array<u32>;      // per node, into the edges
@group(0) @binding(16) var<storage, read_write> coarse_out: array<f32>; // 36 per (G, H), G <= H

// Per task edge: 36 low words, then 36 high words.
var<workgroup> acc: array<atomic<u32>, CHUNK * 72u>;

// Add w * 2^61 (|w| <= 2 keeps the total in range) to the 64-bit counter
// whose low word is acc[i] and high word acc[i + 36].
fn add_fixed(i: u32, w: f32) {
    let x = w * 536870912.0; // 2^29: the high word's units
    let h = floor(x);
    let lo = u32((x - h) * 4294967296.0);
    let old = atomicAdd(&acc[i], lo);
    let carry = select(0u, 1u, old + lo < old);
    atomicAdd(&acc[i + 36u], bitcast<u32>(i32(h)) + carry);
}

@compute @workgroup_size(256)
fn eliminate(@builtin(local_invocation_index) lid: u32,
             @builtin(workgroup_id) wg: vec3<u32>) {
    let task_id = wg.x + wg.y * E.tasks_x;
    if (task_id >= E.n_tasks) {
        return;
    }
    let task = tasks[task_id];
    let s = task.x;
    let lo = task.y;
    let n_e = task.z - task.y;
    for (var i = lid; i < n_e * 72u; i = i + 256u) {
        atomicStore(&acc[i], 0u);
    }
    workgroupBarrier();
    let t_first = edge_b[lo];
    let t_last = edge_b[lo + n_e - 1u];
    for (var si = slot_off[s] + lid; si < slot_off[s + 1u]; si = si + 256u) {
        let sl = slots[si];
        let p = sl.x;
        let k = sl.y;
        let po = point_off[p];
        let pn = point_off[p + 1u] - po;
        // This point's partners j > k inside the task's edges: from the
        // first node >= t_first (binary search), while node <= t_last.
        var jlo = k + 1u;
        var jhi = pn;
        while (jlo < jhi) {
            let mid = (jlo + jhi) / 2u;
            if (point_nodes[po + mid] < t_first) {
                jlo = mid + 1u;
            } else {
                jhi = mid;
            }
        }
        if (jlo >= pn || point_nodes[po + jlo] > t_last) {
            continue;
        }
        // sk = W_sp V_p^-1 (6x3); a zero inverse marks a skipped point.
        var vi: array<f32, 9>;
        var nz = false;
        for (var q = 0u; q < 9u; q = q + 1u) {
            vi[q] = vinv[9u * p + q];
            nz = nz || vi[q] != 0.0;
        }
        if (!nz) {
            continue;
        }
        let hk = 18u * (po + k);
        var sk: array<f32, 18>; // column-major 6x3
        for (var i = 0u; i < 3u; i = i + 1u) {
            for (var r = 0u; r < 6u; r = r + 1u) {
                sk[i * 6u + r] = hcp[hk + r] * vi[i * 3u]
                    + hcp[hk + 6u + r] * vi[i * 3u + 1u]
                    + hcp[hk + 12u + r] * vi[i * 3u + 2u];
            }
        }
        var sc_s: array<f32, 6>;
        for (var c = 0u; c < 6u; c = c + 1u) {
            sc_s[c] = scale[6u * s + c];
        }
        for (var j = jlo; j < pn; j = j + 1u) {
            let t = point_nodes[po + j];
            if (t > t_last) {
                break;
            }
            var a = lo;
            var b = lo + n_e;
            while (a < b) {
                let m = (a + b) / 2u;
                if (edge_b[m] < t) {
                    a = m + 1u;
                } else {
                    b = m;
                }
            }
            let base = (a - lo) * 72u;
            let hj = 18u * (po + j);
            for (var r = 0u; r < 6u; r = r + 1u) {
                let h0 = hcp[hj + r];
                let h1 = hcp[hj + 6u + r];
                let h2 = hcp[hj + 12u + r];
                let st = scale[6u * t + r];
                for (var c = 0u; c < 6u; c = c + 1u) {
                    let v = h0 * sk[c] + h1 * sk[6u + c] + h2 * sk[12u + c];
                    add_fixed(base + c * 6u + r, -v / (st * sc_s[c]));
                }
            }
        }
    }
    workgroupBarrier();
    for (var i = lid; i < n_e * 36u; i = i + 256u) {
        let el = i / 36u;
        let ent = i % 36u;
        let c = ent / 6u;
        let r = ent % 6u;
        let low = atomicLoad(&acc[el * 72u + ent]);
        let high = bitcast<i32>(atomicLoad(&acc[el * 72u + 36u + ent]));
        let w = f32(high) * 1.862645149230957e-9 + f32(low) * 4.336808689942018e-19;
        let t = edge_b[lo + el];
        blocks[(lo + el) * 36u + ent] = w * scale[6u * t + r] * scale[6u * s + c];
    }
}

// blocks[e] += the direct coupling of an extra node (rig sensor, shared
// camera), which the CPU computes during linearization.
@compute @workgroup_size(64)
fn add_extra(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= E.n_extra * 36u) {
        return;
    }
    let e = extra_edge[i / 36u];
    blocks[e * 36u + i % 36u] = blocks[e * 36u + i % 36u] + extra_blk[i];
}

// Copies the blocks inside preconditioner runs out for the CPU, which
// factors the runs.
@compute @workgroup_size(64)
fn gather_runs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= E.n_run_edges * 36u) {
        return;
    }
    run_out[i] = blocks[run_edges[i / 36u] * 36u + i % 36u];
}

// coarse_out[G * m + H] = sum of the blocks of edges (a, b), a in aggregate
// G, b in aggregate H (G <= H), both below `coarse_nodes`: the coarse
// level's off-diagonal part, which the CPU completes and factors. One
// workgroup per G, one invocation per H; a row's edges to H are contiguous
// (edges are sorted), found by binary search, and summed in edge order.
@compute @workgroup_size(256)
fn coarse_sums(@builtin(local_invocation_index) lid: u32,
               @builtin(workgroup_id) wg: vec3<u32>) {
    let g = wg.x;
    let m = E.coarse_m;
    let grp = E.coarse_group;
    let nodes = E.coarse_nodes;
    for (var h = g + lid; h < m; h = h + 256u) {
        var sum: array<f32, 36>;
        for (var q = 0u; q < 36u; q = q + 1u) {
            sum[q] = 0.0;
        }
        let t0 = h * grp;
        let t1 = min((h + 1u) * grp, nodes);
        for (var a = g * grp; a < min((g + 1u) * grp, nodes); a = a + 1u) {
            var lo = row_off[a];
            var hi = row_off[a + 1u];
            while (lo < hi) {
                let mid = (lo + hi) / 2u;
                if (edge_b[mid] < t0) {
                    lo = mid + 1u;
                } else {
                    hi = mid;
                }
            }
            for (var e = lo; e < row_off[a + 1u] && edge_b[e] < t1; e = e + 1u) {
                for (var q = 0u; q < 36u; q = q + 1u) {
                    sum[q] = sum[q] + blocks[e * 36u + q];
                }
            }
        }
        let o = (g * m + h) * 36u;
        for (var q = 0u; q < 36u; q = q + 1u) {
            coarse_out[o + q] = sum[q];
        }
    }
}
