// VLFeat-compatible SIFT descriptors (f32 ports of visloc-vision's
// `gradient_layer`, `describe_raw_vlfeat` with `accumulate_vlfeat_sample`,
// `normalize_vlfeat_descriptor` and `transform_vlfeat_to_ubc`). The host
// selects each keypoint's level and window in f64 (`VlfeatPyramid::select`),
// hands them over in `DescKp`, and describes one (octave, level) group of
// keypoints per dispatch with that level's gradients bound.

struct GradParams {
    w: u32,
    h: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> G: GradParams;
@group(0) @binding(1) var<storage, read> gauss: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_out: array<vec2<f32>>;

fn g_at(x: i32, y: i32) -> f32 {
    let xi = u32(clamp(x, 0, i32(G.w) - 1));
    let yi = u32(clamp(y, 0, i32(G.h) - 1));
    return gauss[yi * G.w + xi];
}

// `gradient_layer`: VLFeat's stencil, central inside and one-sided on the
// border, for every pixel of one Gaussian level.
@compute @workgroup_size(16, 16)
fn vl_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= G.w || gid.y >= G.h) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let w = i32(G.w);
    let h = i32(G.h);
    var gx: f32;
    if (x == 0) {
        gx = g_at(1, y) - g_at(0, y);
    } else if (x == w - 1) {
        gx = g_at(x, y) - g_at(x - 1, y);
    } else {
        gx = 0.5 * (g_at(x + 1, y) - g_at(x - 1, y));
    }
    var gy: f32;
    if (y == 0) {
        gy = g_at(x, 1) - g_at(x, 0);
    } else if (y == h - 1) {
        gy = g_at(x, y) - g_at(x, y - 1);
    } else {
        gy = 0.5 * (g_at(x, y + 1) - g_at(x, y - 1));
    }
    grad_out[gid.y * G.w + gid.x] = vec2<f32>(gx, gy);
}

struct DescParams {
    // This dispatch's keypoints: kps[first .. first + num].
    first: u32,
    num: u32,
    // 0 = L2 (clip at 0.2, renormalize), 1 = L1-root.
    normalization: u32,
    pad: u32,
};

// One keypoint, prepared on the host.
struct DescKp {
    // The selected gradient level's width; `off` is unused.
    off: u32,
    lw: u32,
    xi: i32,
    yi: i32,
    min_dx: i32,
    max_dx: i32,
    min_dy: i32,
    max_dy: i32,
    // xi - local x, yi - local y (so a sample's offset is exact).
    off_x: f32,
    off_y: f32,
    sbp: f32,
    cos_o: f32,
    sin_o: f32,
    orientation: f32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> D: DescParams;
@group(0) @binding(1) var<storage, read> grad: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> kps: array<DescKp>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;

const TAU: f32 = 6.2831855;
const EPS: f32 = 2.220446e-16;
const WG: u32 = 256u;

fn rem_tau(a: f32) -> f32 {
    var r = a % TAU;
    if (r < 0.0) {
        r = r + TAU;
    }
    return r;
}

// `round` of a non-negative value, halves away from zero (Rust's `round`).
fn round_half_up(v: f32) -> f32 {
    let f = floor(v);
    return select(f, f + 1.0, v - f >= 0.5);
}

// One tile of samples: magnitude times the Gaussian weight (0: no vote),
// the position in cell units shifted so cell centres are 0..3, and the
// orientation bin in [0, 8).
var<workgroup> s_val: array<f32, 256>;
var<workgroup> s_cx: array<f32, 256>;
var<workgroup> s_cy: array<f32, 256>;
var<workgroup> s_ob: array<f32, 256>;
var<workgroup> red: array<f32, 2048>;
var<workgroup> desc: array<f32, 128>;
var<workgroup> bytes: array<u32, 128>;
var<workgroup> norm: f32;

// One workgroup per keypoint (`describe_raw_vlfeat`): the window's samples
// are evaluated a tile at a time, one per thread, then gathered by 16 cells
// x 16 lanes that keep their cell's eight orientation bins in registers.
// The trilinear votes are tents around the cell and bin centres, as in
// `accumulate_vlfeat_sample`. No atomics and a fixed summation order: the
// output is deterministic. Then `normalize_vlfeat_descriptor` (bytes) and
// COLMAP's UBC orientation order, packed four bytes a word.
@compute @workgroup_size(256)
fn vl_describe(@builtin(local_invocation_index) t: u32,
               @builtin(workgroup_id) wid: vec3<u32>) {
    let local = wid.x + wid.y * 65535u;
    if (local >= D.num) {
        return;
    }
    let i = D.first + local;
    let kp = kps[i];
    let nx_side = max(kp.max_dx - kp.min_dx + 1, 0);
    let ny_side = max(kp.max_dy - kp.min_dy + 1, 0);
    let total = u32(nx_side * ny_side);
    let my_cell = t / 16u;
    let lane = t % 16u;
    let ci = f32(my_cell % 4u);
    let ri = f32(my_cell / 4u);
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    let b0 = vec4<f32>(0.0, 1.0, 2.0, 3.0);
    let b1 = vec4<f32>(4.0, 5.0, 6.0, 7.0);
    for (var base = 0u; base < total; base = base + WG) {
        let si = base + t;
        var val = 0.0;
        var cx = -10.0;
        var cy = -10.0;
        var ob = 0.0;
        if (si < total) {
            let dxi = kp.min_dx + i32(si % u32(nx_side));
            let dyi = kp.min_dy + i32(si / u32(nx_side));
            let g = grad[u32(kp.yi + dyi) * kp.lw + u32(kp.xi + dxi)];
            let mag = sqrt(g.x * g.x + g.y * g.y);
            if (mag > EPS && mag <= 3.0e38) {
                let dx = f32(dxi) + kp.off_x;
                let dy = f32(dyi) + kp.off_y;
                let nx = (kp.cos_o * dx + kp.sin_o * dy) / kp.sbp;
                let ny = (-kp.sin_o * dx + kp.cos_o * dy) / kp.sbp;
                ob = rem_tau(atan2(g.y, g.x) - kp.orientation) * 8.0 / TAU;
                val = mag * exp(-(nx * nx + ny * ny) / 8.0);
                cx = nx + 1.5;
                cy = ny + 1.5;
            }
        }
        s_val[t] = val;
        s_cx[t] = cx;
        s_cy[t] = cy;
        s_ob[t] = ob;
        workgroupBarrier();
        for (var j = lane; j < WG; j = j + 16u) {
            let wxy = max(0.0, 1.0 - abs(s_cx[j] - ci)) * max(0.0, 1.0 - abs(s_cy[j] - ri));
            if (wxy > 0.0 && s_val[j] > 0.0) {
                let v = s_val[j] * wxy;
                let o = vec4<f32>(s_ob[j]);
                var d0 = abs(o - b0);
                var d1 = abs(o - b1);
                d0 = min(d0, vec4<f32>(8.0) - d0);
                d1 = min(d1, vec4<f32>(8.0) - d1);
                acc0 = acc0 + v * max(vec4<f32>(0.0), vec4<f32>(1.0) - d0);
                acc1 = acc1 + v * max(vec4<f32>(0.0), vec4<f32>(1.0) - d1);
            }
        }
        workgroupBarrier();
    }
    for (var b = 0u; b < 4u; b = b + 1u) {
        red[t * 8u + b] = acc0[b];
        red[t * 8u + 4u + b] = acc1[b];
    }
    workgroupBarrier();
    if (t < 128u) {
        // Bin t = cell * 8 + orientation, summed over the cell's lanes.
        let cell_id = t / 8u;
        let o = t % 8u;
        var sum = 0.0;
        for (var l = 0u; l < 16u; l = l + 1u) {
            sum = sum + red[(cell_id * 16u + l) * 8u + o];
        }
        desc[t] = sum;
    }
    workgroupBarrier();
    // `normalize_vlfeat_descriptor`.
    if (t == 0u) {
        var n = 0.0;
        for (var j = 0u; j < 128u; j = j + 1u) {
            let v = desc[j];
            n = select(n + v * v, n + abs(v), D.normalization == 1u);
        }
        norm = n;
    }
    workgroupBarrier();
    var v = desc[min(t, 127u)];
    if (D.normalization == 1u) {
        let l1 = norm;
        if (l1 > EPS && l1 <= 3.0e38) {
            v = sqrt(max(v / l1, 0.0));
        }
    } else {
        let l2 = sqrt(norm);
        if (l2 > EPS && l2 <= 3.0e38) {
            v = v / l2;
        }
        v = max(min(v, 0.2), 0.0);
    }
    workgroupBarrier();
    if (t < 128u) {
        desc[t] = v;
    }
    workgroupBarrier();
    if (D.normalization != 1u) {
        if (t == 0u) {
            var c2 = 0.0;
            for (var j = 0u; j < 128u; j = j + 1u) {
                c2 = c2 + desc[j] * desc[j];
            }
            norm = sqrt(c2);
        }
        workgroupBarrier();
        if (t < 128u && norm > EPS && norm <= 3.0e38) {
            desc[t] = desc[t] / norm;
        }
        workgroupBarrier();
    }
    // Bytes, then the VLFeat -> UBC orientation order within each cell
    // (orientation k goes to (8 - k) % 8), four bytes a word.
    if (t < 128u) {
        let q = u32(clamp(round_half_up(desc[t] * 512.0), 0.0, 255.0));
        bytes[(t & ~7u) + ((8u - (t & 7u)) & 7u)] = q;
    }
    workgroupBarrier();
    if (t < 32u) {
        let w = bytes[4u * t] | (bytes[4u * t + 1u] << 8u) | (bytes[4u * t + 2u] << 16u)
            | (bytes[4u * t + 3u] << 24u);
        out[i * 32u + t] = w;
    }
}
