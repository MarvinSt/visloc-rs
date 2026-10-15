// VLFeat-compatible SIFT descriptors (f32 ports of visloc-vision's
// `gradient_layer`, `describe_raw_vlfeat` with `accumulate_vlfeat_sample`,
// `normalize_vlfeat_descriptor` and `transform_vlfeat_to_ubc`). The host
// selects each keypoint's level and window in f64 (`VlfeatPyramid::select`)
// and hands them over in `DescKp`.

struct GradParams {
    w: u32,
    h: u32,
    // Gaussian levels of the octave (gradients are written for each).
    levels: u32,
    // Element offset of the octave's first gradient.
    dst_off: u32,
};

@group(0) @binding(0) var<uniform> G: GradParams;
@group(0) @binding(1) var<storage, read> gauss: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_out: array<vec2<f32>>;

fn g_at(level: u32, x: i32, y: i32) -> f32 {
    let xi = u32(clamp(x, 0, i32(G.w) - 1));
    let yi = u32(clamp(y, 0, i32(G.h) - 1));
    return gauss[level * G.w * G.h + yi * G.w + xi];
}

// `gradient_layer`: VLFeat's stencil, central inside and one-sided on the
// border, for every pixel of every level of one octave.
@compute @workgroup_size(16, 16)
fn vl_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= G.w || gid.y >= G.h || gid.z >= G.levels) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let l = gid.z;
    let w = i32(G.w);
    let h = i32(G.h);
    var gx: f32;
    if (x == 0) {
        gx = g_at(l, 1, y) - g_at(l, 0, y);
    } else if (x == w - 1) {
        gx = g_at(l, x, y) - g_at(l, x - 1, y);
    } else {
        gx = 0.5 * (g_at(l, x + 1, y) - g_at(l, x - 1, y));
    }
    var gy: f32;
    if (y == 0) {
        gy = g_at(l, x, 1) - g_at(l, x, 0);
    } else if (y == h - 1) {
        gy = g_at(l, x, y) - g_at(l, x, y - 1);
    } else {
        gy = 0.5 * (g_at(l, x, y + 1) - g_at(l, x, y - 1));
    }
    grad_out[G.dst_off + (l * G.h + gid.y) * G.w + gid.x] = vec2<f32>(gx, gy);
}

struct DescParams {
    num: u32,
    // 0 = L2 (clip at 0.2, renormalize), 1 = L1-root.
    normalization: u32,
    pad0: u32,
    pad1: u32,
};

// One keypoint, prepared on the host.
struct DescKp {
    // Element offset of the selected gradient level, and its width.
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

// One thread per keypoint: the 4 x 4 x 8 histogram, its normalization, the
// byte quantization and the UBC bin order, packed four bytes a word.
@compute @workgroup_size(64)
fn vl_describe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 65535u * 64u;
    if (i >= D.num) {
        return;
    }
    let kp = kps[i];
    var d: array<f32, 128>;
    for (var j = 0u; j < 128u; j = j + 1u) {
        d[j] = 0.0;
    }
    for (var dyi = kp.min_dy; dyi <= kp.max_dy; dyi = dyi + 1) {
        for (var dxi = kp.min_dx; dxi <= kp.max_dx; dxi = dxi + 1) {
            let sx = kp.xi + dxi;
            let sy = kp.yi + dyi;
            let g = grad[kp.off + u32(sy) * kp.lw + u32(sx)];
            let mag = sqrt(g.x * g.x + g.y * g.y);
            if (!(mag > EPS) || mag > 3.0e38) {
                continue;
            }
            let dx = f32(dxi) + kp.off_x;
            let dy = f32(dyi) + kp.off_y;
            let nx = (kp.cos_o * dx + kp.sin_o * dy) / kp.sbp;
            let ny = (-kp.sin_o * dx + kp.cos_o * dy) / kp.sbp;
            let theta = rem_tau(atan2(g.y, g.x) - kp.orientation);
            let obin = theta * 8.0 / TAU;
            let wgt = exp(-(nx * nx + ny * ny) / 8.0);
            if (!(wgt > 0.0)) {
                continue;
            }
            // `accumulate_vlfeat_sample`.
            let binx = floor(nx - 0.5);
            let biny = floor(ny - 0.5);
            let bint = floor(obin);
            let rx = nx - (binx + 0.5);
            let ry = ny - (biny + 0.5);
            let rt = obin - bint;
            for (var dbx = 0; dbx <= 1; dbx = dbx + 1) {
                let cx = i32(binx) + dbx;
                if (cx < -2 || cx >= 2) {
                    continue;
                }
                let wx = max(1.0 - abs(f32(dbx) - rx), 0.0);
                if (wx <= 0.0) {
                    continue;
                }
                for (var dby = 0; dby <= 1; dby = dby + 1) {
                    let cy = i32(biny) + dby;
                    if (cy < -2 || cy >= 2) {
                        continue;
                    }
                    let wy = max(1.0 - abs(f32(dby) - ry), 0.0);
                    if (wy <= 0.0) {
                        continue;
                    }
                    for (var dbt = 0; dbt <= 1; dbt = dbt + 1) {
                        let ori = u32(((i32(bint) + dbt) % 8 + 8) % 8);
                        let wt = select(rt, 1.0 - rt, dbt == 0);
                        if (wt <= 0.0) {
                            continue;
                        }
                        let idx = (u32(cy + 2) * 4u + u32(cx + 2)) * 8u + ori;
                        d[idx] = d[idx] + mag * wgt * wx * wy * wt;
                    }
                }
            }
        }
    }
    // `normalize_vlfeat_descriptor`.
    if (D.normalization == 1u) {
        var l1 = 0.0;
        for (var j = 0u; j < 128u; j = j + 1u) {
            l1 = l1 + abs(d[j]);
        }
        if (l1 > EPS && l1 <= 3.0e38) {
            for (var j = 0u; j < 128u; j = j + 1u) {
                d[j] = sqrt(max(d[j] / l1, 0.0));
            }
        }
    } else {
        var n2 = 0.0;
        for (var j = 0u; j < 128u; j = j + 1u) {
            n2 = n2 + d[j] * d[j];
        }
        let norm = sqrt(n2);
        if (norm > EPS && norm <= 3.0e38) {
            for (var j = 0u; j < 128u; j = j + 1u) {
                d[j] = d[j] / norm;
            }
        }
        for (var j = 0u; j < 128u; j = j + 1u) {
            d[j] = max(min(d[j], 0.2), 0.0);
        }
        var c2 = 0.0;
        for (var j = 0u; j < 128u; j = j + 1u) {
            c2 = c2 + d[j] * d[j];
        }
        let cnorm = sqrt(c2);
        if (cnorm > EPS && cnorm <= 3.0e38) {
            for (var j = 0u; j < 128u; j = j + 1u) {
                d[j] = d[j] / cnorm;
            }
        }
    }
    // Bytes, then COLMAP's VLFeat -> UBC orientation order within each cell
    // (orientation t goes to (8 - t) % 8), four bytes a word.
    var words: array<u32, 32>;
    for (var j = 0u; j < 32u; j = j + 1u) {
        words[j] = 0u;
    }
    for (var j = 0u; j < 128u; j = j + 1u) {
        let q = u32(clamp(round_half_up(d[j] * 512.0), 0.0, 255.0));
        let base = j & ~7u;
        let t = j & 7u;
        let dst = base + ((8u - t) & 7u);
        words[dst >> 2u] = words[dst >> 2u] | (q << (8u * (dst & 3u)));
    }
    for (var j = 0u; j < 32u; j = j + 1u) {
        out[i * 32u + j] = words[j];
    }
}
