// VLFeat-compatible SIFT detection (f32 ports of visloc-vision's
// `detect_sift_vlfeat_keypoints`: `refine_vlfeat_extremum` and
// `vlfeat_orientation_peaks`). The scale space comes from `image.wgsl`; the
// host orders and caps the candidates (`vlfeat.rs`), and
// `vlfeat_describe.wgsl` describes the kept ones.

struct VlParams {
    w: u32,
    h: u32,
    octave: u32,
    // Candidate capacity of each octave.
    cap: u32,
    // contrast_threshold / intervals
    peak_thresh: f32,
    edge_thresh: f32,
    // Detection: the octave's base sigma (sigma_base).
    base_sigma: f32,
    // 2^(1 / intervals)
    k: f32,
    // intervals + 1: the largest refined scale index kept.
    scale_max: f32,
    bilinear: u32,
    // Extrema: DoG levels per octave. Gradients: Gaussian levels.
    levels: u32,
    pad: u32,
};

@group(0) @binding(0) var<uniform> P: VlParams;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> cand: array<u32>;
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;

const TAU: f32 = 6.2831855;
// Words per candidate: ox, oy, dog level, orientations, x, y, scale, value,
// edge score, four angles, three spare.
const CW: u32 = 16u;

fn dog_at(level: u32, x: i32, y: i32) -> f32 {
    let xi = u32(clamp(x, 0, i32(P.w) - 1));
    let yi = u32(clamp(y, 0, i32(P.h) - 1));
    return src[level * P.w * P.h + yi * P.w + xi];
}

// `solve_vlfeat_3x3`: Gaussian elimination with partial pivoting; a zero
// step for a (near-)singular system.
fn solve3(m_in: mat3x3<f32>, r_in: vec3<f32>) -> vec3<f32> {
    // Row-major copy: a[row][col].
    var a: array<array<f32, 3>, 3>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        for (var j = 0u; j < 3u; j = j + 1u) {
            a[i][j] = m_in[j][i];
        }
    }
    var r = array<f32, 3>(r_in.x, r_in.y, r_in.z);
    for (var c = 0u; c < 3u; c = c + 1u) {
        var piv = c;
        var piv_abs = abs(a[c][c]);
        for (var row = c + 1u; row < 3u; row = row + 1u) {
            let cand_abs = abs(a[row][c]);
            if (cand_abs > piv_abs) {
                piv = row;
                piv_abs = cand_abs;
            }
        }
        if (!(piv_abs >= 1e-10) || piv_abs > 3.0e38) {
            return vec3<f32>(0.0);
        }
        if (piv != c) {
            let ta = a[piv];
            a[piv] = a[c];
            a[c] = ta;
            let tr = r[piv];
            r[piv] = r[c];
            r[c] = tr;
        }
        let d = a[c][c];
        for (var j = c; j < 3u; j = j + 1u) {
            a[c][j] = a[c][j] / d;
        }
        r[c] = r[c] / d;
        for (var row = c + 1u; row < 3u; row = row + 1u) {
            let f = a[row][c];
            for (var j = c; j < 3u; j = j + 1u) {
                a[row][j] = a[row][j] - f * a[c][j];
            }
            r[row] = r[row] - f * r[c];
        }
    }
    for (var ii = 0u; ii < 3u; ii = ii + 1u) {
        let row = 2u - ii;
        for (var j = row + 1u; j < 3u; j = j + 1u) {
            r[row] = r[row] - a[row][j] * r[j];
        }
    }
    let out = vec3<f32>(r[0], r[1], r[2]);
    if (any(out != out) || any(abs(out) > vec3<f32>(3.0e38))) {
        return vec3<f32>(0.0);
    }
    return out;
}

// One thread per interior pixel of the middle DoG level `gid.z + 1`:
// VLFeat's neighbour test, five-step quadratic refinement and gates
// (`refine_vlfeat_extremum`); survivors are appended to this octave's list.
@compute @workgroup_size(16, 16)
fn vl_extrema(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let w = i32(P.w);
    let h = i32(P.h);
    if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) {
        return;
    }
    let di = gid.z + 1u;
    let value = dog_at(di, x, y);
    let thr = max(P.peak_thresh, 0.0);
    var is_max = value >= 0.8 * thr;
    var is_min = value <= -0.8 * thr;
    if (!(is_max || is_min)) {
        return;
    }
    for (var ds = -1; ds <= 1; ds = ds + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                if (ds == 0 && dx == 0 && dy == 0) {
                    continue;
                }
                let n = dog_at(u32(i32(di) + ds), x + dx, y + dy);
                if (n >= value) {
                    is_max = false;
                }
                if (n <= value) {
                    is_min = false;
                }
                if (!(is_max || is_min)) {
                    return;
                }
            }
        }
    }

    var ix = x;
    var iy = y;
    var move_x = 0;
    var move_y = 0;
    var disp = vec3<f32>(0.0);
    var der = vec3<f32>(0.0);
    var hes = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    for (var it = 0; it < 5; it = it + 1) {
        ix = ix + move_x;
        iy = iy + move_y;
        let b = di - 1u;
        let m = di;
        let a = di + 1u;
        let centre = dog_at(m, ix, iy);
        let dxx = dog_at(m, ix + 1, iy) + dog_at(m, ix - 1, iy) - 2.0 * centre;
        let dyy = dog_at(m, ix, iy + 1) + dog_at(m, ix, iy - 1) - 2.0 * centre;
        let dss = dog_at(a, ix, iy) + dog_at(b, ix, iy) - 2.0 * centre;
        let dxy = 0.25 * (dog_at(m, ix + 1, iy + 1) + dog_at(m, ix - 1, iy - 1)
            - dog_at(m, ix - 1, iy + 1) - dog_at(m, ix + 1, iy - 1));
        let dxs = 0.25 * (dog_at(a, ix + 1, iy) + dog_at(b, ix - 1, iy)
            - dog_at(a, ix - 1, iy) - dog_at(b, ix + 1, iy));
        let dys = 0.25 * (dog_at(a, ix, iy + 1) + dog_at(b, ix, iy - 1)
            - dog_at(a, ix, iy - 1) - dog_at(b, ix, iy + 1));
        der = vec3<f32>(
            0.5 * (dog_at(m, ix + 1, iy) - dog_at(m, ix - 1, iy)),
            0.5 * (dog_at(m, ix, iy + 1) - dog_at(m, ix, iy - 1)),
            0.5 * (dog_at(a, ix, iy) - dog_at(b, ix, iy)));
        // Columns of the symmetric Hessian.
        hes = mat3x3<f32>(
            vec3<f32>(dxx, dxy, dxs),
            vec3<f32>(dxy, dyy, dys),
            vec3<f32>(dxs, dys, dss));
        disp = solve3(hes, -der);
        if (disp.x > 0.6 && ix < w - 2) {
            move_x = 1;
        } else if (disp.x < -0.6 && ix > 1) {
            move_x = -1;
        } else {
            move_x = 0;
        }
        if (disp.y > 0.6 && iy < h - 2) {
            move_y = 1;
        } else if (disp.y < -0.6 && iy > 1) {
            move_y = -1;
        } else {
            move_y = 0;
        }
        if (move_x == 0 && move_y == 0) {
            break;
        }
    }

    let centre = dog_at(di, ix, iy);
    let refined = centre + 0.5 * (der.x * disp.x + der.y * disp.y + der.z * disp.z);
    let h00 = hes[0][0];
    let h11 = hes[1][1];
    let h01 = hes[1][0];
    let det = h00 * h11 - h01 * h01;
    let trace = h00 + h11;
    let edge = trace * trace / det;
    let xn = f32(ix) + disp.x;
    let yn = f32(iy) + disp.y;
    let sn = f32(di - 1u) + disp.z;
    let edge_limit = (P.edge_thresh + 1.0) * (P.edge_thresh + 1.0) / max(P.edge_thresh, 1e-12);
    let finite_edge = edge == edge && abs(edge) <= 3.0e38;
    if (!(abs(refined) > thr && finite_edge && edge < edge_limit && edge >= 0.0
        && all(abs(disp) < vec3<f32>(1.5))
        && xn >= 0.0 && xn <= f32(w - 1) && yn >= 0.0 && yn <= f32(h - 1)
        && sn >= -1.0 && sn <= P.scale_max)) {
        return;
    }
    let slot = atomicAdd(&counters[P.octave], 1u);
    if (slot >= P.cap) {
        return;
    }
    let o = (P.octave * P.cap + slot) * CW;
    cand[o] = u32(x);
    cand[o + 1u] = u32(y);
    cand[o + 2u] = di;
    cand[o + 3u] = 0u;
    cand[o + 4u] = bitcast<u32>(xn);
    cand[o + 5u] = bitcast<u32>(yn);
    cand[o + 6u] = bitcast<u32>(sn);
    cand[o + 7u] = bitcast<u32>(refined);
    cand[o + 8u] = bitcast<u32>(edge);
}

fn gauss_at(level: u32, x: i32, y: i32) -> f32 {
    let xi = u32(clamp(x, 0, i32(P.w) - 1));
    let yi = u32(clamp(y, 0, i32(P.h) - 1));
    return src[level * P.w * P.h + yi * P.w + xi];
}

// `vlfeat_gradient_at`: central differences, one-sided on the border.
fn grad_at(level: u32, x: i32, y: i32) -> vec2<f32> {
    let w = i32(P.w);
    let h = i32(P.h);
    var gx: f32;
    if (x == 0) {
        gx = gauss_at(level, 1, y) - gauss_at(level, 0, y);
    } else if (x == w - 1) {
        gx = gauss_at(level, x, y) - gauss_at(level, x - 1, y);
    } else {
        gx = 0.5 * (gauss_at(level, x + 1, y) - gauss_at(level, x - 1, y));
    }
    var gy: f32;
    if (y == 0) {
        gy = gauss_at(level, x, 1) - gauss_at(level, x, 0);
    } else if (y == h - 1) {
        gy = gauss_at(level, x, y) - gauss_at(level, x, y - 1);
    } else {
        gy = 0.5 * (gauss_at(level, x, y + 1) - gauss_at(level, x, y - 1));
    }
    return vec2<f32>(gx, gy);
}

fn rem_tau(a: f32) -> f32 {
    var r = a % TAU;
    if (r < 0.0) {
        r = r + TAU;
    }
    return r;
}

// One thread per candidate of this octave: VLFeat's orientation histogram
// on the candidate's Gaussian level (`vlfeat_orientation_peaks`), six
// circular box smoothings and up to four peaks.
@compute @workgroup_size(64)
fn vl_orient(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 65535u * 64u;
    let n = min(atomicLoad(&counters[P.octave]), P.cap);
    if (i >= n) {
        return;
    }
    let o = (P.octave * P.cap + i) * CW;
    let level = cand[o + 2u];
    let x = bitcast<f32>(cand[o + 4u]);
    let y = bitcast<f32>(cand[o + 5u]);
    let sn = bitcast<f32>(cand[o + 6u]);
    let sigma = P.base_sigma * pow(P.k, sn + 1.0);
    let xi = i32(floor(x + 0.5));
    let yi = i32(floor(y + 0.5));
    let sigmaw = 1.5 * max(sigma, 1e-6);
    let win = i32(max(floor(3.0 * sigmaw), 1.0));
    var hist: array<f32, 36>;
    for (var b = 0u; b < 36u; b = b + 1u) {
        hist[b] = 0.0;
    }
    let y0 = max(-win, -yi);
    let y1 = min(win, i32(P.h) - 1 - yi);
    let x0 = max(-win, -xi);
    let x1 = min(win, i32(P.w) - 1 - xi);
    let r2max = f32(win * win) + 0.6;
    let denom = 2.0 * sigmaw * sigmaw;
    for (var ys = y0; ys <= y1; ys = ys + 1) {
        for (var xs = x0; xs <= x1; xs = xs + 1) {
            let dx = f32(xi + xs) - x;
            let dy = f32(yi + ys) - y;
            let r2 = dx * dx + dy * dy;
            if (r2 >= r2max) {
                continue;
            }
            let g = grad_at(level, xi + xs, yi + ys);
            let mag = sqrt(g.x * g.x + g.y * g.y);
            if (!(mag > 0.0) || mag > 3.0e38) {
                continue;
            }
            let wgt = exp(-r2 / denom);
            let ang = rem_tau(atan2(g.y, g.x));
            let fbin = rem_tau(ang) * 36.0 / TAU;
            if (P.bilinear != 0u) {
                let bin = i32(floor(fbin - 0.5));
                let rbin = fbin - f32(bin) - 0.5;
                let contrib = mag * wgt;
                let b0 = u32((bin % 36 + 36) % 36);
                let b1 = u32(((bin + 1) % 36 + 36) % 36);
                hist[b0] = hist[b0] + (1.0 - rbin) * contrib;
                hist[b1] = hist[b1] + rbin * contrib;
            } else {
                let b = u32(floor(fbin)) % 36u;
                hist[b] = hist[b] + mag * wgt;
            }
        }
    }
    // `smooth_orientation_histogram`.
    for (var b = 0u; b < 36u; b = b + 1u) {
        let v = hist[b];
        hist[b] = select(0.0, v, v > 0.0 && v <= 3.0e38);
    }
    for (var smooth_i = 0u; smooth_i < 6u; smooth_i = smooth_i + 1u) {
        var next: array<f32, 36>;
        for (var b = 0u; b < 36u; b = b + 1u) {
            let prev = hist[(b + 35u) % 36u];
            let nx = hist[(b + 1u) % 36u];
            next[b] = (prev + hist[b] + nx) / 3.0;
        }
        hist = next;
    }
    // `select_vlfeat_orientation_peaks`.
    var max_hist = 0.0;
    for (var b = 0u; b < 36u; b = b + 1u) {
        max_hist = max(max_hist, hist[b]);
    }
    var count = 0u;
    if (max_hist > 0.0 && max_hist <= 3.0e38) {
        for (var b = 0u; b < 36u; b = b + 1u) {
            let v = hist[b];
            let prev = hist[(b + 35u) % 36u];
            let nx = hist[(b + 1u) % 36u];
            if (!(v > 0.8 * max_hist && v > prev && v > nx)) {
                continue;
            }
            let den = nx + prev - 2.0 * v;
            var delta = 0.0;
            if (abs(den) > 2.220446e-16 && abs(den) <= 3.0e38) {
                delta = clamp(-0.5 * (nx - prev) / den, -0.5, 0.5);
            }
            let angle = rem_tau((f32(b) + delta + 0.5) * TAU / 36.0);
            cand[o + 9u + count] = bitcast<u32>(angle);
            count = count + 1u;
            if (count == 4u) {
                break;
            }
        }
    }
    cand[o + 3u] = count;
}
