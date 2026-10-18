// ACMH/ACMM PatchMatch on the GPU: the per-pixel update of
// `patch_match.rs`, one invocation per pixel. `init` gives every pixel its
// first plane and cost; `propagate` updates the pixels of one checkerboard
// colour from the other's planes. With APD-MVS's deformable patches,
// `classify` sorts the pixels and `find_anchors` gives each weak pixel its
// anchors between the two. Planes are `(n.xyz, rho)` in the reference
// camera frame, `rho = n · X`.

const MAX_COST: f32 = 2.0;
const MAX_SAMPLES: u32 = 64u;
const MIN_WEIGHT_SUM: f32 = 1e-6;
const MAX_SOURCES: u32 = 32u;
const NO_VIEW: u32 = 0xffffffffu;
// Draws of the sampled view weights, and the mark of their random streams.
const VIEW_SAMPLES: u32 = 15u;
const SAMPLING_STREAM: u32 = 0x80000000u;
// Deformable patches: anchors per weak pixel, samples of an anchor window,
// the longest cost curve, sectors searched at most, the border no pixel
// within is sorted or anchored, and the pixel classes.
const MAX_ANCHORS: u32 = 8u;
const ANCHOR_SAMPLES: u32 = 9u;
const MAX_CURVE: u32 = 61u;
const MAX_SECTORS: u32 = 32u;
const BORDER: u32 = 6u;
const WEAK: u32 = 0u;
const STRONG: u32 = 1u;
const UNKNOWN: u32 = 2u;

struct Params {
    width: u32,
    height: u32,
    num_sources: u32,
    color: u32,
    iteration: u32,
    seed: u32,
    view_id: u32,
    geometric: u32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    depth_min: f32,
    depth_max: f32,
    tau: f32,
    tau1: f32,
    beta: f32,
    n1: u32,
    n2: u32,
    init_top_k: u32,
    geom_lambda: f32,
    geom_delta: f32,
    depth_perturbation: f32,
    normal_perturbation: f32,
    window_radius: i32,
    window_step: i32,
    sigma_spatial: f32,
    sigma_color: f32,
    // 0: the paper's confidence weights; 1: ACMM's sampled weights.
    view_selection: u32,
    // One over the source texture's size, for normalized coordinates.
    inv_texture_width: f32,
    inv_texture_height: f32,
    // 1: ACMP's planar prior pass (`prior_planes`); its posterior's terms:
    // 1 / alpha, gamma, 1 / (2 sigma_depth^2), 1 / (2 sigma_angle^2), and
    // sigma_depth and sigma_angle (radians) for the refinement's draws.
    planar_prior: u32,
    prior_inv_alpha: f32,
    prior_gamma: f32,
    prior_depth_weight: f32,
    prior_angle_weight: f32,
    prior_depth_sigma: f32,
    prior_angle_sigma: f32,
    // The detail restorer: 0 off; 1 against the start plane's cost; 2
    // against the plane at infinity's (`pre_costs`), with this margin.
    restore: u32,
    restore_margin: f32,
    // 1: the pass has deformable patches (`anchors`), with the sorting's
    // sweep, peak radius (disparity steps), peak cost, single-peak cost and
    // spread; the anchors' sectors, RANSAC threshold (depth units) and
    // inliers; the window's weight in a weak pixel's cost; and the mean
    // baseline to the sources.
    deformable: u32,
    sweep: i32,
    peak_radius: i32,
    max_peak_cost: f32,
    strong_cost: f32,
    peak_spread: f32,
    sectors: u32,
    ransac_threshold: f32,
    min_inliers: u32,
    center_weight: f32,
    baseline: f32,
}

// Reference camera frame to source camera frame: rows of R with t in w.
struct Source {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
    // fx, fy, cx, cy
    k: vec4<f32>,
    // width, height, unused, prior depth offset
    dims: vec4<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sources: array<Source>;
// The reference image.
@group(0) @binding(2) var<storage, read> images: array<f32>;
// Each source's earlier depth map (geometric mode).
@group(0) @binding(3) var<storage, read> priors: array<f32>;
@group(0) @binding(4) var<storage, read_write> planes: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> costs: array<f32>;
// Per pixel: the previous iteration's most important view (confidence
// weights) or the selected views as bits (sampled weights).
@group(0) @binding(6) var<storage, read_write> best_views: array<u32>;
// `init`: the earlier map's (normal, depth) per pixel, depth 0 for none.
@group(0) @binding(7) var<storage, read> init_planes: array<vec4<f32>>;
// Source j's grey levels in layer j (half floats), sampled bilinearly by
// the texture unit.
@group(0) @binding(8) var source_images: texture_2d_array<f32>;
@group(0) @binding(9) var bilinear: sampler;
// `propagate` in the planar prior pass: per pixel, the prior plane
// `(n.xyz, rho)`, all zeros for none.
@group(0) @binding(10) var<storage, read> prior_planes: array<vec4<f32>>;
// The detail restorer's baseline costs, from `init`: a pixel takes a new
// plane only if its cost is below its baseline by `restore_margin`.
@group(0) @binding(11) var<storage, read_write> pre_costs: array<f32>;
// Deformable patches: per pixel its class (`classify`), its nearest strong
// pixel (the CPU's breadth-first search, NO_VIEW for none) and its anchors
// (`find_anchors`, MAX_ANCHORS per pixel, NO_VIEW past the last; a first
// NO_VIEW: matched as a strong pixel).
@group(0) @binding(12) var<storage, read_write> classes: array<u32>;
@group(0) @binding(13) var<storage, read> nearest: array<u32>;
@group(0) @binding(14) var<storage, read_write> anchors: array<u32>;

// The reference window of the current pixel, row by row.
var<private> patch_values: array<f32, 64>;
var<private> patch_weights: array<f32, 64>;
var<private> patch_n: u32;
var<private> patch_center: f32;
var<private> patch_sum_w: f32;
var<private> patch_mean: f32;
var<private> patch_var: f32;
var<private> rng_state: u32;
var<private> top_k_views: u32;
// A weak pixel's anchor windows (offsets from the pixel; grey levels minus
// each window's centre); `anchor_count` 0 matches the plain window only.
var<private> anchor_count: u32;
var<private> anchor_offsets: array<vec2<f32>, 72>;
var<private> anchor_values: array<f32, 72>;
var<private> anchor_centers: array<f32, 8>;
var<private> anchor_means: array<f32, 8>;
var<private> anchor_vars: array<f32, 8>;
var<private> draw_state: u32;

fn hash(v: u32) -> u32 {
    var x = v;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn rng_init(x: u32, y: u32, pass_id: u32) {
    var h = hash(params.seed ^ 0x9e3779b9u);
    h = hash(h ^ params.view_id);
    h = hash(h ^ x);
    h = hash(h ^ y);
    h = hash(h ^ pass_id);
    rng_state = h;
}

fn uniform01() -> f32 {
    rng_state = hash(rng_state + 0x9e3779b9u);
    return f32(rng_state >> 8u) * (1.0 / 16777216.0);
}

fn ref_gray(x: i32, y: i32) -> f32 {
    let w = i32(params.width);
    let h = i32(params.height);
    let cx = clamp(x, 0, w - 1);
    let cy = clamp(y, 0, h - 1);
    return images[u32(cy) * params.width + u32(cx)];
}

fn pixel_ray(x: u32, y: u32) -> vec3<f32> {
    return vec3<f32>(
        (f32(x) + 0.5 - params.cx) / params.fx,
        (f32(y) + 0.5 - params.cy) / params.fy,
        1.0,
    );
}

fn load_patch(x: u32, y: u32) {
    let center = ref_gray(i32(x), i32(y));
    patch_center = center;
    let inv_color = 1.0 / (2.0 * params.sigma_color * params.sigma_color);
    let inv_spatial = 1.0 / (2.0 * params.sigma_spatial * params.sigma_spatial);
    var n = 0u;
    var sum_w = 0.0;
    var sum = 0.0;
    var sum_sq = 0.0;
    let r = params.window_radius;
    let step = max(params.window_step, 1);
    for (var dy = -r; dy <= r; dy += step) {
        for (var dx = -r; dx <= r; dx += step) {
            if (n >= MAX_SAMPLES) {
                break;
            }
            let value = ref_gray(i32(x) + dx, i32(y) + dy) - center;
            let weight = exp(-value * value * inv_color - f32(dx * dx + dy * dy) * inv_spatial);
            patch_values[n] = value;
            patch_weights[n] = weight;
            sum_w += weight;
            sum += weight * value;
            sum_sq += weight * value * value;
            n += 1u;
        }
    }
    patch_n = n;
    patch_sum_w = sum_w;
    if (sum_w < MIN_WEIGHT_SUM) {
        // Every sample differs too much from the centre (the window skips
        // it): nothing to match, where 0/0 would give NaN costs.
        patch_mean = 0.0;
        patch_var = 0.0;
        return;
    }
    patch_mean = sum / sum_w;
    patch_var = sum_sq / sum_w - patch_mean * patch_mean;
}

// The loaded window uniformly weighted (`RefPatch::uniform`): APD-MVS sorts
// pixels with plain NCC.
fn uniform_patch() {
    var sum = 0.0;
    var sum_sq = 0.0;
    for (var k = 0u; k < patch_n; k++) {
        patch_weights[k] = 1.0;
        sum += patch_values[k];
        sum_sq += patch_values[k] * patch_values[k];
    }
    let n = f32(patch_n);
    patch_sum_w = n;
    patch_mean = sum / n;
    patch_var = sum_sq / n - patch_mean * patch_mean;
}

fn plane_depth(plane: vec4<f32>, ray: vec3<f32>) -> f32 {
    let denom = dot(plane.xyz, ray);
    if (denom >= -1e-6) {
        return -1.0;
    }
    let d = plane.w / denom;
    if (!(d > 0.0) || d > 3.0e38) {
        return -1.0;
    }
    return d;
}

// Source j's grey level at pixel coordinates (u, v) (pixel centres at
// +0.5), clamped to its image: the CPU path's bilinear lookup, by the
// texture unit.
fn source_gray(j: u32, s: Source, u: f32, v: f32) -> f32 {
    let x = clamp(u, 0.5, f32(s.dims.x) - 0.5);
    let y = clamp(v, 0.5, f32(s.dims.y) - 0.5);
    let uv = vec2<f32>(x * params.inv_texture_width, y * params.inv_texture_height);
    return textureSampleLevel(source_images, bilinear, uv, j, 0.0).r;
}

// The pixel of the image of `ray`, plus `dx`, `dy` pixels, in source `s`
// under the homography `a` (the ray's image), `bx`, `by` (unit steps):
// (u, v), or (-1, -1) behind the camera.
fn warp_sample(s: Source, a: vec3<f32>, bx: vec3<f32>, by: vec3<f32>, dx: f32, dy: f32) -> vec2<f32> {
    let p = a + dx * bx + dy * by;
    if (p.z <= 0.0) {
        return vec2<f32>(-1.0, -1.0);
    }
    return vec2<f32>(s.k.x * p.x / p.z + s.k.z, s.k.y * p.y / p.z + s.k.w);
}

// One minus the bilateral NCC of `plane` at the current pixel in source j;
// for a weak pixel with anchors (`anchor_count`), the deformable cost:
// `center_weight` times that plus the rest times the anchor windows' mean.
fn photometric_cost(j: u32, ray: vec3<f32>, plane: vec4<f32>) -> f32 {
    if (abs(plane.w) < 1e-12) {
        return MAX_COST;
    }
    let s = sources[j];
    let t = vec3<f32>(s.r0.w, s.r1.w, s.r2.w);
    let inv_rho = 1.0 / plane.w;
    let m0 = s.r0.xyz + t.x * plane.xyz * inv_rho;
    let m1 = s.r1.xyz + t.y * plane.xyz * inv_rho;
    let m2 = s.r2.xyz + t.z * plane.xyz * inv_rho;
    let a = vec3<f32>(dot(m0, ray), dot(m1, ray), dot(m2, ray));
    let bx = vec3<f32>(m0.x, m1.x, m2.x) / params.fx;
    let by = vec3<f32>(m0.y, m1.y, m2.y) / params.fy;
    if (a.z <= 0.0) {
        return MAX_COST;
    }
    let cu = s.k.x * a.x / a.z + s.k.z;
    let cv = s.k.y * a.y / a.z + s.k.w;
    if (cu < 0.0 || cv < 0.0 || cu >= f32(s.dims.x) || cv >= f32(s.dims.y)) {
        return MAX_COST;
    }
    var center = MAX_COST;
    if (patch_var >= 1e-5) {
        var s_w = 0.0;
        var s_ww = 0.0;
        var s_rw = 0.0;
        var ok = true;
        // The window in `load_patch`'s order, without integer division.
        let r = params.window_radius;
        let step = max(params.window_step, 1);
        var k = 0u;
        for (var dy = -r; dy <= r; dy += step) {
            for (var dx = -r; dx <= r; dx += step) {
                if (k >= patch_n) {
                    break;
                }
                let p = a + f32(dx) * bx + f32(dy) * by;
                if (p.z <= 0.0) {
                    ok = false;
                    break;
                }
                let u = s.k.x * p.x / p.z + s.k.z;
                let v = s.k.y * p.y / p.z + s.k.w;
                let value = source_gray(j, s, u, v) - patch_center;
                let weight = patch_weights[k];
                s_w += weight * value;
                s_ww += weight * value * value;
                s_rw += weight * patch_values[k] * value;
                k += 1u;
            }
            if (!ok) {
                break;
            }
        }
        let mean = s_w / patch_sum_w;
        let var_s = s_ww / patch_sum_w - mean * mean;
        if (ok && var_s >= 1e-5) {
            let cov = s_rw / patch_sum_w - patch_mean * mean;
            let ncc = cov / sqrt(patch_var * var_s);
            center = clamp(1.0 - ncc, 0.0, MAX_COST);
        }
    }
    if (anchor_count == 0u) {
        return center;
    }
    var sum = 0.0;
    for (var q = 0u; q < anchor_count; q++) {
        var cost = MAX_COST;
        var ok = anchor_vars[q] >= 1e-5;
        var s1 = 0.0;
        var s2 = 0.0;
        var sr = 0.0;
        for (var m = 0u; m < ANCHOR_SAMPLES; m++) {
            if (!ok) {
                break;
            }
            let o = anchor_offsets[q * ANCHOR_SAMPLES + m];
            let uv = warp_sample(s, a, bx, by, o.x, o.y);
            // An anchor window leaving the source image costs the maximum.
            if (uv.x < 0.0 || uv.y < 0.0 || uv.x >= f32(s.dims.x) || uv.y >= f32(s.dims.y)) {
                ok = false;
                break;
            }
            let value = source_gray(j, s, uv.x, uv.y) - anchor_centers[q];
            s1 += value;
            s2 += value * value;
            sr += anchor_values[q * ANCHOR_SAMPLES + m] * value;
        }
        if (ok) {
            let nn = f32(ANCHOR_SAMPLES);
            let mean = s1 / nn;
            let var_s = s2 / nn - mean * mean;
            if (var_s >= 1e-5) {
                let cov = sr / nn - anchor_means[q] * mean;
                cost = clamp(1.0 - cov / sqrt(anchor_vars[q] * var_s), 0.0, MAX_COST);
            }
        }
        sum += cost;
    }
    return params.center_weight * center
        + (1.0 - params.center_weight) * min(sum / f32(anchor_count), MAX_COST);
}

// lambda times the forward-backward reprojection error through source j's
// earlier depth map (zero outside the geometric mode).
fn geometric_cost(j: u32, x: u32, y: u32, ray: vec3<f32>, plane: vec4<f32>) -> f32 {
    if (params.geometric == 0u) {
        return 0.0;
    }
    let delta = params.geom_delta;
    let lambda = params.geom_lambda;
    let depth = plane_depth(plane, ray);
    if (depth <= 0.0) {
        return lambda * delta;
    }
    let s = sources[j];
    let t = vec3<f32>(s.r0.w, s.r1.w, s.r2.w);
    let point = ray * depth;
    let p = vec3<f32>(dot(s.r0.xyz, point), dot(s.r1.xyz, point), dot(s.r2.xyz, point)) + t;
    if (p.z <= 0.0) {
        return lambda * delta;
    }
    let u = s.k.x * p.x / p.z + s.k.z;
    let v = s.k.y * p.y / p.z + s.k.w;
    if (u < 0.0 || v < 0.0 || u >= f32(s.dims.x) || v >= f32(s.dims.y)) {
        return lambda * delta;
    }
    let source_depth = priors[s.dims.w + u32(v) * s.dims.x + u32(u)];
    if (source_depth <= 0.0) {
        return lambda * delta;
    }
    let back = vec3<f32>(
        (u - s.k.z) / s.k.x * source_depth,
        (v - s.k.w) / s.k.y * source_depth,
        source_depth,
    );
    let d = back - t;
    // Rᵀ d: R's columns are the rows' components.
    let q = vec3<f32>(
        s.r0.x * d.x + s.r1.x * d.y + s.r2.x * d.z,
        s.r0.y * d.x + s.r1.y * d.y + s.r2.y * d.z,
        s.r0.z * d.x + s.r1.z * d.y + s.r2.z * d.z,
    );
    if (q.z <= 0.0) {
        return lambda * delta;
    }
    let du = params.fx * q.x / q.z + params.cx - (f32(x) + 0.5);
    let dv = params.fy * q.y / q.z + params.cy - (f32(y) + 0.5);
    return lambda * min(sqrt(du * du + dv * dv), delta);
}

fn random_normal(ray: vec3<f32>) -> vec3<f32> {
    let z = 2.0 * uniform01() - 1.0;
    let phi = 6.2831855 * uniform01();
    let s = sqrt(max(1.0 - z * z, 0.0));
    let n = vec3<f32>(s * cos(phi), s * sin(phi), z);
    if (dot(n, ray) > 0.0) {
        return -n;
    }
    return n;
}

fn perturb_normal(n: vec3<f32>, ray: vec3<f32>, amount: f32) -> vec3<f32> {
    let p = vec3<f32>(
        n.x + (uniform01() - 0.5) * amount,
        n.y + (uniform01() - 0.5) * amount,
        n.z + (uniform01() - 0.5) * amount,
    );
    let q = p / sqrt(dot(p, p));
    if (dot(q, ray) >= 0.0) {
        return n;
    }
    return q;
}

fn plane_from_depth(depth: f32, n: vec3<f32>, ray: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(n, depth * dot(n, ray));
}

// The mean of the best `init_top_k` per-view costs of a plane; those views
// go to `top_k_views` as bits.
fn top_k_cost(x: u32, y: u32, ray: vec3<f32>, plane: vec4<f32>) -> f32 {
    let n = params.num_sources;
    top_k_views = 0u;
    if (n == 0u) {
        return MAX_COST;
    }
    let k = clamp(params.init_top_k, 1u, min(n, 8u));
    var best: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) {
        best[i] = 3.0e38;
    }
    var view_costs: array<f32, 32>;
    for (var j = 0u; j < n; j++) {
        var c = photometric_cost(j, ray, plane) + geometric_cost(j, x, y, ray, plane);
        view_costs[j] = c;
        // Insert into the sorted best k.
        for (var i = 0u; i < k; i++) {
            if (c < best[i]) {
                let tmp = best[i];
                best[i] = c;
                c = tmp;
            }
        }
    }
    var sum = 0.0;
    for (var i = 0u; i < k; i++) {
        sum += best[i];
    }
    for (var j = 0u; j < n; j++) {
        if (view_costs[j] <= best[k - 1u]) {
            top_k_views |= 1u << j;
        }
    }
    return sum / f32(k);
}

@compute @workgroup_size(16, 8, 1)
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    let i = y * params.width + x;
    let ray = pixel_ray(x, y);
    var plane: vec4<f32>;
    let prior = init_planes[i];
    if (prior.w > 0.0) {
        plane = plane_from_depth(prior.w, prior.xyz, ray);
    } else {
        rng_init(x, y, 0xffffffffu);
        let depth = params.depth_min + uniform01() * (params.depth_max - params.depth_min);
        plane = plane_from_depth(depth, random_normal(ray), ray);
    }
    load_patch(x, y);
    planes[i] = plane;
    costs[i] = top_k_cost(x, y, ray, plane);
    if (params.view_selection == 1u) {
        best_views[i] = top_k_views;
    } else {
        best_views[i] = NO_VIEW;
    }
    if (params.restore == 1u) {
        pre_costs[i] = costs[i];
    } else if (params.restore == 2u) {
        // With no normal, a plane's homography is the rotation's.
        pre_costs[i] = top_k_cost(x, y, ray, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    }
}

// ---- APD-MVS's deformable patches (`deformable.rs`) ----

// The class of a cost curve of `len` samples (centre at the pixel's
// depth): `deformable::classify`.
fn classify_curve(curve: ptr<function, array<f32, 61>>, len: u32) -> u32 {
    let center = i32(len / 2u);
    var lowest = -1;
    var low = 3.0e38;
    var peaks = 0u;
    for (var i = 2u; i + 2u < len; i++) {
        let c = (*curve)[i];
        if ((*curve)[i - 1u] > c && (*curve)[i + 1u] > c) {
            peaks += 1u;
            if (c < low) {
                low = c;
                lowest = i32(i);
            }
        }
    }
    if (lowest < 0) {
        return WEAK;
    }
    if (abs(lowest - center) > params.peak_radius || low > params.max_peak_cost) {
        return WEAK;
    }
    if (peaks == 1u) {
        if (low <= params.strong_cost) {
            return STRONG;
        }
        return WEAK;
    }
    var excess = 0.0;
    for (var i = 2u; i + 2u < len; i++) {
        let c = (*curve)[i];
        if (i32(i) != lowest && (*curve)[i - 1u] > c && (*curve)[i + 1u] > c) {
            excess += (c - low) * (c - low);
        }
    }
    if (sqrt(excess) / f32(peaks - 1u) > params.peak_spread) {
        return STRONG;
    }
    return WEAK;
}

// Each pixel's class from its plain-NCC cost (plus the geometric term) over
// fixed views at `2 sweep + 1` depths one pseudo-disparity step apart along
// its ray, normal fixed.
@compute @workgroup_size(16, 8, 1)
fn classify(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = params.width;
    let h = params.height;
    if (x >= w || y >= h) {
        return;
    }
    let i = y * w + x;
    if (params.num_sources == 0u || x < BORDER || y < BORDER || x + BORDER >= w || y + BORDER >= h) {
        classes[i] = UNKNOWN;
        return;
    }
    let ray = pixel_ray(x, y);
    let plane = planes[i];
    let depth = plane_depth(plane, ray);
    if (depth <= 0.0) {
        classes[i] = UNKNOWN;
        return;
    }
    load_patch(x, y);
    // The curve's views: the best `init_top_k` at the pixel's depth (ties
    // to the lower index), the same at every depth.
    let n = params.num_sources;
    let k = clamp(params.init_top_k, 1u, min(n, 8u));
    var center_costs: array<f32, 32>;
    for (var j = 0u; j < n; j++) {
        center_costs[j] = view_cost(j, x, y, ray, plane);
    }
    var views: array<u32, 8>;
    var taken = 0u;
    for (var q = 0u; q < k; q++) {
        var best = NO_VIEW;
        for (var j = 0u; j < n; j++) {
            if (((taken >> j) & 1u) == 1u) {
                continue;
            }
            if (best == NO_VIEW || center_costs[j] < center_costs[best]) {
                best = j;
            }
        }
        views[q] = best;
        taken |= 1u << best;
    }
    uniform_patch();
    let disparity = params.fx * params.baseline / depth;
    let len = min(u32(2 * params.sweep + 1), MAX_CURVE);
    var curve: array<f32, 61>;
    for (var m = 0u; m < len; m++) {
        let step = disparity + f32(i32(m) - params.sweep);
        let d = params.fx * params.baseline / step;
        if (step <= 0.0 || !(d >= params.depth_min && d <= params.depth_max)) {
            curve[m] = MAX_COST;
        } else {
            let p = plane_from_depth(d, plane.xyz, ray);
            var sum = 0.0;
            for (var q = 0u; q < k; q++) {
                sum += view_cost(views[q], x, y, ray, p);
            }
            curve[m] = min(sum / f32(k), MAX_COST);
        }
    }
    classes[i] = classify_curve(&curve, len);
}

// `deformable::Draws`: xorshift32, seeded with a set low bit.
fn draws_below(n: u32) -> u32 {
    draw_state = draw_state ^ (draw_state << 13u);
    draw_state = draw_state ^ (draw_state >> 17u);
    draw_state = draw_state ^ (draw_state << 5u);
    return draw_state % max(n, 1u);
}

// `deformable::in_triangle`.
fn in_triangle(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, p: vec2<f32>) -> bool {
    if (distance(a, b) <= 2.0 || distance(b, c) <= 2.0 || distance(c, a) <= 2.0) {
        return false;
    }
    let area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    if (abs(area) < 1e-6) {
        return false;
    }
    let d1 = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
    let d2 = (c.x - b.x) * (p.y - b.y) - (c.y - b.y) * (p.x - b.x);
    let d3 = (a.x - c.x) * (p.y - c.y) - (a.y - c.y) * (p.x - c.x);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    return !(has_neg && has_pos);
}

// `deformable::plane_through`: (n, w) with n · X + w = 0, zero for
// collinear points.
fn plane_through(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec4<f32> {
    let n = cross(a - c, b - c);
    let norm = length(n);
    if (norm < 1e-12) {
        return vec4<f32>(0.0);
    }
    let u = n / norm;
    return vec4<f32>(u, -dot(u, a));
}

// The 3D point of pixel index `q` on its current plane, w 0 for none.
fn anchor_point(q: u32) -> vec4<f32> {
    let ray = pixel_ray(q % params.width, q / params.width);
    let d = plane_depth(planes[q], ray);
    if (d <= 0.0) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(ray * d, 1.0);
}

fn pixel_xy(q: u32) -> vec2<f32> {
    return vec2<f32>(f32(q % params.width), f32(q / params.width));
}

// A weak pixel's anchors: the first strong pixel along each sector's ray
// (or a weak sample's nearest strong pixel within the sector), kept if
// RANSAC finds enough coplanar around it (`Context::find_anchors`).
@compute @workgroup_size(16, 8, 1)
fn find_anchors(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = params.width;
    let h = params.height;
    if (x >= w || y >= h) {
        return;
    }
    let i = y * w + x;
    for (var k = 0u; k < MAX_ANCHORS; k++) {
        anchors[i * MAX_ANCHORS + k] = NO_VIEW;
    }
    if (classes[i] != WEAK) {
        return;
    }
    let sectors = clamp(params.sectors, 1u, MAX_SECTORS);
    let half = 3.14159265 / f32(sectors);
    let cos_half = cos(half);
    var found: array<u32, 32>;
    var count = 0u;
    for (var sector = 0u; sector < sectors; sector++) {
        let angle = 2.0 * half * f32(sector);
        let dir = vec2<f32>(cos(angle), sin(angle));
        var radius = 2.0;
        loop {
            let qx = floor(f32(x) + 0.5 + dir.x * radius);
            let qy = floor(f32(y) + 0.5 + dir.y * radius);
            if (qx < 0.0 || qy < 0.0 || qx >= f32(w) || qy >= f32(h)) {
                break;
            }
            let ux = u32(qx);
            let uy = u32(qy);
            var taken = false;
            if (ux >= BORDER && uy >= BORDER && ux + BORDER < w && uy + BORDER < h) {
                let q = uy * w + ux;
                var a = NO_VIEW;
                if (classes[q] == STRONG) {
                    a = q;
                } else {
                    a = nearest[q];
                }
                if (a != NO_VIEW) {
                    let off = pixel_xy(a) - vec2<f32>(f32(x), f32(y));
                    let norm = length(off);
                    if (norm > 0.0 && dot(off, dir) / norm > cos_half) {
                        found[count] = a;
                        count += 1u;
                        taken = true;
                    }
                }
            }
            if (taken) {
                break;
            }
            radius = min(2.0 * radius, radius + 25.0);
        }
    }
    if (count < 4u) {
        return;
    }
    let center_point = anchor_point(i);
    if (center_point.w == 0.0) {
        return;
    }
    // The anchors with a depth, as `Context::find_anchors` keeps them.
    var pixels: array<u32, 32>;
    var points: array<vec3<f32>, 32>;
    var n = 0u;
    for (var k = 0u; k < count; k++) {
        let p = anchor_point(found[k]);
        if (p.w != 0.0) {
            pixels[n] = found[k];
            points[n] = p.xyz;
            n += 1u;
        }
    }
    if (n < 3u) {
        return;
    }
    let center = vec2<f32>(f32(x), f32(y));
    draw_state = (params.seed ^ (i * 0x9e3779b9u)) | 1u;
    var best_inliers = 0u;
    var best_center = 3.0e38;
    var best_plane = vec4<f32>(0.0);
    var best_sample = vec3<u32>(0u);
    for (var it = 0u; it < 50u; it++) {
        let ia = draws_below(n);
        let ib = draws_below(n);
        let ic = draws_below(n);
        if (ia == ib || ib == ic || ia == ic) {
            continue;
        }
        if (!in_triangle(pixel_xy(pixels[ia]), pixel_xy(pixels[ib]), pixel_xy(pixels[ic]), center)) {
            continue;
        }
        let plane = plane_through(points[ia], points[ib], points[ic]);
        if (all(plane.xyz == vec3<f32>(0.0))) {
            continue;
        }
        var inliers = 0u;
        for (var k = 0u; k < n; k++) {
            if (abs(dot(plane.xyz, points[k]) + plane.w) < params.ransac_threshold) {
                inliers += 1u;
            }
        }
        if (inliers < params.min_inliers) {
            continue;
        }
        let cd = abs(dot(plane.xyz, center_point.xyz) + plane.w);
        if (best_inliers == 0u || inliers > best_inliers || (inliers == best_inliers && cd < best_center)) {
            best_inliers = inliers;
            best_center = cd;
            best_plane = plane;
            best_sample = vec3<u32>(ia, ib, ic);
        }
    }
    if (best_inliers == 0u) {
        return;
    }
    // Inliers: the sample first, then by distance (index order on ties).
    var order: array<u32, 32>;
    var keys: array<f32, 32>;
    var m = 0u;
    for (var k = 0u; k < n; k++) {
        let d = abs(dot(best_plane.xyz, points[k]) + best_plane.w);
        if (d < params.ransac_threshold) {
            var key = d;
            if (k == best_sample.x || k == best_sample.y || k == best_sample.z) {
                key = -1.0;
            }
            // Insertion into the sorted keys (stable).
            var at = m;
            while (at > 0u && keys[at - 1u] > key) {
                keys[at] = keys[at - 1u];
                order[at] = order[at - 1u];
                at -= 1u;
            }
            keys[at] = key;
            order[at] = k;
            m += 1u;
        }
    }
    for (var k = 0u; k < min(m, MAX_ANCHORS); k++) {
        anchors[i * MAX_ANCHORS + k] = pixels[order[k]];
    }
    // For the diagnostics: a weak pixel with anchors.
    classes[i] = 3u;
}

// The anchor windows of pixel (x, y): a 3x3 grid 5 pixels apart around each
// anchor (`Context::anchor_patches`).
fn load_anchor_patches(x: u32, y: u32) {
    let i = y * params.width + x;
    anchor_count = 0u;
    for (var k = 0u; k < MAX_ANCHORS; k++) {
        let a = anchors[i * MAX_ANCHORS + k];
        if (a == NO_VIEW) {
            break;
        }
        let ax = i32(a % params.width);
        let ay = i32(a / params.width);
        let center = ref_gray(ax, ay);
        var s1 = 0.0;
        var s2 = 0.0;
        var m = 0u;
        for (var dy = -5; dy <= 5; dy += 5) {
            for (var dx = -5; dx <= 5; dx += 5) {
                let sx = clamp(ax + dx, 0, i32(params.width) - 1);
                let sy = clamp(ay + dy, 0, i32(params.height) - 1);
                let value = ref_gray(sx, sy) - center;
                anchor_offsets[k * ANCHOR_SAMPLES + m] =
                    vec2<f32>(f32(sx - i32(x)), f32(sy - i32(y)));
                anchor_values[k * ANCHOR_SAMPLES + m] = value;
                s1 += value;
                s2 += value * value;
                m += 1u;
            }
        }
        let nn = f32(ANCHOR_SAMPLES);
        anchor_centers[k] = center;
        anchor_means[k] = s1 / nn;
        anchor_vars[k] = s2 / nn - (s1 / nn) * (s1 / nn);
        anchor_count = k + 1u;
    }
}

// `deformable::fitted_plane` on pixel (x, y)'s anchors' current points, as
// a plane `(n, rho)` facing the camera; zero for none.
fn fitted_plane(x: u32, y: u32, ray: vec3<f32>) -> vec4<f32> {
    let i = y * params.width + x;
    var pixels: array<u32, 8>;
    var points: array<vec3<f32>, 8>;
    var n = 0u;
    for (var k = 0u; k < MAX_ANCHORS; k++) {
        let a = anchors[i * MAX_ANCHORS + k];
        if (a == NO_VIEW) {
            break;
        }
        let p = anchor_point(a);
        if (p.w != 0.0) {
            pixels[n] = a;
            points[n] = p.xyz;
            n += 1u;
        }
    }
    if (n < 3u) {
        return vec4<f32>(0.0);
    }
    let center = vec2<f32>(f32(x), f32(y));
    var best = vec4<f32>(0.0);
    var best_score = 3.0e38;
    var found = false;
    for (var it = 0u; it < 50u; it++) {
        let ia = draws_below(n);
        let ib = draws_below(n);
        let ic = draws_below(n);
        if (ia == ib || ib == ic || ia == ic) {
            continue;
        }
        if (!in_triangle(pixel_xy(pixels[ia]), pixel_xy(pixels[ib]), pixel_xy(pixels[ic]), center)) {
            continue;
        }
        let plane = plane_through(points[ia], points[ib], points[ic]);
        if (all(plane.xyz == vec3<f32>(0.0))) {
            continue;
        }
        var score = 0.0;
        for (var k = 0u; k < n; k++) {
            if (k != ia && k != ib && k != ic) {
                score += abs(dot(plane.xyz, points[k]) + plane.w);
            }
        }
        if (!found || score < best_score) {
            found = true;
            best_score = score;
            best = plane;
            if (score == 0.0) {
                break;
            }
        }
    }
    if (!found) {
        return vec4<f32>(0.0);
    }
    // n · X + w = 0 is n · X = rho with rho = -w, the normal facing the
    // camera.
    var plane = vec4<f32>(best.xyz, -best.w);
    if (dot(plane.xyz, ray) > 0.0) {
        plane = -plane;
    }
    return plane;
}

// Hypotheses rank by cost, or by ACMP's posterior where the planar prior
// pass has a prior (`prior_depth > 0`, the prior's depth on the ray):
// larger is better either way.
fn hypothesis_score(
    cost: f32,
    plane: vec4<f32>,
    ray: vec3<f32>,
    prior_plane: vec4<f32>,
    prior_depth: f32,
) -> f32 {
    if (prior_depth <= 0.0) {
        return -cost;
    }
    let depth = plane_depth(plane, ray);
    if (depth <= 0.0) {
        return -3.0e38;
    }
    let dd = depth - prior_depth;
    let angle = acos(clamp(dot(plane.xyz, prior_plane.xyz), -1.0, 1.0));
    return exp(-cost * cost * params.prior_inv_alpha)
        * (params.prior_gamma
            + exp(-dd * dd * params.prior_depth_weight)
                * exp(-angle * angle * params.prior_angle_weight));
}

// A plane's cost in source view `j`: photometric plus geometric.
fn view_cost(j: u32, x: u32, y: u32, ray: vec3<f32>, plane: vec4<f32>) -> f32 {
    return photometric_cost(j, ray, plane) + geometric_cost(j, x, y, ray, plane);
}

// The eight sampling regions: V shapes (7 offsets) then long strips (11),
// up, down, left, right; offsets of the up shapes, rotated per direction.
fn region_offset(region: u32, k: u32) -> vec2<i32> {
    var up: vec2<i32>;
    if (region < 4u) {
        // (0,-1), (-1,-2), (1,-2), (-2,-3), (2,-3), (-3,-4), (3,-4)
        let level = i32((k + 1u) / 2u);
        var dx = level;
        if (k % 2u == 1u) {
            dx = -level;
        }
        up = vec2<i32>(dx, -(level + 1));
    } else {
        up = vec2<i32>(0, -(2 * i32(k) + 3));
    }
    switch (region % 4u) {
        case 0u: { return up; }
        case 1u: { return vec2<i32>(up.x, -up.y); }
        case 2u: { return vec2<i32>(up.y, up.x); }
        default: { return vec2<i32>(-up.y, up.x); }
    }
}

@compute @workgroup_size(16, 8, 1)
fn propagate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let y = gid.y;
    let x = 2u * gid.x + ((y + params.color) % 2u);
    if (x >= params.width || y >= params.height) {
        return;
    }
    let w = params.width;
    let h = params.height;
    let n = params.num_sources;
    let i = y * w + x;
    if (n == 0u) {
        costs[i] = MAX_COST;
        return;
    }
    let ray = pixel_ray(x, y);
    load_patch(x, y);
    // A weak pixel with anchors: matched with its deformable patch, its
    // candidates the anchors' planes (APD-MVS).
    anchor_count = 0u;
    if (params.deformable == 1u && anchors[i * MAX_ANCHORS] != NO_VIEW) {
        load_anchor_patches(x, y);
    }
    let weak = anchor_count > 0u;
    // The planar prior pass's prior here, and its depth on the ray.
    var prior_plane = vec4<f32>(0.0);
    var prior_depth = -1.0;
    if (params.planar_prior == 1u) {
        prior_plane = prior_planes[i];
        if (any(prior_plane.xyz != vec3<f32>(0.0))) {
            prior_depth = plane_depth(prior_plane, ray);
        }
    }

    // Adaptive checkerboard sampling.
    var cand: array<vec4<f32>, 9>;
    var valid: array<bool, 9>;
    for (var region = 0u; region < 8u; region++) {
        if (weak) {
            valid[region] = region < anchor_count;
            if (valid[region]) {
                cand[region] = planes[anchors[i * MAX_ANCHORS + region]];
            }
            continue;
        }
        var count = 7u;
        if (region >= 4u) {
            count = 11u;
        }
        var best_cost = 3.0e38;
        var best_index = NO_VIEW;
        for (var k = 0u; k < count; k++) {
            let o = region_offset(region, k);
            let sx = i32(x) + o.x;
            let sy = i32(y) + o.y;
            if (sx < 0 || sy < 0 || sx >= i32(w) || sy >= i32(h)) {
                continue;
            }
            let j = u32(sy) * w + u32(sx);
            if (best_index == NO_VIEW || costs[j] < best_cost) {
                best_cost = costs[j];
                best_index = j;
            }
        }
        valid[region] = best_index != NO_VIEW;
        if (valid[region]) {
            cand[region] = planes[best_index];
        }
    }
    let current = planes[i];
    cand[8] = current;
    valid[8] = true;

    // Multi-hypothesis joint view selection, accumulated view by view.
    let previous = best_views[i];
    let sampled = params.view_selection == 1u;
    var weights: array<f32, 32>;
    var agg_w: array<f32, 9>;
    var agg_u: array<f32, 9>;
    var weight_sum = 0.0;
    var new_best = NO_VIEW;
    var new_best_weight = 0.0;
    // Sampled weights: each view's probability before the prior.
    var probs: array<f32, 32>;
    let two_beta_sq = 2.0 * params.beta * params.beta;
    for (var j = 0u; j < n; j++) {
        var row_cost: array<f32, 8>;
        var good = 0u;
        var bad = 0u;
        var confidence = 0.0;
        for (var r = 0u; r < 8u; r++) {
            if (!valid[r]) {
                continue;
            }
            let m = photometric_cost(j, ray, cand[r]);
            if (!sampled) {
                row_cost[r] = m + geometric_cost(j, x, y, ray, cand[r]);
            }
            if (m < params.tau) {
                good += 1u;
                confidence += exp(-m * m / two_beta_sq);
            } else if (m > params.tau1) {
                bad += 1u;
            }
        }
        if (sampled) {
            if (good > params.n1 && bad < params.n2) {
                probs[j] = confidence / f32(good);
            } else if (bad < params.n2) {
                probs[j] = exp(-params.tau * params.tau / 0.32);
            }
            continue;
        }
        let selected = good > params.n1 && bad < params.n2;
        let is_previous = previous == j;
        var weight = 0.0;
        if (selected) {
            weight = confidence / f32(good);
            if (is_previous) {
                weight = 2.0 * weight;
            }
        } else if (is_previous) {
            weight = 0.2;
        }
        weights[j] = weight;
        weight_sum += weight;
        if (weight > new_best_weight) {
            new_best_weight = weight;
            new_best = j;
        }
        for (var r = 0u; r < 8u; r++) {
            if (valid[r]) {
                agg_w[r] += weight * row_cost[r];
                agg_u[r] += row_cost[r];
            }
        }
        // The current plane only counts in views with weight.
        if (weight > 0.0) {
            agg_w[8] += weight * view_cost(j, x, y, ray, current);
        }
    }
    var sampled_views = 0u;
    if (sampled) {
        // ACMM's: the probabilities times a prior from the four adjacent
        // pixels' selected views, then VIEW_SAMPLES draws; the candidates
        // are matched in the drawn views only.
        var prior: array<f32, 32>;
        for (var d = 0u; d < 4u; d++) {
            var o = vec2<i32>(0, -1);
            switch (d) {
                case 1u: { o = vec2<i32>(0, 1); }
                case 2u: { o = vec2<i32>(-1, 0); }
                case 3u: { o = vec2<i32>(1, 0); }
                default: {}
            }
            let sx = i32(x) + o.x;
            let sy = i32(y) + o.y;
            if (sx < 0 || sy < 0 || sx >= i32(w) || sy >= i32(h)) {
                continue;
            }
            let bits = best_views[u32(sy) * w + u32(sx)];
            for (var j = 0u; j < n; j++) {
                if (((bits >> j) & 1u) == 1u) {
                    prior[j] += 0.9;
                } else {
                    prior[j] += 0.1;
                }
            }
        }
        var cdf: array<f32, 32>;
        var total = 0.0;
        for (var j = 0u; j < n; j++) {
            total += probs[j] * prior[j];
            cdf[j] = total;
        }
        if (total > 0.0) {
            rng_init(x, y, SAMPLING_STREAM | (params.iteration * 2u + params.color));
            for (var sample = 0u; sample < VIEW_SAMPLES; sample++) {
                let u = uniform01() * total;
                for (var j = 0u; j < n; j++) {
                    if (cdf[j] > u) {
                        weights[j] += 1.0;
                        sampled_views |= 1u << j;
                        break;
                    }
                }
            }
        }
        for (var j = 0u; j < n; j++) {
            weight_sum += weights[j];
        }
        for (var j = 0u; j < n; j++) {
            if (weight_sum > 0.0 && weights[j] <= 0.0) {
                continue;
            }
            for (var r = 0u; r < 8u; r++) {
                if (valid[r]) {
                    let c = view_cost(j, x, y, ray, cand[r]);
                    agg_w[r] += weights[j] * c;
                    agg_u[r] += c;
                }
            }
            if (weight_sum > 0.0) {
                agg_w[8] += weights[j] * view_cost(j, x, y, ray, current);
            }
        }
    }
    let uniform_weights = weight_sum <= 0.0;
    if (uniform_weights) {
        weight_sum = f32(n);
        new_best = 0u;
        for (var j = 0u; j < n; j++) {
            agg_u[8] += view_cost(j, x, y, ray, current);
        }
    }
    var best_plane = current;
    var best_cost: f32;
    if (uniform_weights) {
        best_cost = agg_u[8] / weight_sum;
    } else {
        best_cost = agg_w[8] / weight_sum;
    }
    let current_cost = best_cost;
    var best_score = hypothesis_score(best_cost, best_plane, ray, prior_plane, prior_depth);
    var adopted = false;
    for (var r = 0u; r < 8u; r++) {
        if (!valid[r]) {
            continue;
        }
        var c: f32;
        if (uniform_weights) {
            c = agg_u[r] / weight_sum;
        } else {
            c = agg_w[r] / weight_sum;
        }
        let s = hypothesis_score(c, cand[r], ray, prior_plane, prior_depth);
        if (s > best_score && plane_depth(cand[r], ray) > 0.0) {
            best_cost = c;
            best_score = s;
            best_plane = cand[r];
            adopted = true;
        }
    }

    // Refinement; with a prior, the random depth and normal are drawn
    // around it.
    rng_init(x, y, params.iteration * 2u + params.color);
    var depth_now = plane_depth(best_plane, ray);
    if (depth_now <= 0.0) {
        depth_now = 0.5 * (params.depth_min + params.depth_max);
    }
    let normal_now = best_plane.xyz;
    var depth_random: f32;
    var normal_random: vec3<f32>;
    if (prior_depth > 0.0) {
        depth_random = prior_depth + (uniform01() - 0.5) * 6.0 * params.prior_depth_sigma;
        normal_random = perturb_normal(prior_plane.xyz, ray, params.prior_angle_sigma);
    } else {
        depth_random = params.depth_min + uniform01() * (params.depth_max - params.depth_min);
        normal_random = random_normal(ray);
    }
    let depth_perturbed = depth_now * (1.0 + (uniform01() - 0.5) * params.depth_perturbation);
    let normal_perturbed = perturb_normal(normal_now, ray, params.normal_perturbation);
    // A weak pixel first tries the plane fitted to its anchors' depths.
    if (weak) {
        draw_state = rng_state | 1u;
        let plane = fitted_plane(x, y, ray);
        let depth = plane_depth(plane, ray);
        if (any(plane.xyz != vec3<f32>(0.0)) && depth >= params.depth_min && depth <= params.depth_max) {
            var sum = 0.0;
            for (var j = 0u; j < n; j++) {
                if (uniform_weights) {
                    sum += view_cost(j, x, y, ray, plane);
                } else if (weights[j] > 0.0) {
                    sum += weights[j] * view_cost(j, x, y, ray, plane);
                }
            }
            let c = sum / weight_sum;
            let s = hypothesis_score(c, plane, ray, prior_plane, prior_depth);
            if (s > best_score) {
                best_cost = c;
                best_score = s;
                best_plane = plane;
            }
        }
    }
    for (var hyp = 0u; hyp < 6u; hyp++) {
        var depth = depth_now;
        var normal = normal_now;
        switch (hyp) {
            case 0u: { depth = depth_random; }
            case 1u: { normal = normal_random; }
            case 2u: { depth = depth_random; normal = normal_random; }
            case 3u: { depth = depth_perturbed; }
            case 4u: { normal = normal_perturbed; }
            default: { depth = depth_perturbed; normal = normal_perturbed; }
        }
        if (!(depth >= params.depth_min && depth <= params.depth_max)) {
            continue;
        }
        let plane = plane_from_depth(depth, normal, ray);
        var sum = 0.0;
        for (var j = 0u; j < n; j++) {
            if (uniform_weights) {
                sum += view_cost(j, x, y, ray, plane);
            } else if (weights[j] > 0.0) {
                sum += weights[j] * view_cost(j, x, y, ray, plane);
            }
        }
        let c = sum / weight_sum;
        let s = hypothesis_score(c, plane, ray, prior_plane, prior_depth);
        if (s > best_score) {
            best_cost = c;
            best_score = s;
            best_plane = plane;
        }
    }
    if (weak) {
        // A weak pixel keeps the plain cost, which its neighbours' sampling
        // compares. Under the detail restorer it takes a new plane only if
        // its deformable cost beats its start plane's by the margin.
        if (params.restore != 0u && best_cost >= current_cost - params.restore_margin) {
            best_plane = current;
        }
        anchor_count = 0u;
        var sum = 0.0;
        for (var j = 0u; j < n; j++) {
            if (uniform_weights) {
                sum += view_cost(j, x, y, ray, best_plane);
            } else if (weights[j] > 0.0) {
                sum += weights[j] * view_cost(j, x, y, ray, best_plane);
            }
        }
        best_cost = sum / weight_sum;
    } else if (params.restore != 0u && best_cost >= pre_costs[i] - params.restore_margin) {
        best_plane = current;
        best_cost = current_cost;
    }
    planes[i] = best_plane;
    costs[i] = best_cost;
    if (!sampled) {
        best_views[i] = new_best;
    } else if (adopted) {
        // As ACMM: a pixel's selection changes with its plane.
        best_views[i] = sampled_views;
    }
}
