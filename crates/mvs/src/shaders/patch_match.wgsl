// ACMH/ACMM PatchMatch on the GPU: the per-pixel update of
// `patch_match.rs`, one invocation per pixel. `init` gives every pixel its
// first plane and cost; `propagate` updates the pixels of one checkerboard
// colour from the other's planes. Planes are `(n.xyz, rho)` in the
// reference camera frame, `rho = n · X`.

const MAX_COST: f32 = 2.0;
const MAX_SAMPLES: u32 = 64u;
const MAX_SOURCES: u32 = 32u;
const NO_VIEW: u32 = 0xffffffffu;

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
}

// Reference camera frame to source camera frame: rows of R with t in w.
struct Source {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
    // fx, fy, cx, cy
    k: vec4<f32>,
    // width, height, image offset, prior depth offset
    dims: vec4<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sources: array<Source>;
// The reference image at offset 0, then each source's.
@group(0) @binding(2) var<storage, read> images: array<f32>;
// Each source's earlier depth map (geometric mode).
@group(0) @binding(3) var<storage, read> priors: array<f32>;
@group(0) @binding(4) var<storage, read_write> planes: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> costs: array<f32>;
@group(0) @binding(6) var<storage, read_write> best_views: array<u32>;
// `init`: the earlier map's (normal, depth) per pixel, depth 0 for none.
@group(0) @binding(7) var<storage, read> init_planes: array<vec4<f32>>;

// The reference window of the current pixel; sample k's offset is
// `window_offset(k)`.
var<private> patch_values: array<f32, 64>;
var<private> patch_weights: array<f32, 64>;
var<private> patch_n: u32;
var<private> window_cols: u32;
var<private> patch_center: f32;
var<private> patch_sum_w: f32;
var<private> patch_mean: f32;
var<private> patch_var: f32;
var<private> rng_state: u32;

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

fn window_offset(k: u32) -> vec2<f32> {
    let r = params.window_radius;
    let step = max(params.window_step, 1);
    return vec2<f32>(
        f32(-r + i32(k % window_cols) * step),
        f32(-r + i32(k / window_cols) * step),
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
    window_cols = u32(2 * r / step) + 1u;
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
    patch_mean = sum / sum_w;
    patch_var = sum_sq / sum_w - patch_mean * patch_mean;
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

fn source_gray(s: Source, u: f32, v: f32) -> f32 {
    let w = s.dims.x;
    let h = s.dims.y;
    let x = clamp(u - 0.5, 0.0, f32(w - 1u));
    let y = clamp(v - 0.5, 0.0, f32(h - 1u));
    let x0 = u32(x);
    let y0 = u32(y);
    let x1 = min(x0 + 1u, w - 1u);
    let y1 = min(y0 + 1u, h - 1u);
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    let base = s.dims.z;
    let top = images[base + y0 * w + x0] * (1.0 - fx) + images[base + y0 * w + x1] * fx;
    let bottom = images[base + y1 * w + x0] * (1.0 - fx) + images[base + y1 * w + x1] * fx;
    return top * (1.0 - fy) + bottom * fy;
}

fn photometric_cost(j: u32, ray: vec3<f32>, plane: vec4<f32>) -> f32 {
    if (patch_var < 1e-5 || abs(plane.w) < 1e-12) {
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
    var s_w = 0.0;
    var s_ww = 0.0;
    var s_rw = 0.0;
    for (var k = 0u; k < patch_n; k++) {
        let o = window_offset(k);
        let p = a + o.x * bx + o.y * by;
        if (p.z <= 0.0) {
            return MAX_COST;
        }
        let u = s.k.x * p.x / p.z + s.k.z;
        let v = s.k.y * p.y / p.z + s.k.w;
        let value = source_gray(s, u, v) - patch_center;
        let weight = patch_weights[k];
        s_w += weight * value;
        s_ww += weight * value * value;
        s_rw += weight * patch_values[k] * value;
    }
    let mean = s_w / patch_sum_w;
    let var_s = s_ww / patch_sum_w - mean * mean;
    if (var_s < 1e-5) {
        return MAX_COST;
    }
    let cov = s_rw / patch_sum_w - patch_mean * mean;
    let ncc = cov / sqrt(patch_var * var_s);
    return clamp(1.0 - ncc, 0.0, MAX_COST);
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

// The mean of the best `init_top_k` per-view costs of a plane.
fn top_k_cost(x: u32, y: u32, ray: vec3<f32>, plane: vec4<f32>) -> f32 {
    let n = params.num_sources;
    if (n == 0u) {
        return MAX_COST;
    }
    let k = clamp(params.init_top_k, 1u, min(n, 8u));
    var best: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) {
        best[i] = 3.0e38;
    }
    for (var j = 0u; j < n; j++) {
        var c = photometric_cost(j, ray, plane) + geometric_cost(j, x, y, ray, plane);
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
    best_views[i] = NO_VIEW;
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
        best_views[i] = NO_VIEW;
        return;
    }
    let ray = pixel_ray(x, y);
    load_patch(x, y);

    // Adaptive checkerboard sampling.
    var cand: array<vec4<f32>, 9>;
    var valid: array<bool, 9>;
    for (var region = 0u; region < 8u; region++) {
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
    var weights: array<f32, 32>;
    var agg_w: array<f32, 9>;
    var agg_u: array<f32, 9>;
    var weight_sum = 0.0;
    var new_best = NO_VIEW;
    var new_best_weight = 0.0;
    let two_beta_sq = 2.0 * params.beta * params.beta;
    for (var j = 0u; j < n; j++) {
        var row_cost: array<f32, 9>;
        var good = 0u;
        var bad = 0u;
        var confidence = 0.0;
        for (var r = 0u; r < 9u; r++) {
            if (!valid[r]) {
                continue;
            }
            let m = photometric_cost(j, ray, cand[r]);
            row_cost[r] = m + geometric_cost(j, x, y, ray, cand[r]);
            if (r < 8u) {
                if (m < params.tau) {
                    good += 1u;
                    confidence += exp(-m * m / two_beta_sq);
                } else if (m > params.tau1) {
                    bad += 1u;
                }
            }
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
        for (var r = 0u; r < 9u; r++) {
            if (valid[r]) {
                agg_w[r] += weight * row_cost[r];
                agg_u[r] += row_cost[r];
            }
        }
    }
    let uniform_weights = weight_sum <= 0.0;
    if (uniform_weights) {
        weight_sum = f32(n);
        new_best = 0u;
    }
    var best_plane = current;
    var best_cost: f32;
    if (uniform_weights) {
        best_cost = agg_u[8] / weight_sum;
    } else {
        best_cost = agg_w[8] / weight_sum;
    }
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
        if (c < best_cost && plane_depth(cand[r], ray) > 0.0) {
            best_cost = c;
            best_plane = cand[r];
        }
    }

    // Refinement.
    rng_init(x, y, params.iteration * 2u + params.color);
    var depth_now = plane_depth(best_plane, ray);
    if (depth_now <= 0.0) {
        depth_now = 0.5 * (params.depth_min + params.depth_max);
    }
    let normal_now = best_plane.xyz;
    let depth_random = params.depth_min + uniform01() * (params.depth_max - params.depth_min);
    let normal_random = random_normal(ray);
    let depth_perturbed = depth_now * (1.0 + (uniform01() - 0.5) * params.depth_perturbation);
    let normal_perturbed = perturb_normal(normal_now, ray, params.normal_perturbation);
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
            let total = photometric_cost(j, ray, plane) + geometric_cost(j, x, y, ray, plane);
            if (uniform_weights) {
                sum += total;
            } else {
                sum += weights[j] * total;
            }
        }
        let c = sum / weight_sum;
        if (c < best_cost) {
            best_cost = c;
            best_plane = plane;
        }
    }
    planes[i] = best_plane;
    costs[i] = best_cost;
    best_views[i] = new_best;
}
