// Backward of `rasterize`: per-(tile, splat) gradients of the screen-space
// splat parameters for L = sum_px dot(d_image[px], C[px]).
//
// One workgroup per tile, one thread per pixel, walking the tile's list back
// to front from the tile's last blended entry. Each pixel recovers the
// transmittance in front of a splat as T_before = T_after / (1 - alpha) from
// the forward residuals (final transmittance, last blended index) and keeps
// the colour composited behind it, so
//   dC/dcolour_i = alpha_i T_i,  dC/dalpha_i = T_i (colour_i - behind_i).
// alpha = min(0.99, opacity exp(-sigma)), sigma = 0.5 (A dx^2 + C dy^2) + B dx dy
// with d = pixel centre - mean; a clamped alpha passes no gradient to opacity,
// conic or mean.
//
// The 256 per-pixel contributions of a splat are summed within each subgroup
// (a transposed xor butterfly on 32-wide subgroups, subgroupAdd otherwise)
// and a shared-memory float add (CAS on u32 bits) across subgroups; the tile's
// total is then added to the gaussian's record in screen_grads with one
// global CAS float add per component (no per-intersection storage).
//
// Record layout (10 floats, per compact id): du, dv, dA, dB, dC, dopacity,
// dr, dg, db.

@group(0) @binding(0) var<uniform> u: RasterUniforms;
@group(0) @binding(1) var<storage, read> projected_splats: array<f32>;
@group(0) @binding(2) var<storage, read> compact_sorted: array<u32>;
@group(0) @binding(3) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(5) var<storage, read> final_t: array<f32>;
@group(0) @binding(6) var<storage, read> last_idx: array<u32>;
@group(0) @binding(7) var<storage, read> d_image: array<f32>;
// f32 bits, accumulated with CAS (no float atomics in wgpu here).
@group(0) @binding(8) var<storage, read_write> screen_grads: array<atomic<u32>>;
// #if GEO
// Geometry variant: the normal / depth channels (see rasterize.wgsl) with
// their per-pixel gradient, and per compact id gradients of (normal, depth).
@group(0) @binding(9) var<storage, read> geo_splats: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> d_geo: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read_write> geo_grads: array<atomic<u32>>;
// #endif

const TILE_W: u32 = 16u;
const TILE_H: u32 = 16u;
const BB: u32 = 128u;
// du dv dA dB dC dopacity dr dg db, and the refine weight (brush / AbsGS:
// sum over pixels of |dL/du| W + |dL/dv| H).
const NG: u32 = 10u;

var<workgroup> bsplat: array<f32, BB * 9u>;
var<workgroup> bcompact: array<u32, BB>;
var<workgroup> gacc: array<atomic<u32>, BB * NG>;
var<workgroup> tile_last: atomic<u32>;
// #if GEO
var<workgroup> bgeo: array<vec4<f32>, BB>;
var<workgroup> gacc_geo: array<atomic<u32>, BB * 4u>;
// #endif
var<workgroup> tile_last_u: u32;

fn gacc_add(i: u32, v: f32) {
    var old = atomicLoad(&gacc[i]);
    loop {
        let r = atomicCompareExchangeWeak(&gacc[i], old, bitcast<u32>(bitcast<f32>(old) + v));
        if (r.exchanged) {
            break;
        }
        old = r.old_value;
    }
}

// One butterfly step: keep `lo` or `hi` by this lane's bit, add the partner's
// other half.
fn xstep(lo: f32, hi: f32, h: bool, m: u32) -> f32 {
    return select(lo, hi, h) + subgroupShuffleXor(select(hi, lo, h), m);
}

// #if GEO
fn gacc_geo_add(i: u32, v: f32) {
    var old = atomicLoad(&gacc_geo[i]);
    loop {
        let r = atomicCompareExchangeWeak(&gacc_geo[i], old, bitcast<u32>(bitcast<f32>(old) + v));
        if (r.exchanged) {
            break;
        }
        old = r.old_value;
    }
}

fn geo_global_add(i: u32, v: f32) {
    var old = atomicLoad(&geo_grads[i]);
    loop {
        let r = atomicCompareExchangeWeak(&geo_grads[i], old, bitcast<u32>(bitcast<f32>(old) + v));
        if (r.exchanged) {
            break;
        }
        old = r.old_value;
    }
}
// #endif

fn global_add(i: u32, v: f32) {
    var old = atomicLoad(&screen_grads[i]);
    loop {
        let r = atomicCompareExchangeWeak(&screen_grads[i], old, bitcast<u32>(bitcast<f32>(old) + v));
        if (r.exchanged) {
            break;
        }
        old = r.old_value;
    }
}

@compute @workgroup_size(256)
fn rasterize_backward(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let tile = wid.x + wid.y * nwg.x;
    if (tile >= u.tile_bw * u.tile_bh) { return; }
    let tid = lid.x;
    let tile_x = tile % u.tile_bw;
    let tile_y = tile / u.tile_bw;
    let px = tile_x * TILE_W + (tid % TILE_W);
    let py = tile_y * TILE_H + (tid / TILE_W);
    let sample_x = f32(px) + 0.5;
    let sample_y = f32(py) + 0.5;
    let in_image = px < u.img_w && py < u.img_h;

    let start = tile_offsets[tile * 2u];
    let end = tile_offsets[tile * 2u + 1u];

    var t_cur = 1.0;
    var last = 0u;
    var g = vec3<f32>(0.0, 0.0, 0.0);
    var behind = vec3<f32>(u.bg_r, u.bg_g, u.bg_b);
// #if GEO
    var gg = vec4<f32>(0.0);
    var behind_geo = vec4<f32>(0.0);
// #endif
    if (in_image) {
        let pix = py * u.img_w + px;
        t_cur = final_t[pix];
        last = last_idx[pix];
        g = vec3<f32>(d_image[pix * 3u], d_image[pix * 3u + 1u], d_image[pix * 3u + 2u]);
// #if GEO
        gg = d_geo[pix];
// #endif
    }
    if (tid == 0u) {
        atomicStore(&tile_last, 0u);
    }
    workgroupBarrier();
    atomicMax(&tile_last, last);
    workgroupBarrier();
    if (tid == 0u) {
        tile_last_u = atomicLoad(&tile_last);
    }
    // Empty tiles have start = u32::MAX, end = 0; their pixels have last = 0.
    let stop = min(workgroupUniformLoad(&tile_last_u), end);

    var bend = stop;
    loop {
        if (bend <= start || bend == 0u) { break; }
        var bstart = start;
        if (bend - start > BB) {
            bstart = bend - BB;
        }
        let cnt = bend - bstart;
        if (tid < cnt) {
            let j = bstart + tid;
            let cg = compact_sorted[j];
            let src = cg * 9u;
            for (var k = 0u; k < 9u; k = k + 1u) {
                bsplat[tid * 9u + k] = projected_splats[src + k];
            }
            bcompact[tid] = cg;
// #if GEO
            bgeo[tid] = geo_splats[cg];
// #endif
        }
        for (var i = tid; i < BB * NG; i = i + 256u) {
            atomicStore(&gacc[i], 0u);
        }
// #if GEO
        for (var i = tid; i < BB * 4u; i = i + 256u) {
            atomicStore(&gacc_geo[i], 0u);
        }
// #endif
        workgroupBarrier();

        var s = cnt;
        loop {
            if (s == 0u) { break; }
            s = s - 1u;
            let j = bstart + s;
            // Per-lane gradient as scalars (a runtime-indexed array would be
            // spilled to local memory).
            var g_u = 0.0;
            var g_v = 0.0;
            var g_a = 0.0;
            var g_b = 0.0;
            var g_c = 0.0;
            var g_o = 0.0;
            var g_col = vec3<f32>(0.0, 0.0, 0.0);
            var g_r = 0.0;
            // Normal xyz / depth gradient (geometry variant only).
            var g_geo = vec4<f32>(0.0);
            var hit = false;
            if (in_image && j < last) {
                let opac = bsplat[s * 9u + 5u];
                let dx = sample_x - bsplat[s * 9u + 0u];
                let dy = sample_y - bsplat[s * 9u + 1u];
                let c00 = bsplat[s * 9u + 2u];
                let c01 = bsplat[s * 9u + 3u];
                let c11 = bsplat[s * 9u + 4u];
                // Same expressions as the forward, so the same splats pass.
                let sigma = 0.5 * (c00 * dx * dx + c11 * dy * dy) + c01 * dx * dy;
                if (sigma >= 0.0) {
                    let e = exp(-sigma);
                    let raw = opac * e;
                    let a = min(0.99, raw);
                    if (a >= 1.0 / 255.0) {
                        hit = true;
                        let t_before = t_cur / (1.0 - a);
                        let col = max(
                            vec3<f32>(bsplat[s * 9u + 6u], bsplat[s * 9u + 7u], bsplat[s * 9u + 8u]),
                            vec3<f32>(0.0, 0.0, 0.0),
                        );
                        g_col = a * t_before * g;
                        var d_alpha = t_before * dot(col - behind, g);
                        behind = a * col + (1.0 - a) * behind;
// #if GEO
                        let geo = bgeo[s];
                        g_geo = a * t_before * gg;
                        d_alpha = d_alpha + t_before * dot(geo - behind_geo, gg);
                        behind_geo = a * geo + (1.0 - a) * behind_geo;
// #endif
                        t_cur = t_before;
                        if (raw <= 0.99) {
                            let d_sigma = -d_alpha * a;
                            g_u = d_sigma * -(c00 * dx + c01 * dy);
                            g_v = d_sigma * -(c01 * dx + c11 * dy);
                            g_a = d_sigma * 0.5 * dx * dx;
                            g_b = d_sigma * dx * dy;
                            g_c = d_sigma * 0.5 * dy * dy;
                            g_o = d_alpha * e;
                            g_r = abs(g_u) * f32(u.img_w) + abs(g_v) * f32(u.img_h);
                        }
                    }
                }
            }
            // Most splats cover only part of a tile: skip the reductions when
            // no lane of this subgroup blended it. The condition is
            // subgroup-uniform, so the subgroup ops stay in uniform flow.
            if (subgroupAny(hit)) {
                if (sg_size == 32u) {
                    // Transposed butterfly: each xor step halves the values a
                    // lane carries, so the 10 (padded to 16) sums take
                    // 8+4+2+1+1 shuffles instead of 10 full reductions, and
                    // lanes 2c (c < 10) end up holding component c.
                    let h4 = (lane & 16u) != 0u;
                    let w0 = xstep(g_u, g_col.z, h4, 16u);
                    let w1 = xstep(g_v, g_r, h4, 16u);
                    let w2 = xstep(g_a, g_geo.x, h4, 16u);
                    let w3 = xstep(g_b, g_geo.y, h4, 16u);
                    let w4 = xstep(g_c, g_geo.z, h4, 16u);
                    let w5 = xstep(g_o, g_geo.w, h4, 16u);
                    let w6 = xstep(g_col.x, 0.0, h4, 16u);
                    let w7 = xstep(g_col.y, 0.0, h4, 16u);
                    let h3 = (lane & 8u) != 0u;
                    let x0 = xstep(w0, w4, h3, 8u);
                    let x1 = xstep(w1, w5, h3, 8u);
                    let x2 = xstep(w2, w6, h3, 8u);
                    let x3 = xstep(w3, w7, h3, 8u);
                    let h2 = (lane & 4u) != 0u;
                    let y0 = xstep(x0, x2, h2, 4u);
                    let y1 = xstep(x1, x3, h2, 4u);
                    var z = xstep(y0, y1, (lane & 2u) != 0u, 2u);
                    z = z + subgroupShuffleXor(z, 1u);
                    let comp = lane >> 1u;
                    if ((lane & 1u) == 0u && comp < NG && z != 0.0) {
                        gacc_add(s * NG + comp, z);
                    }
// #if GEO
                    if ((lane & 1u) == 0u && comp >= NG && comp < NG + 4u && z != 0.0) {
                        gacc_geo_add(s * 4u + comp - NG, z);
                    }
// #endif
                } else {
                    let t_uv = subgroupAdd(vec2<f32>(g_u, g_v));
                    let t_abc = subgroupAdd(vec3<f32>(g_a, g_b, g_c));
                    let t_o = subgroupAdd(g_o);
                    let t_col = subgroupAdd(g_col);
                    let t_r = subgroupAdd(g_r);
// #if GEO
                    let t_geo = subgroupAdd(g_geo);
                    if (lane == 0u) {
                        gacc_geo_add(s * 4u + 0u, t_geo.x);
                        gacc_geo_add(s * 4u + 1u, t_geo.y);
                        gacc_geo_add(s * 4u + 2u, t_geo.z);
                        gacc_geo_add(s * 4u + 3u, t_geo.w);
                    }
// #endif
                    if (lane == 0u) {
                        let base = s * NG;
                        gacc_add(base + 0u, t_uv.x);
                        gacc_add(base + 1u, t_uv.y);
                        gacc_add(base + 2u, t_abc.x);
                        gacc_add(base + 3u, t_abc.y);
                        gacc_add(base + 4u, t_abc.z);
                        gacc_add(base + 5u, t_o);
                        gacc_add(base + 6u, t_col.x);
                        gacc_add(base + 7u, t_col.y);
                        gacc_add(base + 8u, t_col.z);
                        gacc_add(base + 9u, t_r);
                    }
                }
            }
        }
        workgroupBarrier();
        for (var i = tid; i < cnt * NG; i = i + 256u) {
            let v = bitcast<f32>(atomicLoad(&gacc[i]));
            if (v != 0.0) {
                global_add(bcompact[i / NG] * NG + i % NG, v);
            }
        }
// #if GEO
        for (var i = tid; i < cnt * 4u; i = i + 256u) {
            let v = bitcast<f32>(atomicLoad(&gacc_geo[i]));
            if (v != 0.0) {
                geo_global_add(bcompact[i / 4u] * 4u + i % 4u, v);
            }
        }
// #endif
        // bsplat / bcompact / gacc are reused by the next batch.
        workgroupBarrier();
        bend = bstart;
    }
}
