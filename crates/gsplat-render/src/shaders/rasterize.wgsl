// Pass 5: alpha-composite the tile-sorted gaussian list for each tile.
//
// One workgroup per tile; thread `tid` owns pixel `(tid % 16, tid / 16)`.
// Splats are processed in batches of 256 so the whole batch can be staged in
// shared memory before each pixel walks it (front-to-back). Once every pixel in
// the tile is saturated (transmittance <= 1e-4) the workgroup stops loading
// batches, so a dense tile costs only as many batches as it takes to go opaque.

@group(0) @binding(0) var<uniform> u: RasterUniforms;
@group(0) @binding(1) var<storage, read> projected_splats: array<f32>;
// Compact gaussian id of each tile-sorted list entry.
@group(0) @binding(2) var<storage, read> compact_sorted: array<u32>;
@group(0) @binding(3) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_img: array<f32>;
@group(0) @binding(5) var<storage, read> global_from_compact: array<u32>;
// Residuals for the backward pass, per pixel: final transmittance, and one
// past the list index of the last splat that was blended (0 = none).
@group(0) @binding(6) var<storage, read_write> final_t: array<f32>;
@group(0) @binding(7) var<storage, read_write> last_idx: array<u32>;
// #if GEO
// Geometry variant: per compact id (normal xyz, depth), and per pixel their
// alpha-weighted sums (no background).
@group(0) @binding(8) var<storage, read> geo_splats: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> out_geo: array<vec4<f32>>;
// #endif

const TILE_W: u32 = 16u;
const TILE_H: u32 = 16u;
const BATCH: u32 = 256u;

var<workgroup> batch: array<f32, BATCH * 9u>;
// #if GEO
var<workgroup> batch_geo: array<vec4<f32>, BATCH>;
// #endif
// Pixels in this tile that are finished (saturated or outside the image).
var<workgroup> done_count: atomic<u32>;
// 1 once every pixel is finished; read back uniformly to break the batch loop.
var<workgroup> all_done: u32;

@compute @workgroup_size(256)
fn rasterize(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
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

    var accum = vec3<f32>(0.0, 0.0, 0.0);
// #if GEO
    var accum_geo = vec4<f32>(0.0);
// #endif
    var trans = 1.0;
    var last = 0u;

    let start = tile_offsets[tile * 2u];
    let end = tile_offsets[tile * 2u + 1u];

    if (tid == 0u) {
        atomicStore(&done_count, 0u);
    }
    workgroupBarrier();

    var batch_start = start;
    loop {
        if (batch_start >= end) { break; }
        // Stage one batch into shared memory.
        let batch_end = min(batch_start + BATCH, end);
        let load_count = batch_end - batch_start;
        for (var s = tid; s < BATCH; s = s + 256u) {
            if (s < load_count) {
                let cg = compact_sorted[batch_start + s];
                let src = cg * 9u;
                for (var k = 0u; k < 9u; k = k + 1u) {
                    batch[s * 9u + k] = projected_splats[src + k];
                }
// #if GEO
                batch_geo[s] = geo_splats[cg];
// #endif
            }
        }
        workgroupBarrier();

        if (in_image && trans > 1e-4) {
            var s = 0u;
            loop {
                if (s >= load_count || trans <= 1e-4) { break; }
                let alpha_i = batch[s * 9u + 5u];
                let dx = sample_x - batch[s * 9u + 0u];
                let dy = sample_y - batch[s * 9u + 1u];
                let c00 = batch[s * 9u + 2u];
                let c01 = batch[s * 9u + 3u];
                let c11 = batch[s * 9u + 4u];
                let sigma = 0.5 * (c00 * dx * dx + c11 * dy * dy) + c01 * dx * dy;
                if (sigma >= 0.0) {
                    let a = min(0.99, alpha_i * exp(-sigma));
                    if (a >= 1.0 / 255.0) {
                        let cr = max(batch[s * 9u + 6u], 0.0);
                        let cg = max(batch[s * 9u + 7u], 0.0);
                        let cb = max(batch[s * 9u + 8u], 0.0);
                        accum = accum + trans * a * vec3<f32>(cr, cg, cb);
// #if GEO
                        accum_geo = accum_geo + trans * a * batch_geo[s];
// #endif
                        trans = trans * (1.0 - a);
                        last = batch_start + s + 1u;
                    }
                }
                s = s + 1u;
            }
        }
        if (!in_image || trans <= 1e-4) {
            atomicAdd(&done_count, 1u);
        }
        workgroupBarrier();
        if (tid == 0u) {
            all_done = select(0u, 1u, atomicLoad(&done_count) == 256u);
            atomicStore(&done_count, 0u);
        }
        // Also the barrier that protects `batch` before the next load.
        if (workgroupUniformLoad(&all_done) == 1u) { break; }
        batch_start = batch_end;
    }

    if (in_image && tid < TILE_W * TILE_H) {
        let idx = (py * u.img_w + px) * 3u;
        out_img[idx + 0u] = accum.r + trans * u.bg_r;
        out_img[idx + 1u] = accum.g + trans * u.bg_g;
        out_img[idx + 2u] = accum.b + trans * u.bg_b;
        let pix = py * u.img_w + px;
        final_t[pix] = trans;
        last_idx[pix] = last;
// #if GEO
        out_geo[pix] = accum_geo;
// #endif
    }
}
