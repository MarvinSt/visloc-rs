// Depth pass: per pixel, the median depth (camera z of the splat at which the
// transmittance first drops to 0.5 or below) and the final opacity 1 - T.
//
// Runs after `rasterize` on the same tile lists: `depths` holds the camera z
// of each compact id once the depth sort has run. A pixel whose transmittance
// never reaches 0.5 gets depth 0.

@group(0) @binding(0) var<uniform> u: RasterUniforms;
@group(0) @binding(1) var<storage, read> projected_splats: array<f32>;
@group(0) @binding(2) var<storage, read> compact_sorted: array<u32>;
@group(0) @binding(3) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> depths: array<f32>;
// Two floats per pixel: median depth, opacity.
@group(0) @binding(5) var<storage, read_write> out_depth: array<f32>;

const TILE_W: u32 = 16u;
const TILE_H: u32 = 16u;
const BATCH: u32 = 256u;

var<workgroup> batch: array<f32, BATCH * 7u>;
var<workgroup> done_count: atomic<u32>;
var<workgroup> all_done: u32;

@compute @workgroup_size(256)
fn rasterize_depth(
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

    var trans = 1.0;
    var median = 0.0;
    var found = false;

    let start = tile_offsets[tile * 2u];
    let end = tile_offsets[tile * 2u + 1u];

    if (tid == 0u) {
        atomicStore(&done_count, 0u);
    }
    workgroupBarrier();

    var batch_start = start;
    loop {
        if (batch_start >= end) { break; }
        let batch_end = min(batch_start + BATCH, end);
        let load_count = batch_end - batch_start;
        if (tid < load_count) {
            let cg = compact_sorted[batch_start + tid];
            let src = cg * 9u;
            for (var k = 0u; k < 6u; k = k + 1u) {
                batch[tid * 7u + k] = projected_splats[src + k];
            }
            batch[tid * 7u + 6u] = depths[cg];
        }
        workgroupBarrier();

        if (in_image && trans > 1e-4) {
            var s = 0u;
            loop {
                if (s >= load_count || trans <= 1e-4) { break; }
                let alpha_i = batch[s * 7u + 5u];
                let dx = sample_x - batch[s * 7u + 0u];
                let dy = sample_y - batch[s * 7u + 1u];
                let c00 = batch[s * 7u + 2u];
                let c01 = batch[s * 7u + 3u];
                let c11 = batch[s * 7u + 4u];
                let sigma = 0.5 * (c00 * dx * dx + c11 * dy * dy) + c01 * dx * dy;
                if (sigma >= 0.0) {
                    let a = min(0.99, alpha_i * exp(-sigma));
                    if (a >= 1.0 / 255.0) {
                        trans = trans * (1.0 - a);
                        if (!found && trans <= 0.5) {
                            median = batch[s * 7u + 6u];
                            found = true;
                        }
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
        if (workgroupUniformLoad(&all_done) == 1u) { break; }
        batch_start = batch_end;
    }

    if (in_image) {
        let pix = py * u.img_w + px;
        out_depth[pix * 2u] = median;
        out_depth[pix * 2u + 1u] = 1.0 - trans;
    }
}
