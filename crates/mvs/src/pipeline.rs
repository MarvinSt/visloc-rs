//! ACMM's multi-scale depth estimation: ACMH on an image pyramid, coarse to
//! fine, with geometric consistency at every scale.
//!
//! At the coarsest scale every view gets a photometric pass, then
//! `geom_passes` geometric ones. Each finer scale starts from the coarser
//! maps, upsampled by joint bilateral upsampling guided by the finer image.
//! The detail restorer then runs a photometric pass at the finer scale and
//! keeps its estimate wherever the upsampled plane's photometric cost is
//! worse by more than `detail_threshold` (upsampling blurs thin structures
//! and boundaries), before the geometric passes at that scale.

use rayon::prelude::*;

use crate::patch_match::{photometric_score, DepthMap, PatchMatchOptions};
use crate::workspace::{View, Workspace};

/// Multi-scale estimation parameters (ACMM's `k`, `eta`, `xi` and two
/// geometric passes per scale by default).
#[derive(Debug, Clone)]
pub struct MultiScaleOptions {
    pub patch_match: PatchMatchOptions,
    /// Pyramid levels, each half the size of the next.
    pub scales: usize,
    /// Geometric passes per scale.
    pub geom_passes: usize,
    /// The detail restorer's cost margin.
    pub detail_threshold: f32,
}

impl Default for MultiScaleOptions {
    fn default() -> Self {
        Self {
            patch_match: PatchMatchOptions::default(),
            scales: 3,
            geom_passes: 2,
            detail_threshold: 0.1,
        }
    }
}

/// One view's depth map: [`crate::patch_match::estimate`] or the GPU's
/// equivalent.
pub type Estimator<'a> =
    dyn Fn(&Workspace, usize, &PatchMatchOptions, Option<&[DepthMap]>) -> DepthMap + 'a;

/// Depth maps of every view of `ws` (at its resolution), estimated coarse
/// to fine by `estimate`. `log` receives a line per stage.
pub fn multi_scale_depth_maps(
    ws: &Workspace,
    options: &MultiScaleOptions,
    estimate: &Estimator<'_>,
    log: &mut dyn FnMut(&str),
) -> Vec<DepthMap> {
    let scales = options.scales.max(1);
    // levels[0] is the coarsest.
    let mut levels = vec![ws.clone()];
    for _ in 1..scales {
        let finer = levels.last().expect("one level");
        levels.push(downscale(finer));
    }
    levels.reverse();
    let pm = &options.patch_match;
    let all = |ws: &Workspace, prior: Option<&[DepthMap]>| -> Vec<DepthMap> {
        (0..ws.views.len())
            .map(|v| estimate(ws, v, pm, prior))
            .collect()
    };

    let coarsest = &levels[0];
    let started = std::time::Instant::now();
    let mut maps = all(coarsest, None);
    log(&format!(
        "scale 0 ({}x{}): photometric {:.1} s",
        coarsest.views[0].width,
        coarsest.views[0].height,
        started.elapsed().as_secs_f64()
    ));
    for pass in 0..options.geom_passes {
        let t = std::time::Instant::now();
        maps = all(coarsest, Some(&maps));
        log(&format!(
            "scale 0: geometric {} {:.1} s",
            pass + 1,
            t.elapsed().as_secs_f64()
        ));
    }
    for level in 1..scales {
        let coarse = &levels[level - 1];
        let fine = &levels[level];
        let t = std::time::Instant::now();
        let upsampled: Vec<DepthMap> = (0..fine.views.len())
            .into_par_iter()
            .map(|v| upsample(&maps[v], &coarse.views[v], &fine.views[v]))
            .collect();
        let photometric = all(fine, None);
        let mut replaced = 0usize;
        let mut total = 0usize;
        let restored: Vec<DepthMap> = (0..fine.views.len())
            .map(|v| {
                let (map, count) = restore_details(
                    fine,
                    v,
                    pm,
                    &upsampled[v],
                    &photometric[v],
                    options.detail_threshold,
                );
                replaced += count;
                total += map.depth.len();
                map
            })
            .collect();
        log(&format!(
            "scale {level} ({}x{}): upsampled, photometric and restored {:.1}% of pixels in {:.1} s",
            fine.views[0].width,
            fine.views[0].height,
            100.0 * replaced as f64 / total.max(1) as f64,
            t.elapsed().as_secs_f64()
        ));
        maps = restored;
        for pass in 0..options.geom_passes {
            let t = std::time::Instant::now();
            maps = all(fine, Some(&maps));
            log(&format!(
                "scale {level}: geometric {} {:.1} s",
                pass + 1,
                t.elapsed().as_secs_f64()
            ));
        }
    }
    maps
}

/// The workspace at half the size (images by 2×2 box averages, odd
/// trailing rows and columns dropped; intrinsics scaled to match).
pub fn downscale(ws: &Workspace) -> Workspace {
    let views = ws
        .views
        .par_iter()
        .map(|view| {
            let (w, h) = ((view.width / 2).max(1), (view.height / 2).max(1));
            let mut gray = vec![0.0f32; w * h];
            let mut rgb = vec![[0u8; 3]; w * h];
            for y in 0..h {
                for x in 0..w {
                    let mut g = 0.0f32;
                    let mut c = [0u32; 3];
                    for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        let sx = (2 * x + dx).min(view.width - 1);
                        let sy = (2 * y + dy).min(view.height - 1);
                        let i = sy * view.width + sx;
                        g += view.gray[i];
                        for (acc, v) in c.iter_mut().zip(view.rgb[i]) {
                            *acc += u32::from(v);
                        }
                    }
                    gray[y * w + x] = 0.25 * g;
                    rgb[y * w + x] = c.map(|v| ((v + 2) / 4) as u8);
                }
            }
            let (sx, sy) = (w as f64 / view.width as f64, h as f64 / view.height as f64);
            View {
                width: w,
                height: h,
                fx: view.fx * sx,
                fy: view.fy * sy,
                cx: view.cx * sx,
                cy: view.cy * sy,
                gray,
                rgb,
                ..view.clone()
            }
        })
        .collect();
    Workspace { views }
}

/// Joint bilateral upsampling of `map` (of `coarse`) to `fine`'s size: each
/// fine pixel averages the depths (evaluated on its own ray from the coarse
/// pixels' planes) and normals of the 5×5 coarse pixels around it, weighted
/// by distance in coarse pixels and by the grey-level difference in the
/// fine image.
pub fn upsample(map: &DepthMap, coarse: &View, fine: &View) -> DepthMap {
    const RADIUS: i32 = 2;
    const SIGMA_SPATIAL: f32 = 1.0;
    const SIGMA_COLOR: f32 = 10.0;
    let (fw, fh) = (fine.width, fine.height);
    let (sx, sy) = (
        coarse.width as f32 / fw as f32,
        coarse.height as f32 / fh as f32,
    );
    let ray = |view: &View, x: f32, y: f32| -> [f32; 3] {
        [
            (x - view.cx as f32) / view.fx as f32,
            (y - view.cy as f32) / view.fy as f32,
            1.0,
        ]
    };
    // Per row: depths, normals, costs.
    type Row = (Vec<f32>, Vec<[f32; 3]>, Vec<f32>);
    let rows: Vec<Row> = (0..fh)
        .into_par_iter()
        .map(|y| {
            let mut depth = vec![0.0f32; fw];
            let mut normal = vec![[0.0f32; 3]; fw];
            let mut cost = vec![2.0f32; fw];
            for x in 0..fw {
                let fine_ray = ray(fine, x as f32 + 0.5, y as f32 + 0.5);
                let gray = fine.gray[y * fw + x];
                // The fine pixel's centre in coarse pixel coordinates.
                let (cxf, cyf) = ((x as f32 + 0.5) * sx - 0.5, (y as f32 + 0.5) * sy - 0.5);
                let (ci, cj) = (cxf.round() as i32, cyf.round() as i32);
                let (mut sum_w, mut sum_d, mut sum_c) = (0.0f32, 0.0f32, 0.0f32);
                let mut sum_n = [0.0f32; 3];
                for dj in -RADIUS..=RADIUS {
                    for di in -RADIUS..=RADIUS {
                        let (qi, qj) = (ci + di, cj + dj);
                        if qi < 0 || qj < 0 || qi >= map.width as i32 || qj >= map.height as i32 {
                            continue;
                        }
                        let q = qj as usize * map.width + qi as usize;
                        let d = map.depth[q];
                        if d <= 0.0 {
                            continue;
                        }
                        let n = map.normal[q];
                        // The coarse pixel's plane at the fine pixel's ray.
                        let coarse_ray = ray(coarse, qi as f32 + 0.5, qj as f32 + 0.5);
                        let rho = d * dot(n, coarse_ray);
                        let denom = dot(n, fine_ray);
                        if denom >= -1e-6 {
                            continue;
                        }
                        let depth_here = rho / denom;
                        if !(depth_here > 0.0 && depth_here.is_finite()) {
                            continue;
                        }
                        // The fine image's grey level at the coarse pixel.
                        let fx = (((qi as f32 + 0.5) / sx) as usize).min(fw - 1);
                        let fy = (((qj as f32 + 0.5) / sy) as usize).min(fh - 1);
                        let dg = fine.gray[fy * fw + fx] - gray;
                        let (dx, dy) = (qi as f32 - cxf, qj as f32 - cyf);
                        let w = (-(dx * dx + dy * dy) / (2.0 * SIGMA_SPATIAL * SIGMA_SPATIAL)
                            - dg * dg / (2.0 * SIGMA_COLOR * SIGMA_COLOR))
                            .exp();
                        sum_w += w;
                        sum_d += w * depth_here;
                        sum_c += w * map.cost[q];
                        for (s, v) in sum_n.iter_mut().zip(n) {
                            *s += w * v;
                        }
                    }
                }
                if sum_w > 0.0 {
                    let norm = dot(sum_n, sum_n).sqrt();
                    if norm > 0.0 {
                        let n = sum_n.map(|v| v / norm);
                        if dot(n, fine_ray) < 0.0 {
                            depth[x] = sum_d / sum_w;
                            normal[x] = n;
                            cost[x] = sum_c / sum_w;
                        }
                    }
                }
            }
            (depth, normal, cost)
        })
        .collect();
    let mut out = DepthMap {
        width: fw,
        height: fh,
        depth: Vec::with_capacity(fw * fh),
        normal: Vec::with_capacity(fw * fh),
        cost: Vec::with_capacity(fw * fh),
    };
    for (d, n, c) in rows {
        out.depth.extend(d);
        out.normal.extend(n);
        out.cost.extend(c);
    }
    out
}

/// ACMM's detail restorer: the upsampled map, with the photometric pass's
/// estimate wherever the upsampled plane's photometric score at this scale
/// exceeds the photometric estimate's by more than `threshold`. Returns the
/// map and how many pixels it replaced.
fn restore_details(
    ws: &Workspace,
    view: usize,
    options: &PatchMatchOptions,
    upsampled: &DepthMap,
    photometric: &DepthMap,
    threshold: f32,
) -> (DepthMap, usize) {
    let init_scores = photometric_score(ws, view, options, upsampled);
    let photo_scores = photometric_score(ws, view, options, photometric);
    let mut out = upsampled.clone();
    let mut replaced = 0usize;
    for i in 0..out.depth.len() {
        let upsampled_missing = upsampled.depth[i] <= 0.0;
        if photometric.depth[i] > 0.0
            && (upsampled_missing || init_scores[i] - photo_scores[i] > threshold)
        {
            out.depth[i] = photometric.depth[i];
            out.normal[i] = photometric.normal[i];
            out.cost[i] = photometric.cost[i];
            replaced += 1;
        }
    }
    (out, replaced)
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch_match::{plane_scene, plane_scene_truth};

    #[test]
    fn downscale_keeps_projections() {
        let ws = plane_scene();
        let half = downscale(&ws);
        let point = nalgebra::Vector3::new(0.3, -0.2, 4.0);
        for (full, small) in ws.views.iter().zip(&half.views) {
            assert_eq!(
                (small.width, small.height),
                (full.width / 2, full.height / 2)
            );
            let (u, v, _) = full.project(&point);
            let (us, vs, _) = small.project(&point);
            assert!((us - u / 2.0).abs() < 1e-9 && (vs - v / 2.0).abs() < 1e-9);
        }
    }

    #[test]
    fn upsampling_a_plane_keeps_it() {
        let ws = plane_scene();
        let half = downscale(&ws);
        let coarse = plane_scene_truth(&half, 1);
        let fine = upsample(&coarse, &half.views[1], &ws.views[1]);
        let truth = plane_scene_truth(&ws, 1);
        for (d, t) in fine.depth.iter().zip(&truth.depth) {
            assert!((d - t).abs() < 1e-3 * t, "{d} vs {t}");
        }
    }
}
