//! ACMM's multi-scale depth estimation, with ACMMP's planar priors: ACMH on
//! an image pyramid, coarse to fine, with geometric consistency at every
//! scale.
//!
//! At the coarsest scale every view gets a photometric pass; each finer
//! scale starts from the coarser maps, upsampled by joint bilateral
//! upsampling guided by the finer image, and runs a photometric pass from
//! them in which the detail restorer decides which pixels may leave their
//! upsampled plane (upsampling blurs thin structures and boundaries, while
//! weak texture is better resolved at the coarser scale). With ACMMP, a
//! planar prior pass follows the photometric pass at every scale (ACMP,
//! [`mod@crate::planar_prior`]); then `geom_passes` geometric passes.

use rayon::prelude::*;

use crate::deformable::Deformable;
use crate::patch_match::{DepthMap, DetailRestorer, PatchMatchOptions};
use crate::planar_prior::{planar_prior, PlanarPrior, PlanarPriorOptions};
use crate::workspace::{View, Workspace};

/// Multi-scale estimation parameters (ACMMP's code by default: three
/// scales, the restorer against the plane at infinity with a margin of
/// 0.1, a planar prior pass and two geometric passes per scale).
#[derive(Debug, Clone)]
pub struct MultiScaleOptions {
    pub patch_match: PatchMatchOptions,
    /// Pyramid levels, each half the size of the next.
    pub scales: usize,
    /// Geometric passes per scale.
    pub geom_passes: usize,
    /// Red-black iterations of each geometric pass (`patch_match`'s for
    /// the other passes); ACMMP's code runs 2.
    pub geom_iterations: usize,
    /// The detail restorer of the photometric pass at each finer scale.
    pub restorer: DetailRestorer,
    /// The planar prior pass after each scale's photometric pass, if any.
    pub planar_prior: Option<PlanarPriorOptions>,
    /// APD-MVS's deformable patches at every scale but the coarsest, as its
    /// code schedules them: 16 sectors and a RANSAC threshold of 0.875% at
    /// the middle scales, 32 and 0.75% at the finest; a minimum within 6
    /// disparity steps in the photometric pass, then 4 and 2.
    pub deformable: Option<Deformable>,
}

impl Default for MultiScaleOptions {
    fn default() -> Self {
        Self {
            patch_match: PatchMatchOptions::default(),
            scales: 3,
            geom_passes: 2,
            geom_iterations: 3,
            restorer: DetailRestorer::Infinity(0.1),
            planar_prior: Some(PlanarPriorOptions::default()),
            deformable: None,
        }
    }
}

/// One view's depth map: [`crate::patch_match::estimate`] or the GPU's
/// equivalent.
pub type Estimator<'a> =
    dyn Fn(&Workspace, usize, &PatchMatchOptions, Option<&[DepthMap]>) -> DepthMap + 'a;

/// One view's planar prior pass from its photometric map and prior:
/// [`crate::patch_match::estimate_with_prior`] or the GPU's equivalent.
pub type PriorEstimator<'a> = dyn Fn(
        &Workspace,
        usize,
        &PatchMatchOptions,
        &PlanarPriorOptions,
        &DepthMap,
        &PlanarPrior,
    ) -> DepthMap
    + 'a;

/// One view's photometric pass from a start map:
/// [`crate::patch_match::estimate_from`] or the GPU's equivalent.
pub type StartEstimator<'a> =
    dyn Fn(&Workspace, usize, &PatchMatchOptions, &DepthMap, DetailRestorer) -> DepthMap + 'a;

/// One geometric pass: the depth map of every view of `ws` re-estimated by
/// `estimate` against `earlier`, the previous pass's maps (indexed like
/// `ws.views`), each view starting from its own earlier map.
///
/// The earlier maps are released as the pass goes, so it holds about one
/// set of maps rather than two: their costs at the start, a view's normals
/// once it is re-estimated, and a view's depths once the last view using
/// them (itself or as a source) is. So `estimate` may read, for its view,
/// the earlier depths of the view and of its sources and the view's own
/// earlier normals, nothing else.
pub fn geometric_pass(
    ws: &Workspace,
    mut earlier: Vec<DepthMap>,
    options: &PatchMatchOptions,
    estimate: &Estimator<'_>,
) -> Vec<DepthMap> {
    let n = ws.views.len();
    assert_eq!(earlier.len(), n, "one earlier depth map per view");
    let last_read_by = ws.last_read_by();
    for map in &mut earlier {
        map.cost = Vec::new();
    }
    let mut maps = Vec::with_capacity(n);
    for v in 0..n {
        maps.push(estimate(ws, v, options, Some(&earlier)));
        earlier[v].normal = Vec::new();
        for &s in &last_read_by[v] {
            earlier[s].depth = Vec::new();
        }
    }
    maps
}

/// ACMP's planar prior pass over every view of `ws`: each view's photometric
/// map in `maps` (with its costs) gives the view's planar prior
/// ([`planar_prior`]) and its start, and is dropped once the view is done.
/// `log` receives the share of pixels the priors cover.
pub fn planar_prior_pass(
    ws: &Workspace,
    maps: Vec<DepthMap>,
    options: &PatchMatchOptions,
    prior_options: &PlanarPriorOptions,
    estimate: &PriorEstimator<'_>,
    log: &mut dyn FnMut(&str),
) -> Vec<DepthMap> {
    assert_eq!(maps.len(), ws.views.len(), "one photometric map per view");
    let (mut covered, mut total) = (0usize, 0usize);
    let out = maps
        .into_iter()
        .enumerate()
        .map(|(v, start)| {
            let prior = planar_prior(&ws.views[v], &start, prior_options);
            covered += prior.covered();
            total += prior.planes.len();
            estimate(ws, v, options, prior_options, &start, &prior)
        })
        .collect();
    log(&format!(
        "planar priors cover {:.1}% of pixels",
        100.0 * covered as f64 / total.max(1) as f64
    ));
    out
}

/// Depth maps of every view of `ws` (at its resolution), estimated coarse
/// to fine: `photometric` passes at the coarsest scale, `from_start` ones at
/// the finer scales, `prior` passes (when `options.planar_prior` is set)
/// and `geometric` passes. The photometric passes must keep their costs
/// for the planar prior. `log` receives a line per stage.
pub fn multi_scale_depth_maps(
    ws: &Workspace,
    options: &MultiScaleOptions,
    photometric: &Estimator<'_>,
    from_start: &StartEstimator<'_>,
    prior: &PriorEstimator<'_>,
    geometric: &Estimator<'_>,
    log: &mut dyn FnMut(&str),
) -> Vec<DepthMap> {
    let scales = options.scales.max(1);
    // The coarser levels, coarsest first; `ws` is the finest. Each is
    // dropped once the next finer one has been restored from it.
    let mut coarser: Vec<Workspace> = Vec::with_capacity(scales - 1);
    for _ in 1..scales {
        let finer = coarser.last().unwrap_or(ws);
        coarser.push(downscale(finer));
    }
    coarser.reverse();
    let mut coarser = coarser.into_iter();
    // The current level, `None` for `ws`.
    let mut current = coarser.next();
    let pm = &options.patch_match;
    // A pass's options at scale `level`: `pass` 0 for the photometric
    // pass, 1 for the planar prior pass, 2 and on for the geometric ones.
    let pass_options = |level: usize, pass: usize| -> PatchMatchOptions {
        let mut o = pm.clone();
        if pass >= 2 {
            o.iterations = options.geom_iterations;
        }
        o.deformable = options.deformable.as_ref().filter(|_| level > 0).map(|d| {
            let finest = level + 1 == scales;
            Deformable {
                sectors: if finest { 32 } else { 16 },
                ransac_threshold: if finest { 0.0075 } else { 0.00875 },
                peak_radius: [6, 4, 2][pass.min(2)],
                ..d.clone()
            }
        });
        o
    };
    // The planar prior and geometric passes of one scale.
    let refine = |level: usize,
                  ws: &Workspace,
                  mut maps: Vec<DepthMap>,
                  log: &mut dyn FnMut(&str)|
     -> Vec<DepthMap> {
        if let Some(prior_options) = &options.planar_prior {
            let t = std::time::Instant::now();
            let prior_pm = pass_options(level, 1);
            maps = planar_prior_pass(ws, maps, &prior_pm, prior_options, prior, &mut |line| {
                log(&format!("scale {level}: {line}"))
            });
            log(&format!(
                "scale {level}: planar prior {:.1} s",
                t.elapsed().as_secs_f64()
            ));
        }
        for pass in 0..options.geom_passes {
            let t = std::time::Instant::now();
            let geom_pm = pass_options(level, 2 + pass);
            maps = geometric_pass(ws, maps, &geom_pm, geometric);
            log(&format!(
                "scale {level}: geometric {} {:.1} s",
                pass + 1,
                t.elapsed().as_secs_f64()
            ));
        }
        maps
    };

    let coarsest = current.as_ref().unwrap_or(ws);
    let started = std::time::Instant::now();
    let maps: Vec<DepthMap> = (0..coarsest.views.len())
        .map(|v| photometric(coarsest, v, pm, None))
        .collect();
    log(&format!(
        "scale 0 ({}x{}): photometric {:.1} s",
        coarsest.views[0].width,
        coarsest.views[0].height,
        started.elapsed().as_secs_f64()
    ));
    let mut maps = refine(0, coarsest, maps, log);
    for level in 1..scales {
        let next = coarser.next();
        let coarse = current.as_ref().unwrap_or(ws);
        let fine = next.as_ref().unwrap_or(ws);
        let t = std::time::Instant::now();
        let photometric_pm = pass_options(level, 0);
        // View by view, so each coarse map is dropped once upsampled.
        let restored: Vec<DepthMap> = maps
            .into_iter()
            .enumerate()
            .map(|(v, coarse_map)| {
                let upsampled = upsample(&coarse_map, &coarse.views[v], &fine.views[v]);
                drop(coarse_map);
                from_start(fine, v, &photometric_pm, &upsampled, options.restorer)
            })
            .collect();
        log(&format!(
            "scale {level} ({}x{}): upsampled and photometric {:.1} s",
            fine.views[0].width,
            fine.views[0].height,
            t.elapsed().as_secs_f64()
        ));
        current = next;
        let fine = current.as_ref().unwrap_or(ws);
        maps = refine(level, fine, restored, log);
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
            let fine_gray = view.gray_levels();
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
                        g += fine_gray[i];
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
/// fine pixel takes the plane of the 5×5 coarse pixels around it with the
/// largest bilateral weight (distance in coarse pixels, grey-level
/// difference in the fine image), evaluated on its own ray. Averaging the
/// planes' depths instead would bridge depth edges with surfaces that exist
/// in no view.
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
    let fine_gray = fine.gray_levels();
    // Costs carry over where `map` has them.
    let with_costs = !map.cost.is_empty();
    // Per row: depths, normals, costs.
    type Row = (Vec<f32>, Vec<[f32; 3]>, Vec<f32>);
    let rows: Vec<Row> = (0..fh)
        .into_par_iter()
        .map(|y| {
            let mut depth = vec![0.0f32; fw];
            let mut normal = vec![[0.0f32; 3]; fw];
            let mut cost = if with_costs {
                vec![2.0f32; fw]
            } else {
                Vec::new()
            };
            for x in 0..fw {
                let fine_ray = ray(fine, x as f32 + 0.5, y as f32 + 0.5);
                let gray = fine_gray[y * fw + x];
                // The fine pixel's centre in coarse pixel coordinates.
                let (cxf, cyf) = ((x as f32 + 0.5) * sx - 0.5, (y as f32 + 0.5) * sy - 0.5);
                let (ci, cj) = (cxf.round() as i32, cyf.round() as i32);
                // (weight, depth here, normal, coarse pixel) of the best
                // plane.
                let mut best: Option<(f32, f32, [f32; 3], usize)> = None;
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
                        let dg = fine_gray[fy * fw + fx] - gray;
                        let (dx, dy) = (qi as f32 - cxf, qj as f32 - cyf);
                        let w = (-(dx * dx + dy * dy) / (2.0 * SIGMA_SPATIAL * SIGMA_SPATIAL)
                            - dg * dg / (2.0 * SIGMA_COLOR * SIGMA_COLOR))
                            .exp();
                        if best.is_none_or(|(best_w, ..)| w > best_w) {
                            best = Some((w, depth_here, n, q));
                        }
                    }
                }
                if let Some((_, d, n, q)) = best {
                    depth[x] = d;
                    normal[x] = n;
                    if with_costs {
                        cost[x] = map.cost[q];
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
        cost: Vec::with_capacity(if with_costs { fw * fh } else { 0 }),
    };
    for (d, n, c) in rows {
        out.depth.extend(d);
        out.normal.extend(n);
        out.cost.extend(c);
    }
    out
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch_match::{estimate, plane_scene, plane_scene_truth};

    /// Releasing the earlier maps during a pass changes nothing: the same
    /// maps as a pass that keeps them all, also when a view's depths are
    /// released before the pass ends (view 0's, after view 1).
    #[test]
    fn geometric_pass_releases_without_changing_maps() {
        let mut ws = plane_scene();
        ws.views[2].sources = vec![1];
        let options = PatchMatchOptions {
            iterations: 2,
            ..PatchMatchOptions::default()
        };
        let first: Vec<DepthMap> = (0..ws.views.len())
            .map(|v| estimate(&ws, v, &options, None))
            .collect();
        let kept: Vec<DepthMap> = (0..ws.views.len())
            .map(|v| estimate(&ws, v, &options, Some(&first)))
            .collect();
        let released = geometric_pass(&ws, first, &options, &estimate);
        assert_eq!(released.len(), kept.len());
        for (a, b) in released.iter().zip(&kept) {
            assert_eq!(a.depth, b.depth);
            assert_eq!(a.normal, b.normal);
            assert_eq!(a.cost, b.cost);
        }
    }

    /// ACMMP's order (photometric, planar prior and geometric passes at
    /// each scale) recovers the plane, also where view 0 has no texture.
    #[test]
    fn multi_scale_with_priors_recovers_the_plane() {
        use crate::patch_match::{estimate_from, estimate_with_prior};
        use crate::planar_prior::{mean_relative_error, textureless_scene};
        let (ws, textureless) = textureless_scene();
        let truth = plane_scene_truth(&ws, 0);
        let options = MultiScaleOptions {
            scales: 2,
            ..MultiScaleOptions::default()
        };
        let maps = multi_scale_depth_maps(
            &ws,
            &options,
            &estimate,
            &estimate_from,
            &estimate_with_prior,
            &estimate,
            &mut |_| {},
        );
        let inside: Vec<usize> = (8..40)
            .flat_map(|y| (12..52).map(move |x| y * 64 + x))
            .collect();
        for pixels in [&inside, &textureless] {
            let error = mean_relative_error(&maps[0], &truth, pixels);
            assert!(error < 0.005, "{error}");
        }
    }

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
