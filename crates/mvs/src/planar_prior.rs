//! ACMP's planar prior (Q. Xu and W. Tao, "Planar Prior Assisted
//! PatchMatch Multi-View Stereo", AAAI 2020).
//!
//! The pixels a photometric pass matched with a credible cost are
//! triangulated in the image (Delaunay); the three back-projected corners
//! of each triangle span a plane, the prior of the pixels inside it. A
//! planar prior pass ([`crate::patch_match::estimate_with_prior`]) then
//! ranks a pixel's hypotheses by
//! `exp(-c² / alpha) · (gamma + exp(-Δd² / (2 σd²)) · exp(-Δθ² / (2 σθ²)))`:
//! `c` the photometric cost, `Δd` and `Δθ` the depth and normal angle to
//! the prior. Where the photometric cost cannot tell hypotheses apart (weak
//! texture), the one closest to the prior wins; elsewhere the cost decides.

use delaunator::{triangulate, Point};

use crate::patch_match::DepthMap;
use crate::workspace::View;

/// ACMP's parameters (the paper's values by default).
#[derive(Debug, Clone)]
pub struct PlanarPriorOptions {
    /// A pixel is credible when its photometric cost is below this (ε).
    pub credible_cost: f32,
    /// Credible pixels are the lowest-cost pixel of each block of this many
    /// pixels a side (ACMP's code).
    pub block: usize,
    /// The photometric cost's bandwidth α.
    pub alpha: f32,
    /// The prior's floor γ.
    pub gamma: f32,
    /// The depth bandwidth σd, as a fraction of the view's depth range.
    pub depth_bandwidth: f32,
    /// The normal bandwidth σθ, degrees.
    pub angle_bandwidth_deg: f32,
    /// HPM-MVS's KNN supplement for the pixels outside the triangulation.
    pub knn_supplement: bool,
}

impl Default for PlanarPriorOptions {
    fn default() -> Self {
        Self {
            credible_cost: 0.1,
            block: 5,
            alpha: 0.18,
            gamma: 0.5,
            depth_bandwidth: 1.0 / 64.0,
            angle_bandwidth_deg: 5.0,
            knn_supplement: false,
        }
    }
}

/// Per pixel of one view, row-major, its prior plane: the unit normal in the
/// camera frame, facing the camera, and `rho = n · X` for the plane's points
/// `X`; all zeros where no triangle gives one.
#[derive(Debug, Clone)]
pub struct PlanarPrior {
    pub width: usize,
    pub height: usize,
    pub planes: Vec<[f32; 4]>,
}

impl PlanarPrior {
    /// The prior plane of pixel `i`, if it has one.
    pub fn plane(&self, i: usize) -> Option<[f32; 4]> {
        let p = self.planes[i];
        (p[0] != 0.0 || p[1] != 0.0 || p[2] != 0.0).then_some(p)
    }

    /// How many pixels have a prior.
    pub fn covered(&self) -> usize {
        (0..self.planes.len())
            .filter(|&i| self.plane(i).is_some())
            .count()
    }
}

/// The planar prior of `view` from its photometric depth map `map`, which
/// must keep its costs.
pub fn planar_prior(view: &View, map: &DepthMap, options: &PlanarPriorOptions) -> PlanarPrior {
    let (w, h) = (map.width, map.height);
    assert_eq!((w, h), (view.width, view.height), "depth map size");
    assert_eq!(map.cost.len(), w * h, "the planar prior needs the costs");
    let ray =
        |x: f64, y: f64| -> [f64; 3] { [(x - view.cx) / view.fx, (y - view.cy) / view.fy, 1.0] };

    // The credible pixels: per block, the lowest-cost pixel with a depth,
    // if that cost is credible.
    let block = options.block.max(1);
    let mut corners: Vec<(usize, usize)> = Vec::new();
    for by in (0..h).step_by(block) {
        for bx in (0..w).step_by(block) {
            let mut best: Option<(f32, usize, usize)> = None;
            for y in by..(by + block).min(h) {
                for x in bx..(bx + block).min(w) {
                    let i = y * w + x;
                    let cost = map.cost[i];
                    if map.depth[i] > 0.0 && best.is_none_or(|(c, ..)| cost < c) {
                        best = Some((cost, x, y));
                    }
                }
            }
            if let Some((cost, x, y)) = best {
                if cost < options.credible_cost {
                    corners.push((x, y));
                }
            }
        }
    }

    let mut planes = vec![[0.0f32; 4]; w * h];
    // Pixels inside a triangle, whether or not its plane is kept there.
    let mut inside_any = vec![false; w * h];
    let points: Vec<Point> = corners
        .iter()
        .map(|&(x, y)| Point {
            x: x as f64 + 0.5,
            y: y as f64 + 0.5,
        })
        .collect();
    let triangles = triangulate(&points).triangles;
    let (depth_lo, depth_hi) = (
        0.6 * f64::from(view.depth_range.0),
        1.2 * f64::from(view.depth_range.1),
    );
    // The plane through three credible pixels, `(n, rho)` with `n` facing
    // the camera, if they span one.
    let plane_through = |k: [usize; 3]| -> Option<[f64; 4]> {
        // The corners in the camera frame.
        let corner = |c: usize| -> [f64; 3] {
            let (x, y) = corners[c];
            let d = f64::from(map.depth[y * w + x]);
            let r = ray(x as f64 + 0.5, y as f64 + 0.5);
            [r[0] * d, r[1] * d, d]
        };
        let (a, b, c) = (corner(k[0]), corner(k[1]), corner(k[2]));
        let (u, v) = (sub(b, a), sub(c, a));
        let mut n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let norm = dot(n, n).sqrt();
        // Degenerate (or not finite): no plane.
        if norm.is_nan() || norm <= 1e-12 {
            return None;
        }
        n = n.map(|v| v / norm);
        // Facing the camera: the normal points back along the rays.
        if dot(n, a) > 0.0 {
            n = n.map(|v| -v);
        }
        Some([n[0], n[1], n[2], dot(n, a)])
    };
    // The plane's depth on pixel `(x, y)`'s ray.
    let depth_at = |p: [f64; 4], x: usize, y: usize| -> Option<f64> {
        let denom = dot([p[0], p[1], p[2]], ray(x as f64 + 0.5, y as f64 + 0.5));
        (denom < -1e-6).then(|| p[3] / denom)
    };
    let as_f32 = |p: [f64; 4]| p.map(|v| v as f32);
    for t in triangles.chunks_exact(3) {
        let Some(through) = plane_through([t[0], t[1], t[2]]) else {
            continue;
        };
        let plane = as_f32(through);

        // The pixels whose centres fall inside the triangle.
        let p: [(f64, f64); 3] = [0, 1, 2].map(|k| (points[t[k]].x, points[t[k]].y));
        let area = edge(p[0], p[1], p[2]);
        if area == 0.0 {
            continue;
        }
        let x0 = p
            .iter()
            .map(|q| q.0)
            .fold(f64::INFINITY, f64::min)
            .floor()
            .max(0.0) as usize;
        let x1 = (p
            .iter()
            .map(|q| q.0)
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil() as usize)
            .min(w);
        let y0 = p
            .iter()
            .map(|q| q.1)
            .fold(f64::INFINITY, f64::min)
            .floor()
            .max(0.0) as usize;
        let y1 = (p
            .iter()
            .map(|q| q.1)
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil() as usize)
            .min(h);
        for y in y0..y1 {
            for x in x0..x1 {
                let q = (x as f64 + 0.5, y as f64 + 0.5);
                // Edge functions with the triangle's own orientation.
                let inside = [
                    edge(p[1], p[2], q),
                    edge(p[2], p[0], q),
                    edge(p[0], p[1], q),
                ]
                .iter()
                .all(|&e| e * area >= 0.0);
                if !inside {
                    continue;
                }
                inside_any[y * w + x] = true;
                if depth_at(through, x, y).is_some_and(|d| (depth_lo..=depth_hi).contains(&d)) {
                    planes[y * w + x] = plane;
                }
            }
        }
    }

    // HPM-MVS's KNN supplement: a pixel outside every triangle takes the
    // plane of the first three of its six nearest credible pixels (the
    // nearest first) that are not collinear, if its depth there is within
    // the view's depth range.
    if options.knn_supplement && corners.len() >= 3 {
        let (bw, bh) = (w.div_ceil(block), h.div_ceil(block));
        // The credible pixel of each block, if any.
        let mut grid = vec![usize::MAX; bw * bh];
        for (k, &(x, y)) in corners.iter().enumerate() {
            grid[(y / block) * bw + x / block] = k;
        }
        let range = (f64::from(view.depth_range.0), f64::from(view.depth_range.1));
        const TRIPLES: [[usize; 3]; 20] = [
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 4],
            [0, 2, 3],
            [0, 2, 4],
            [0, 3, 4],
            [1, 2, 3],
            [1, 2, 4],
            [1, 3, 4],
            [2, 3, 4],
            [0, 1, 5],
            [0, 2, 5],
            [0, 3, 5],
            [0, 4, 5],
            [1, 2, 5],
            [1, 3, 5],
            [1, 4, 5],
            [2, 3, 5],
            [2, 4, 5],
            [3, 4, 5],
        ];
        let mut near: Vec<(f64, usize)> = Vec::with_capacity(64);
        for y in 0..h {
            for x in 0..w {
                if inside_any[y * w + x] {
                    continue;
                }
                // Rings of blocks around the pixel's, until the six nearest
                // are known: a credible pixel in ring r is at least
                // (r - 1) blocks away.
                near.clear();
                let (cx, cy) = ((x / block) as i64, (y / block) as i64);
                let q = (x as f64 + 0.5, y as f64 + 0.5);
                for r in 0i64.. {
                    if near.len() >= 6 {
                        near.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                        let reach = ((r - 1).max(0) * block as i64) as f64;
                        if near[5].0 <= reach * reach {
                            break;
                        }
                    }
                    if r > bw.max(bh) as i64 {
                        break;
                    }
                    for gy in (cy - r)..=(cy + r) {
                        for gx in (cx - r)..=(cx + r) {
                            let on_ring = (gy - cy).abs() == r || (gx - cx).abs() == r;
                            if !on_ring || gx < 0 || gy < 0 || gx >= bw as i64 || gy >= bh as i64 {
                                continue;
                            }
                            let k = grid[gy as usize * bw + gx as usize];
                            if k == usize::MAX {
                                continue;
                            }
                            let pk = &points[k];
                            let d2 = (pk.x - q.0).powi(2) + (pk.y - q.1).powi(2);
                            near.push((d2, k));
                        }
                    }
                }
                near.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                if near.len() < 3 {
                    continue;
                }
                let found = near.len().min(6);
                let collinear = |t: [usize; 3]| {
                    let p = t.map(|i| (points[near[i].1].x, points[near[i].1].y));
                    edge(p[0], p[1], p[2]) == 0.0
                };
                let Some(t) = TRIPLES
                    .iter()
                    .find(|t| t.iter().all(|&i| i < found) && !collinear(**t))
                else {
                    continue;
                };
                let Some(p) = plane_through(t.map(|i| near[i].1)) else {
                    continue;
                };
                if depth_at(p, x, y).is_some_and(|d| (range.0..=range.1).contains(&d)) {
                    planes[y * w + x] = as_f32(p);
                }
            }
        }
    }
    PlanarPrior {
        width: w,
        height: h,
        planes,
    }
}

/// Twice the signed area of triangle `(a, b, c)`.
fn edge(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `plane_scene` with a textureless rectangle in the reference image (view
/// 0), and the pixels whose whole matching window lies inside it (tests).
#[cfg(test)]
pub(crate) fn textureless_scene() -> (crate::workspace::Workspace, Vec<usize>) {
    let mut ws = crate::patch_match::plane_scene();
    let (w, x0, x1, y0, y1) = (ws.views[0].width, 18, 46, 12, 36);
    for y in y0..y1 {
        for x in x0..x1 {
            ws.views[0].gray[y * w + x] = 128.0;
        }
    }
    let inner = (y0 + 6..y1 - 6)
        .flat_map(|y| (x0 + 6..x1 - 6).map(move |x| y * w + x))
        .collect();
    (ws, inner)
}

/// The mean relative depth error of `map` over `pixels` (tests).
#[cfg(test)]
pub(crate) fn mean_relative_error(map: &DepthMap, truth: &DepthMap, pixels: &[usize]) -> f32 {
    pixels
        .iter()
        .map(|&i| (map.depth[i] - truth.depth[i]).abs() / truth.depth[i])
        .sum::<f32>()
        / pixels.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch_match::{plane_scene, plane_scene_truth};

    /// From the exact depths of a plane, every pixel inside the hull of the
    /// credible pixels gets that plane.
    #[test]
    fn a_plane_gives_its_own_prior() {
        let ws = plane_scene();
        let view = &ws.views[1];
        let mut truth = plane_scene_truth(&ws, 1);
        truth.cost = vec![0.0; truth.depth.len()];
        let prior = planar_prior(view, &truth, &PlanarPriorOptions::default());
        let (w, h) = (prior.width, prior.height);
        // The credible pixels are the blocks' first pixels, so the hull
        // spans (0, 0) to the last block's start.
        for y in 1..h - 5 {
            for x in 1..w - 5 {
                let i = y * w + x;
                let p = prior
                    .plane(i)
                    .unwrap_or_else(|| panic!("no prior at ({x}, {y})"));
                let ray = [
                    (x as f32 + 0.5 - view.cx as f32) / view.fx as f32,
                    (y as f32 + 0.5 - view.cy as f32) / view.fy as f32,
                    1.0,
                ];
                let depth = p[3] / (p[0] * ray[0] + p[1] * ray[1] + p[2] * ray[2]);
                assert!(
                    (depth - truth.depth[i]).abs() < 1e-3 * truth.depth[i],
                    "({x}, {y}): {depth} vs {}",
                    truth.depth[i]
                );
            }
        }
        // With HPM-MVS's KNN supplement the pixels outside the hull (the
        // last columns and rows) get the plane too.
        let knn = PlanarPriorOptions {
            knn_supplement: true,
            ..PlanarPriorOptions::default()
        };
        let supplemented = planar_prior(view, &truth, &knn);
        assert_eq!(supplemented.covered(), w * h);
        for i in 0..w * h {
            let (x, y) = (i % w, i / w);
            let p = supplemented.plane(i).unwrap();
            let ray = [
                (x as f32 + 0.5 - view.cx as f32) / view.fx as f32,
                (y as f32 + 0.5 - view.cy as f32) / view.fy as f32,
                1.0,
            ];
            let depth = p[3] / (p[0] * ray[0] + p[1] * ray[1] + p[2] * ray[2]);
            assert!((depth - truth.depth[i]).abs() < 1e-3 * truth.depth[i]);
            if prior.plane(i).is_some() {
                assert_eq!(supplemented.planes[i], prior.planes[i]);
            }
        }
        // No credible pixel, no prior.
        truth.cost = vec![1.0; truth.depth.len()];
        let none = planar_prior(view, &truth, &PlanarPriorOptions::default());
        assert_eq!(none.covered(), 0);
    }

    /// Where the reference image has no texture, photometric consistency
    /// cannot place the plane, but the prior from the textured pixels
    /// around it does.
    #[test]
    fn the_prior_fills_a_textureless_region() {
        use crate::patch_match::{estimate, estimate_with_prior, PatchMatchOptions};
        let (ws, inner) = textureless_scene();
        let options = PatchMatchOptions::default();
        let photometric = estimate(&ws, 0, &options, None);
        let prior_options = PlanarPriorOptions::default();
        let prior = planar_prior(&ws.views[0], &photometric, &prior_options);
        let with_prior =
            estimate_with_prior(&ws, 0, &options, &prior_options, &photometric, &prior);
        let truth = plane_scene_truth(&ws, 0);
        let before = mean_relative_error(&photometric, &truth, &inner);
        let after = mean_relative_error(&with_prior, &truth, &inner);
        assert!(after < 0.01, "with the prior: {after}");
        assert!(
            before > 5.0 * after,
            "photometric {before} vs prior {after}"
        );
    }
}
