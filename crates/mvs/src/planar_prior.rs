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
    for t in triangles.chunks_exact(3) {
        // The corners in the camera frame.
        let corner = |k: usize| -> [f64; 3] {
            let (x, y) = corners[t[k]];
            let d = f64::from(map.depth[y * w + x]);
            let r = ray(x as f64 + 0.5, y as f64 + 0.5);
            [r[0] * d, r[1] * d, d]
        };
        let (a, b, c) = (corner(0), corner(1), corner(2));
        let (u, v) = (sub(b, a), sub(c, a));
        let mut n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let norm = dot(n, n).sqrt();
        // Degenerate (or not finite): no plane.
        if norm.is_nan() || norm <= 1e-12 {
            continue;
        }
        n = n.map(|v| v / norm);
        // Facing the camera: the normal points back along the rays.
        if dot(n, a) > 0.0 {
            n = n.map(|v| -v);
        }
        let rho = dot(n, a);
        let plane = [n[0] as f32, n[1] as f32, n[2] as f32, rho as f32];

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
                let r = ray(q.0, q.1);
                let denom = dot(n, r);
                if denom >= -1e-6 {
                    continue;
                }
                let depth = rho / denom;
                if (depth_lo..=depth_hi).contains(&depth) {
                    planes[y * w + x] = plane;
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
        let mut ws = plane_scene();
        let (w, x0, x1, y0, y1) = (64usize, 18usize, 46usize, 12usize, 36usize);
        for y in y0..y1 {
            for x in x0..x1 {
                ws.views[0].gray[y * w + x] = 128.0;
            }
        }
        let options = PatchMatchOptions {
            iterations: 3,
            ..PatchMatchOptions::default()
        };
        let photometric = estimate(&ws, 0, &options, None);
        let prior_options = PlanarPriorOptions::default();
        let prior = planar_prior(&ws.views[0], &photometric, &prior_options);
        let with_prior =
            estimate_with_prior(&ws, 0, &options, &prior_options, &photometric, &prior);
        let truth = plane_scene_truth(&ws, 0);
        // Pixels whose whole window is textureless.
        let error = |map: &DepthMap| -> f32 {
            let (mut sum, mut count) = (0.0f32, 0usize);
            for y in y0 + 6..y1 - 6 {
                for x in x0 + 6..x1 - 6 {
                    let i = y * w + x;
                    sum += (map.depth[i] - truth.depth[i]).abs() / truth.depth[i];
                    count += 1;
                }
            }
            sum / count as f32
        };
        let (before, after) = (error(&photometric), error(&with_prior));
        assert!(after < 0.01, "with the prior: {after}");
        assert!(
            before > 5.0 * after,
            "photometric {before} vs prior {after}"
        );
    }
}
