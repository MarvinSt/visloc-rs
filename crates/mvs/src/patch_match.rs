//! PatchMatch depth and normal estimation: ACMH, with ACMM's geometric
//! consistency (Q. Xu and W. Tao, "Multi-Scale Geometric Consistency Guided
//! Multi-View Stereo", CVPR 2019).
//!
//! Each pixel of a reference view holds a plane hypothesis: a unit normal
//! `n` in the camera frame, facing the camera, and `rho = n · X` for the
//! points `X` of the plane, so the depth at pixel ray `r` is
//! `rho / (n · r)`. One iteration updates the pixels of a checkerboard's red
//! squares, then its black squares, each from the other colour:
//!
//! - **Adaptive checkerboard sampling.** Eight regions around the pixel,
//!   four V-shaped ones near it (7 pixels each) and four long strips along
//!   the axes (11 pixels each), each give their lowest-cost plane.
//! - **Multi-hypothesis joint view selection.** The eight planes' matching
//!   costs in every source view vote: a view is selected when more than
//!   `n1` of them are below `tau(t) = tau0 exp(-t² / alpha)` and fewer than
//!   `n2` above `tau1`, weighted by the mean of `exp(-m² / (2 beta²))` over
//!   its good costs. The previous iteration's most important view keeps a
//!   double weight (0.2 if it is no longer selected). The pixel takes the
//!   plane, its own included, with the lowest weighted mean cost.
//! - **Refinement.** A random and a perturbed depth and normal, combined
//!   with the current ones into six more planes.
//!
//! The matching cost is one minus a bilateral-weighted normalized
//! cross-correlation over every other row and column of an 11×11 window,
//! warped into the source view by the plane's homography.
//!
//! With depth maps of every view from an earlier pass, the geometric mode
//! adds `lambda` times each source view's forward-backward reprojection
//! error (truncated at `delta` pixels) to its cost: ACMM's geometric
//! consistency. It starts from the view's earlier depth map.

use std::borrow::Cow;

use rayon::prelude::*;

use crate::workspace::{View, Workspace};

/// The cost of a view where a plane cannot be matched.
pub(crate) const MAX_COST: f32 = 2.0;
/// Draws of the view weights under [`ViewSelection::Sampled`] (ACMM's).
const VIEW_SAMPLES: usize = 15;
/// Marks the random streams of those draws, apart from refinement's.
const SAMPLING_STREAM: u32 = 0x8000_0000;
/// Most window samples a cost uses.
const MAX_SAMPLES: usize = 64;
/// Least total bilateral weight of a matchable reference window.
const MIN_WEIGHT_SUM: f32 = 1e-6;
/// Most source views per reference view.
pub const MAX_SOURCES: usize = 32;

/// How a pixel weighs the source views from its candidates' costs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewSelection {
    /// The ACMH paper's rule: a selected view weighs the mean confidence
    /// of its good costs; the previous iteration's most important view
    /// keeps a double weight (0.2 if it is no longer selected).
    Confidence,
    /// ACMM's released code: that confidence, also `exp(-tau² / 0.32)` for
    /// a view with too few good costs but fewer than `n2` bad ones, times
    /// a prior from the four adjacent pixels' selected views (0.9 each
    /// that selected it, 0.1 each that did not), is a distribution the
    /// weights are drawn from: 15 samples, a view's weight its count.
    Sampled,
}

/// ACMH/ACMM parameters (the paper's values by default).
#[derive(Debug, Clone)]
pub struct PatchMatchOptions {
    /// Window half-size and the stride between its samples.
    pub window_radius: i32,
    pub window_step: i32,
    /// Bilateral weights of the window samples: spatial (pixels) and grey
    /// level (of 255) standard deviations.
    pub sigma_spatial: f32,
    pub sigma_color: f32,
    /// Red-black iterations per pass.
    pub iterations: usize,
    /// View selection: `tau0`, `alpha` (good-cost threshold), `tau1` (bad
    /// cost), `beta` (cost confidence), `n1`, `n2` (votes).
    pub tau0: f32,
    pub tau1: f32,
    pub alpha: f32,
    pub beta: f32,
    pub n1: usize,
    pub n2: usize,
    pub view_selection: ViewSelection,
    /// A random initial plane's cost: the mean of its best this-many views
    /// (they are its first selected views under
    /// [`ViewSelection::Sampled`]).
    pub init_top_k: usize,
    /// Geometric consistency: weight and truncation (pixels) of the
    /// reprojection error.
    pub geom_lambda: f32,
    pub geom_delta: f32,
    /// Refinement's perturbations: relative depth change and normal change
    /// per component, each drawn uniformly from `±value / 2`.
    pub depth_perturbation: f32,
    pub normal_perturbation: f32,
    /// A 5×5 median filter over the final depths (ACMH).
    pub median_filter: bool,
    pub seed: u32,
}

impl Default for PatchMatchOptions {
    fn default() -> Self {
        Self {
            window_radius: 5,
            window_step: 2,
            sigma_spatial: 5.0,
            sigma_color: 3.0,
            iterations: 3,
            tau0: 0.8,
            tau1: 1.2,
            alpha: 90.0,
            beta: 0.3,
            n1: 2,
            n2: 3,
            view_selection: ViewSelection::Confidence,
            init_top_k: 3,
            geom_lambda: 0.2,
            geom_delta: 3.0,
            depth_perturbation: 0.02,
            normal_perturbation: 0.02,
            median_filter: true,
            seed: 0,
        }
    }
}

/// A view's depths (camera `z`, 0 where there is none), unit normals in the
/// camera frame and final costs, row-major. The costs are for inspection
/// only (no pass or fusion reads them); `cost` is empty when they were not
/// kept.
#[derive(Debug, Clone)]
pub struct DepthMap {
    pub width: usize,
    pub height: usize,
    pub depth: Vec<f32>,
    pub normal: Vec<[f32; 3]>,
    pub cost: Vec<f32>,
}

impl DepthMap {
    /// Writes the map as raw little-endian data: width and height (`u32`),
    /// then the depths, normals and the costs if it has them (`f32`).
    pub fn write(&self, path: &std::path::Path) -> std::io::Result<()> {
        if !(self.cost.is_empty() || self.cost.len() == self.depth.len()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "depth map with a partial cost map",
            ));
        }
        let mut bytes = Vec::with_capacity(8 + self.depth.len() * 20);
        bytes.extend_from_slice(&(self.width as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.height as u32).to_le_bytes());
        for d in &self.depth {
            bytes.extend_from_slice(&d.to_le_bytes());
        }
        for n in &self.normal {
            for v in n {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        for c in &self.cost {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
        std::fs::write(path, bytes)
    }

    /// Reads a map [`DepthMap::write`] wrote.
    pub fn read(path: &std::path::Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let invalid =
            || std::io::Error::new(std::io::ErrorKind::InvalidData, "truncated depth map");
        let word = |i: usize| -> std::io::Result<[u8; 4]> {
            bytes
                .get(4 * i..4 * i + 4)
                .map(|b| [b[0], b[1], b[2], b[3]])
                .ok_or_else(invalid)
        };
        let width = u32::from_le_bytes(word(0)?) as usize;
        let height = u32::from_le_bytes(word(1)?) as usize;
        let n = width * height;
        let with_costs = if bytes.len() == 8 + n * 20 {
            true
        } else if bytes.len() == 8 + n * 16 {
            false
        } else {
            return Err(invalid());
        };
        let float = |i: usize| f32::from_le_bytes(word(2 + i).expect("checked length"));
        Ok(Self {
            width,
            height,
            depth: (0..n).map(float).collect(),
            normal: (0..n)
                .map(|i| [float(n + 3 * i), float(n + 3 * i + 1), float(n + 3 * i + 2)])
                .collect(),
            cost: if with_costs {
                (0..n).map(|i| float(4 * n + i)).collect()
            } else {
                Vec::new()
            },
        })
    }
}

/// Estimates the depth map of `ws.views[view]` from photometric consistency
/// alone (`prior` is `None`), or with geometric consistency against the
/// depth maps of every view from an earlier pass (`prior`, indexed like
/// `ws.views`), starting from this view's.
pub fn estimate(
    ws: &Workspace,
    view: usize,
    options: &PatchMatchOptions,
    prior: Option<&[DepthMap]>,
) -> DepthMap {
    let reference = &ws.views[view];
    let sources: Vec<Source<'_>> = reference
        .sources
        .iter()
        .take(MAX_SOURCES)
        .map(|&s| Source::new(reference, &ws.views[s], prior.map(|maps| &maps[s])))
        .collect();
    let ctx = Context::new(reference, sources, options, view as u32, prior.is_some());
    let mut state = match prior {
        None => ctx.random_state(),
        Some(maps) => ctx.state_from(&maps[view]),
    };
    for iteration in 0..options.iterations {
        for color in [0, 1] {
            ctx.pass(&mut state, iteration, color);
        }
    }
    ctx.depth_map(&state)
}

/// Per pixel of `map` (a depth map of `ws.views[view]`), the mean of the
/// best `init_top_k` source views' photometric costs of its plane, and the
/// maximum cost where it has no depth: how well the map fits the images at
/// this scale, independent of view selection.
pub fn photometric_score(
    ws: &Workspace,
    view: usize,
    options: &PatchMatchOptions,
    map: &DepthMap,
) -> Vec<f32> {
    let reference = &ws.views[view];
    let sources: Vec<Source<'_>> = reference
        .sources
        .iter()
        .take(MAX_SOURCES)
        .map(|&s| Source::new(reference, &ws.views[s], None))
        .collect();
    let ctx = Context::new(reference, sources, options, view as u32, false);
    let (w, h) = (reference.width, reference.height);
    assert_eq!((map.width, map.height), (w, h), "depth map size");
    let n = ctx.sources.len();
    let k = options.init_top_k.clamp(1, n.max(1));
    (0..h)
        .into_par_iter()
        .flat_map_iter(|y| {
            let ctx = &ctx;
            let (mut photo, mut geo) = ([0.0; MAX_SOURCES], [0.0; MAX_SOURCES]);
            let mut costs = Vec::with_capacity(n);
            (0..w)
                .map(move |x| {
                    let i = y * w + x;
                    if n == 0 || map.depth[i] <= 0.0 {
                        return MAX_COST;
                    }
                    let ray = ctx.ray(x, y);
                    let plane = Plane::from_depth(map.depth[i], map.normal[i], ray);
                    let patch = ctx.ref_patch(x, y);
                    ctx.view_costs(&patch, (x, y), ray, &plane, (&mut photo, &mut geo));
                    costs.clear();
                    costs.extend_from_slice(&photo[..n]);
                    costs.sort_by(f32::total_cmp);
                    costs[..k].iter().sum::<f32>() / k as f32
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// A plane: unit normal (camera frame, facing the camera) and `rho = n · X`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Plane {
    n: [f32; 3],
    rho: f32,
}

impl Plane {
    fn from_depth(depth: f32, n: [f32; 3], ray: [f32; 3]) -> Self {
        Self {
            n,
            rho: depth * dot(n, ray),
        }
    }

    /// Depth along `ray`, `None` when the ray misses the plane's front.
    fn depth(&self, ray: [f32; 3]) -> Option<f32> {
        let denom = dot(self.n, ray);
        if denom >= -1e-6 {
            return None;
        }
        let depth = self.rho / denom;
        (depth > 0.0 && depth.is_finite()).then_some(depth)
    }
}

/// Per-pixel state of a reference view.
struct State {
    plane: Vec<Plane>,
    cost: Vec<f32>,
    /// [`ViewSelection::Confidence`]: index into the sources of the
    /// previous iteration's most important view, `u32::MAX` for none.
    /// [`ViewSelection::Sampled`]: the selected views, bit `j` for source
    /// `j`.
    views: Vec<u32>,
}

/// One image row of a first state: planes, costs and view states.
type Row = (Vec<Plane>, Vec<f32>, Vec<u32>);

impl State {
    fn from_rows(rows: Vec<Row>, len: usize) -> Self {
        let mut state = State {
            plane: Vec::with_capacity(len),
            cost: Vec::with_capacity(len),
            views: Vec::with_capacity(len),
        };
        for (planes, costs, views) in rows {
            state.plane.extend(planes);
            state.cost.extend(costs);
            state.views.extend(views);
        }
        state
    }
}

/// A source view as the reference view sees it.
struct Source<'a> {
    view: &'a View,
    gray: Cow<'a, [f32]>,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    /// Reference camera frame to source camera frame.
    r: [[f32; 3]; 3],
    t: [f32; 3],
    prior: Option<&'a DepthMap>,
}

impl<'a> Source<'a> {
    fn new(reference: &View, view: &'a View, prior: Option<&'a DepthMap>) -> Self {
        let r = view.rotation * reference.rotation.transpose();
        let t = view.translation - r * reference.translation;
        Self {
            view,
            gray: view.gray_levels(),
            fx: view.fx as f32,
            fy: view.fy as f32,
            cx: view.cx as f32,
            cy: view.cy as f32,
            r: [
                [r[(0, 0)] as f32, r[(0, 1)] as f32, r[(0, 2)] as f32],
                [r[(1, 0)] as f32, r[(1, 1)] as f32, r[(1, 2)] as f32],
                [r[(2, 0)] as f32, r[(2, 1)] as f32, r[(2, 2)] as f32],
            ],
            t: [t.x as f32, t.y as f32, t.z as f32],
            prior,
        }
    }
}

/// The reference window around one pixel: sample offsets, grey levels
/// (minus the centre's, which NCC ignores and which keeps the f32 sums
/// accurate), bilateral weights and weighted statistics.
struct RefPatch {
    n: usize,
    offsets: [(f32, f32); MAX_SAMPLES],
    values: [f32; MAX_SAMPLES],
    weights: [f32; MAX_SAMPLES],
    center: f32,
    sum_w: f32,
    mean: f32,
    var: f32,
}

/// Everything one reference view's passes share.
struct Context<'a> {
    reference: &'a View,
    gray: Cow<'a, [f32]>,
    sources: Vec<Source<'a>>,
    options: &'a PatchMatchOptions,
    view_id: u32,
    geometric: bool,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    depth_min: f32,
    depth_max: f32,
    /// The window's sample offsets.
    window: Vec<(i32, i32)>,
}

/// Per-view matching costs of the eight candidates.
type CostRows = [[f32; MAX_SOURCES]; 8];

impl<'a> Context<'a> {
    fn new(
        reference: &'a View,
        sources: Vec<Source<'a>>,
        options: &'a PatchMatchOptions,
        view_id: u32,
        geometric: bool,
    ) -> Self {
        let r = options.window_radius;
        let step = options.window_step.max(1);
        let mut window = Vec::new();
        let mut dy = -r;
        while dy <= r {
            let mut dx = -r;
            while dx <= r {
                window.push((dx, dy));
                dx += step;
            }
            dy += step;
        }
        assert!(
            window.len() <= MAX_SAMPLES,
            "window of {} samples exceeds {MAX_SAMPLES}",
            window.len()
        );
        Self {
            reference,
            gray: reference.gray_levels(),
            sources,
            options,
            view_id,
            geometric,
            fx: reference.fx as f32,
            fy: reference.fy as f32,
            cx: reference.cx as f32,
            cy: reference.cy as f32,
            depth_min: reference.depth_range.0,
            depth_max: reference.depth_range.1,
            window,
        }
    }

    fn ray(&self, x: usize, y: usize) -> [f32; 3] {
        [
            (x as f32 + 0.5 - self.cx) / self.fx,
            (y as f32 + 0.5 - self.cy) / self.fy,
            1.0,
        ]
    }

    fn ref_patch(&self, x: usize, y: usize) -> RefPatch {
        let (w, h) = (self.reference.width as i32, self.reference.height as i32);
        let gray = &self.gray;
        let center = gray[y * self.reference.width + x];
        let mut patch = RefPatch {
            n: self.window.len(),
            offsets: [(0.0, 0.0); MAX_SAMPLES],
            values: [0.0; MAX_SAMPLES],
            weights: [0.0; MAX_SAMPLES],
            center,
            sum_w: 0.0,
            mean: 0.0,
            var: 0.0,
        };
        let inv_color = 1.0 / (2.0 * self.options.sigma_color * self.options.sigma_color);
        let inv_spatial = 1.0 / (2.0 * self.options.sigma_spatial * self.options.sigma_spatial);
        let (mut sum, mut sum_sq) = (0.0f32, 0.0f32);
        for (k, &(dx, dy)) in self.window.iter().enumerate() {
            let sx = (x as i32 + dx).clamp(0, w - 1) as usize;
            let sy = (y as i32 + dy).clamp(0, h - 1) as usize;
            let value = gray[sy * self.reference.width + sx] - center;
            let weight =
                (-value * value * inv_color - (dx * dx + dy * dy) as f32 * inv_spatial).exp();
            patch.offsets[k] = (dx as f32, dy as f32);
            patch.values[k] = value;
            patch.weights[k] = weight;
            patch.sum_w += weight;
            sum += weight * value;
            sum_sq += weight * value * value;
        }
        if patch.sum_w < MIN_WEIGHT_SUM {
            // Every sample differs too much from the centre (the window
            // skips it): nothing to match, where 0/0 would give NaN costs.
            return patch;
        }
        patch.mean = sum / patch.sum_w;
        patch.var = sum_sq / patch.sum_w - patch.mean * patch.mean;
        patch
    }

    /// One minus the bilateral NCC of `plane` at the pixel of `patch` (ray
    /// `ray`) in `source`.
    fn photometric_cost(
        &self,
        patch: &RefPatch,
        source: &Source<'_>,
        ray: [f32; 3],
        plane: &Plane,
    ) -> f32 {
        if patch.var < 1e-5 || plane.rho.abs() < 1e-12 {
            return MAX_COST;
        }
        // A plane point X maps to R X + t = (R + t nᵀ / rho) X: the
        // homography in camera rays, applied to the ray of each sample
        // (the centre's plus offset / f).
        let inv_rho = 1.0 / plane.rho;
        let mut m = source.r;
        for (row, t) in m.iter_mut().zip(source.t) {
            for (value, n) in row.iter_mut().zip(plane.n) {
                *value += t * n * inv_rho;
            }
        }
        let a = mat_vec(&m, ray);
        let bx = [m[0][0] / self.fx, m[1][0] / self.fx, m[2][0] / self.fx];
        let by = [m[0][1] / self.fy, m[1][1] / self.fy, m[2][1] / self.fy];
        let view = source.view;
        let (w, h) = (view.width as f32, view.height as f32);
        if a[2] <= 0.0 {
            return MAX_COST;
        }
        let cx_s = source.fx * a[0] / a[2] + source.cx;
        let cy_s = source.fy * a[1] / a[2] + source.cy;
        if !(0.0..w).contains(&cx_s) || !(0.0..h).contains(&cy_s) {
            return MAX_COST;
        }
        let (mut s_w, mut s_ww, mut s_rw) = (0.0f32, 0.0f32, 0.0f32);
        for k in 0..patch.n {
            let (dx, dy) = patch.offsets[k];
            let p = [
                a[0] + dx * bx[0] + dy * by[0],
                a[1] + dx * bx[1] + dy * by[1],
                a[2] + dx * bx[2] + dy * by[2],
            ];
            if p[2] <= 0.0 {
                return MAX_COST;
            }
            let u = source.fx * p[0] / p[2] + source.cx;
            let v = source.fy * p[1] / p[2] + source.cy;
            let value = bilinear(&source.gray, view.width, view.height, u, v) - patch.center;
            let weight = patch.weights[k];
            s_w += weight * value;
            s_ww += weight * value * value;
            s_rw += weight * patch.values[k] * value;
        }
        let mean = s_w / patch.sum_w;
        let var = s_ww / patch.sum_w - mean * mean;
        if var < 1e-5 {
            return MAX_COST;
        }
        let cov = s_rw / patch.sum_w - patch.mean * mean;
        let ncc = cov / (patch.var * var).sqrt();
        (1.0 - ncc).clamp(0.0, MAX_COST)
    }

    /// Forward-backward reprojection error of `plane` at pixel `(x, y)`
    /// through `source`'s earlier depth map, truncated at `geom_delta`.
    fn reprojection_error(
        &self,
        source: &Source<'_>,
        x: usize,
        y: usize,
        ray: [f32; 3],
        plane: &Plane,
    ) -> f32 {
        let delta = self.options.geom_delta;
        let (Some(prior), Some(depth)) = (source.prior, plane.depth(ray)) else {
            return delta;
        };
        let point = [ray[0] * depth, ray[1] * depth, depth];
        let p = add(mat_vec(&source.r, point), source.t);
        if p[2] <= 0.0 {
            return delta;
        }
        let u = source.fx * p[0] / p[2] + source.cx;
        let v = source.fy * p[1] / p[2] + source.cy;
        if u < 0.0 || v < 0.0 || u >= prior.width as f32 || v >= prior.height as f32 {
            return delta;
        }
        let source_depth = prior.depth[v as usize * prior.width + u as usize];
        if source_depth <= 0.0 {
            return delta;
        }
        let back = [
            (u - source.cx) / source.fx * source_depth,
            (v - source.cy) / source.fy * source_depth,
            source_depth,
        ];
        // Source frame back to the reference frame: Rᵀ (X - t).
        let d = sub(back, source.t);
        let q = [
            source.r[0][0] * d[0] + source.r[1][0] * d[1] + source.r[2][0] * d[2],
            source.r[0][1] * d[0] + source.r[1][1] * d[1] + source.r[2][1] * d[2],
            source.r[0][2] * d[0] + source.r[1][2] * d[1] + source.r[2][2] * d[2],
        ];
        if q[2] <= 0.0 {
            return delta;
        }
        let du = self.fx * q[0] / q[2] + self.cx - (x as f32 + 0.5);
        let dv = self.fy * q[1] / q[2] + self.cy - (y as f32 + 0.5);
        (du * du + dv * dv).sqrt().min(delta)
    }

    /// Photometric costs of `plane` in every source view, and the
    /// geometric terms (`lambda` times the reprojection error, zero
    /// outside the geometric mode).
    fn view_costs(
        &self,
        patch: &RefPatch,
        (x, y): (usize, usize),
        ray: [f32; 3],
        plane: &Plane,
        (photo, geo): (&mut [f32; MAX_SOURCES], &mut [f32; MAX_SOURCES]),
    ) {
        for (j, source) in self.sources.iter().enumerate() {
            photo[j] = self.photometric_cost(patch, source, ray, plane);
            geo[j] = if self.geometric {
                self.options.geom_lambda * self.reprojection_error(source, x, y, ray, plane)
            } else {
                0.0
            };
        }
    }

    /// A first plane's cost, the mean of its best `init_top_k` views'
    /// costs, and its per-pixel view state: those views as selection bits
    /// under [`ViewSelection::Sampled`], none (`u32::MAX`) otherwise.
    fn initial_cost(&self, photo: &[f32; MAX_SOURCES], geo: &[f32; MAX_SOURCES]) -> (f32, u32) {
        let n = self.sources.len();
        let sampled = self.options.view_selection == ViewSelection::Sampled;
        if n == 0 {
            return (MAX_COST, if sampled { 0 } else { u32::MAX });
        }
        let k = self.options.init_top_k.clamp(1, n);
        let mut total = [0.0f32; MAX_SOURCES];
        for j in 0..n {
            total[j] = photo[j] + geo[j];
        }
        let mut sorted = total;
        sorted[..n].sort_by(f32::total_cmp);
        let cost = sorted[..k].iter().sum::<f32>() / k as f32;
        if !sampled {
            return (cost, u32::MAX);
        }
        let threshold = sorted[k - 1];
        let views = (0..n)
            .filter(|&j| total[j] <= threshold)
            .fold(0u32, |bits, j| bits | 1 << j);
        (cost, views)
    }

    /// The weighted mean over the source views of `plane`'s costs
    /// (photometric plus geometric), matching only the views with weight.
    fn weighted_cost(
        &self,
        patch: &RefPatch,
        (x, y): (usize, usize),
        ray: [f32; 3],
        plane: &Plane,
        weights: &[f32; MAX_SOURCES],
        weight_sum: f32,
    ) -> f32 {
        let geometric = |j: usize| {
            if self.geometric {
                let error = self.reprojection_error(&self.sources[j], x, y, ray, plane);
                self.options.geom_lambda * error
            } else {
                0.0
            }
        };
        (0..self.sources.len())
            .filter(|&j| weights[j] > 0.0)
            .map(|j| {
                let photo = self.photometric_cost(patch, &self.sources[j], ray, plane);
                weights[j] * (photo + geometric(j))
            })
            .sum::<f32>()
            / weight_sum
    }

    /// A random plane at the ray: depth uniform in the depth range, normal
    /// uniform over the directions facing the camera.
    fn random_plane(&self, rng: &mut Rng, ray: [f32; 3]) -> Plane {
        let depth = self.depth_min + rng.uniform() * (self.depth_max - self.depth_min);
        Plane::from_depth(depth, random_normal(rng, ray), ray)
    }

    fn random_state(&self) -> State {
        let (w, h) = (self.reference.width, self.reference.height);
        let rows: Vec<Row> = (0..h)
            .into_par_iter()
            .map(|y| {
                let mut planes = Vec::with_capacity(w);
                let mut costs = Vec::with_capacity(w);
                let mut views = Vec::with_capacity(w);
                let (mut photo, mut geo) = ([0.0; MAX_SOURCES], [0.0; MAX_SOURCES]);
                for x in 0..w {
                    let mut rng = Rng::new(self.options.seed, self.view_id, x, y, u32::MAX);
                    let ray = self.ray(x, y);
                    let plane = self.random_plane(&mut rng, ray);
                    let patch = self.ref_patch(x, y);
                    self.view_costs(&patch, (x, y), ray, &plane, (&mut photo, &mut geo));
                    let (cost, selected) = self.initial_cost(&photo, &geo);
                    planes.push(plane);
                    costs.push(cost);
                    views.push(selected);
                }
                (planes, costs, views)
            })
            .collect();
        State::from_rows(rows, w * h)
    }

    /// The state of an earlier depth map, with costs recomputed under this
    /// pass's terms (the mean of the best `init_top_k` views).
    fn state_from(&self, map: &DepthMap) -> State {
        let (w, h) = (self.reference.width, self.reference.height);
        assert_eq!((map.width, map.height), (w, h), "prior depth map size");
        let rows: Vec<Row> = (0..h)
            .into_par_iter()
            .map(|y| {
                let mut planes = Vec::with_capacity(w);
                let mut costs = Vec::with_capacity(w);
                let mut views = Vec::with_capacity(w);
                let (mut photo, mut geo) = ([0.0; MAX_SOURCES], [0.0; MAX_SOURCES]);
                for x in 0..w {
                    let ray = self.ray(x, y);
                    let i = y * w + x;
                    let plane = if map.depth[i] > 0.0 {
                        Plane::from_depth(map.depth[i], map.normal[i], ray)
                    } else {
                        let mut rng = Rng::new(self.options.seed, self.view_id, x, y, u32::MAX);
                        self.random_plane(&mut rng, ray)
                    };
                    let patch = self.ref_patch(x, y);
                    self.view_costs(&patch, (x, y), ray, &plane, (&mut photo, &mut geo));
                    let (cost, selected) = self.initial_cost(&photo, &geo);
                    planes.push(plane);
                    costs.push(cost);
                    views.push(selected);
                }
                (planes, costs, views)
            })
            .collect();
        State::from_rows(rows, w * h)
    }

    /// Updates the pixels of one checkerboard colour (`(x + y) % 2 ==
    /// color`) from the other's current state.
    fn pass(&self, state: &mut State, iteration: usize, color: usize) {
        let (w, h) = (self.reference.width, self.reference.height);
        let updates: Vec<Vec<(usize, Plane, f32, u32)>> = (0..h)
            .into_par_iter()
            .map(|y| {
                let mut row = Vec::with_capacity(w / 2 + 1);
                let mut x = (y + color) % 2;
                while x < w {
                    let (plane, cost, views) = self.update_pixel(state, x, y, iteration, color);
                    row.push((x, plane, cost, views));
                    x += 2;
                }
                row
            })
            .collect();
        for (y, row) in updates.into_iter().enumerate() {
            for (x, plane, cost, views) in row {
                let i = y * w + x;
                state.plane[i] = plane;
                state.cost[i] = cost;
                state.views[i] = views;
            }
        }
    }

    /// The new plane, cost and view state (see [`State::views`]) of pixel
    /// `(x, y)`.
    fn update_pixel(
        &self,
        state: &State,
        x: usize,
        y: usize,
        iteration: usize,
        color: usize,
    ) -> (Plane, f32, u32) {
        let (w, h) = (self.reference.width, self.reference.height);
        let n = self.sources.len();
        let i = y * w + x;
        if n == 0 {
            return (state.plane[i], MAX_COST, state.views[i]);
        }
        let ray = self.ray(x, y);
        let patch = self.ref_patch(x, y);

        // Adaptive checkerboard sampling: the lowest-cost plane of each of
        // the eight regions (other colour, inside the image).
        let mut candidates: [Option<Plane>; 8] = [None; 8];
        for (region, offsets) in sampling_regions().iter().enumerate() {
            let mut best: Option<(f32, usize)> = None;
            for &(dx, dy) in offsets.iter() {
                let (sx, sy) = (x as i32 + dx, y as i32 + dy);
                if sx < 0 || sy < 0 || sx >= w as i32 || sy >= h as i32 {
                    continue;
                }
                let j = sy as usize * w + sx as usize;
                if best.is_none_or(|(cost, _)| state.cost[j] < cost) {
                    best = Some((state.cost[j], j));
                }
            }
            candidates[region] = best.map(|(_, j)| state.plane[j]);
        }

        // Per-view costs of the candidates.
        let mut photo: CostRows = [[MAX_COST; MAX_SOURCES]; 8];
        let mut geo: CostRows = [[0.0; MAX_SOURCES]; 8];
        for (row, candidate) in candidates.iter().enumerate() {
            if let Some(plane) = candidate {
                self.view_costs(&patch, (x, y), ray, plane, (&mut photo[row], &mut geo[row]));
            }
        }
        let current = state.plane[i];

        // Multi-hypothesis joint view selection over the candidates.
        let opt = self.options;
        let t = iteration as f32;
        let tau = opt.tau0 * (-t * t / opt.alpha).exp();
        // A view's votes: the candidates' good and bad costs in it, and the
        // confidence of the good ones.
        let votes = |j: usize| -> (usize, usize, f32) {
            let (mut good, mut bad, mut confidence) = (0usize, 0usize, 0.0f32);
            for (row, candidate) in candidates.iter().enumerate() {
                if candidate.is_none() {
                    continue;
                }
                let m = photo[row][j];
                if m < tau {
                    good += 1;
                    confidence += (-m * m / (2.0 * opt.beta * opt.beta)).exp();
                } else if m > opt.tau1 {
                    bad += 1;
                }
            }
            (good, bad, confidence)
        };
        let mut weights = [0.0f32; MAX_SOURCES];
        let mut sampled_views = 0u32;
        match opt.view_selection {
            ViewSelection::Confidence => {
                let previous = state.views[i];
                for (j, weight) in weights.iter_mut().enumerate().take(n) {
                    let (good, bad, confidence) = votes(j);
                    let selected = good > opt.n1 && bad < opt.n2;
                    let is_previous = previous as usize == j;
                    *weight = if selected {
                        let w0 = confidence / good as f32;
                        if is_previous {
                            2.0 * w0
                        } else {
                            w0
                        }
                    } else if is_previous {
                        0.2
                    } else {
                        0.0
                    };
                }
            }
            ViewSelection::Sampled => {
                let mut prior = [0.0f32; MAX_SOURCES];
                for (dx, dy) in [(0i32, -1i32), (0, 1), (-1, 0), (1, 0)] {
                    let (sx, sy) = (x as i32 + dx, y as i32 + dy);
                    if sx < 0 || sy < 0 || sx >= w as i32 || sy >= h as i32 {
                        continue;
                    }
                    let bits = state.views[sy as usize * w + sx as usize];
                    for (j, p) in prior.iter_mut().enumerate().take(n) {
                        *p += if bits >> j & 1 == 1 { 0.9 } else { 0.1 };
                    }
                }
                // The unnormalized distribution's running sums.
                let mut cdf = [0.0f32; MAX_SOURCES];
                let mut total = 0.0f32;
                for j in 0..n {
                    let (good, bad, confidence) = votes(j);
                    let p = if good > opt.n1 && bad < opt.n2 {
                        confidence / good as f32
                    } else if bad < opt.n2 {
                        (-tau * tau / 0.32).exp()
                    } else {
                        0.0
                    };
                    total += p * prior[j];
                    cdf[j] = total;
                }
                if total > 0.0 {
                    let pass = SAMPLING_STREAM | (iteration * 2 + color) as u32;
                    let mut rng = Rng::new(opt.seed, self.view_id, x, y, pass);
                    for _ in 0..VIEW_SAMPLES {
                        let u = rng.uniform() * total;
                        if let Some(j) = (0..n).find(|&j| cdf[j] > u) {
                            weights[j] += 1.0;
                            sampled_views |= 1 << j;
                        }
                    }
                }
            }
        }
        let weight_sum: f32 = weights[..n].iter().sum();
        if weight_sum <= 0.0 {
            // No view selected yet (random planes early on): all count.
            weights[..n].iter_mut().for_each(|w| *w = 1.0);
        }
        let weight_sum: f32 = weights[..n].iter().sum();
        let aggregate = |photo: &[f32; MAX_SOURCES], geo: &[f32; MAX_SOURCES]| -> f32 {
            (0..n)
                .map(|j| weights[j] * (photo[j] + geo[j]))
                .sum::<f32>()
                / weight_sum
        };

        // Views without weight add nothing, so the current plane and the
        // refinement's are only matched in the others.
        let mut best_plane = current;
        let mut best_cost = self.weighted_cost(&patch, (x, y), ray, &current, &weights, weight_sum);
        let mut adopted = false;
        for (row, candidate) in candidates.iter().enumerate() {
            if let Some(plane) = candidate {
                let cost = aggregate(&photo[row], &geo[row]);
                if cost < best_cost && plane.depth(ray).is_some() {
                    best_cost = cost;
                    best_plane = *plane;
                    adopted = true;
                }
            }
        }
        let views = match opt.view_selection {
            ViewSelection::Confidence => (0..n)
                .max_by(|&a, &b| weights[a].total_cmp(&weights[b]).then(b.cmp(&a)))
                .map_or(u32::MAX, |j| j as u32),
            // As ACMM: a pixel's selection changes with its plane.
            ViewSelection::Sampled if adopted => sampled_views,
            ViewSelection::Sampled => state.views[i],
        };

        // Refinement: a random and a perturbed depth and normal combined
        // with the current ones.
        let mut rng = Rng::new(opt.seed, self.view_id, x, y, (iteration * 2 + color) as u32);
        let depth_now = best_plane
            .depth(ray)
            .unwrap_or(0.5 * (self.depth_min + self.depth_max));
        let normal_now = best_plane.n;
        let depth_random = self.depth_min + rng.uniform() * (self.depth_max - self.depth_min);
        let normal_random = random_normal(&mut rng, ray);
        let depth_perturbed = depth_now * (1.0 + (rng.uniform() - 0.5) * opt.depth_perturbation);
        let normal_perturbed = perturb_normal(&mut rng, normal_now, ray, opt.normal_perturbation);
        let hypotheses = [
            (depth_random, normal_now),
            (depth_now, normal_random),
            (depth_random, normal_random),
            (depth_perturbed, normal_now),
            (depth_now, normal_perturbed),
            (depth_perturbed, normal_perturbed),
        ];
        for (depth, normal) in hypotheses {
            if !(self.depth_min..=self.depth_max).contains(&depth) {
                continue;
            }
            let plane = Plane::from_depth(depth, normal, ray);
            let cost = self.weighted_cost(&patch, (x, y), ray, &plane, &weights, weight_sum);
            if cost < best_cost {
                best_cost = cost;
                best_plane = plane;
            }
        }
        (best_plane, best_cost, views)
    }

    fn depth_map(&self, state: &State) -> DepthMap {
        let planes: Vec<[f32; 4]> = state
            .plane
            .iter()
            .map(|p| [p.n[0], p.n[1], p.n[2], p.rho])
            .collect();
        finish_depth_map(self.reference, &planes, state.cost.clone(), self.options)
    }
}

/// The depth map of `reference` from per-pixel planes `(n, rho)` and costs
/// (shared with the GPU path): depths on the pixel rays, then the median
/// filter if enabled.
pub(crate) fn finish_depth_map(
    reference: &View,
    planes: &[[f32; 4]],
    cost: Vec<f32>,
    options: &PatchMatchOptions,
) -> DepthMap {
    let (w, h) = (reference.width, reference.height);
    let (fx, fy, cx, cy) = (
        reference.fx as f32,
        reference.fy as f32,
        reference.cx as f32,
        reference.cy as f32,
    );
    let mut depth = vec![0.0f32; w * h];
    let mut normal = vec![[0.0f32; 3]; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let [nx, ny, nz, rho] = planes[i];
            let plane = Plane {
                n: [nx, ny, nz],
                rho,
            };
            let ray = [(x as f32 + 0.5 - cx) / fx, (y as f32 + 0.5 - cy) / fy, 1.0];
            if let Some(d) = plane.depth(ray) {
                depth[i] = d;
                normal[i] = plane.n;
            }
        }
    }
    if options.median_filter {
        depth = median_filter_5x5(&depth, w, h);
    }
    DepthMap {
        width: w,
        height: h,
        depth,
        normal,
        cost,
    }
}

/// The eight sampling regions as pixel offsets, all of the other
/// checkerboard colour: V shapes opening away from the pixel (7 pixels),
/// then long strips along the axes (11 pixels, every other pixel from 3 to
/// 23 away), each up, down, left and right.
fn sampling_regions() -> &'static [Vec<(i32, i32)>; 8] {
    static REGIONS: std::sync::OnceLock<[Vec<(i32, i32)>; 8]> = std::sync::OnceLock::new();
    REGIONS.get_or_init(|| {
        let v_up: Vec<(i32, i32)> = vec![
            (0, -1),
            (-1, -2),
            (1, -2),
            (-2, -3),
            (2, -3),
            (-3, -4),
            (3, -4),
        ];
        let strip_up: Vec<(i32, i32)> = (0..11).map(|k| (0, -(2 * k + 3))).collect();
        let down = |o: &[(i32, i32)]| o.iter().map(|&(dx, dy)| (dx, -dy)).collect::<Vec<_>>();
        let left = |o: &[(i32, i32)]| o.iter().map(|&(dx, dy)| (dy, dx)).collect::<Vec<_>>();
        let right = |o: &[(i32, i32)]| o.iter().map(|&(dx, dy)| (-dy, dx)).collect::<Vec<_>>();
        [
            v_up.clone(),
            down(&v_up),
            left(&v_up),
            right(&v_up),
            strip_up.clone(),
            down(&strip_up),
            left(&strip_up),
            right(&strip_up),
        ]
    })
}

/// A unit normal uniform over the directions facing the camera along `ray`.
fn random_normal(rng: &mut Rng, ray: [f32; 3]) -> [f32; 3] {
    let z = 2.0 * rng.uniform() - 1.0;
    let phi = 2.0 * std::f32::consts::PI * rng.uniform();
    let s = (1.0 - z * z).max(0.0).sqrt();
    let n = [s * phi.cos(), s * phi.sin(), z];
    if dot(n, ray) > 0.0 {
        [-n[0], -n[1], -n[2]]
    } else {
        n
    }
}

/// `n` with each component moved by up to `±amount / 2`, renormalized and
/// kept facing the camera.
fn perturb_normal(rng: &mut Rng, n: [f32; 3], ray: [f32; 3], amount: f32) -> [f32; 3] {
    let p = [
        n[0] + (rng.uniform() - 0.5) * amount,
        n[1] + (rng.uniform() - 0.5) * amount,
        n[2] + (rng.uniform() - 0.5) * amount,
    ];
    let norm = dot(p, p).sqrt();
    let p = [p[0] / norm, p[1] / norm, p[2] / norm];
    if dot(p, ray) >= 0.0 {
        n
    } else {
        p
    }
}

/// Bilinear grey level at pixel coordinates `(u, v)` (pixel centres at
/// `+0.5`), clamped to the image.
fn bilinear(gray: &[f32], w: usize, h: usize, u: f32, v: f32) -> f32 {
    let x = (u - 0.5).clamp(0.0, (w - 1) as f32);
    let y = (v - 0.5).clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let top = gray[y0 * w + x0] * (1.0 - fx) + gray[y0 * w + x1] * fx;
    let bottom = gray[y1 * w + x0] * (1.0 - fx) + gray[y1 * w + x1] * fx;
    top * (1.0 - fy) + bottom * fy
}

/// 5×5 median of the positive depths around each pixel (pixels without a
/// depth keep none).
fn median_filter_5x5(depth: &[f32], w: usize, h: usize) -> Vec<f32> {
    (0..h)
        .into_par_iter()
        .flat_map_iter(|y| {
            let mut window = Vec::with_capacity(25);
            (0..w)
                .map(|x| {
                    if depth[y * w + x] <= 0.0 {
                        return 0.0;
                    }
                    window.clear();
                    for sy in y.saturating_sub(2)..(y + 3).min(h) {
                        for sx in x.saturating_sub(2)..(x + 3).min(w) {
                            let d = depth[sy * w + sx];
                            if d > 0.0 {
                                window.push(d);
                            }
                        }
                    }
                    window.sort_by(f32::total_cmp);
                    window[window.len() / 2]
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// A small counter-based generator: the stream of a pixel in one pass is a
/// pure function of its inputs, so results do not depend on the thread
/// schedule.
struct Rng(u32);

impl Rng {
    fn new(seed: u32, view: u32, x: usize, y: usize, pass: u32) -> Self {
        let mut h = hash(seed ^ 0x9e37_79b9);
        for v in [view, x as u32, y as u32, pass] {
            h = hash(h ^ v);
        }
        Self(h)
    }

    /// Uniform in `[0, 1)`.
    fn uniform(&mut self) -> f32 {
        self.0 = hash(self.0.wrapping_add(0x9e37_79b9));
        (self.0 >> 8) as f32 * (1.0 / 16_777_216.0)
    }
}

/// A 32-bit integer hash (lowbias32).
fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn mat_vec(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

/// A textured fronto-parallel plane at depth 4 seen by a reference and two
/// source cameras (tests).
#[cfg(test)]
pub(crate) fn plane_scene() -> Workspace {
    use nalgebra::{Matrix3, Rotation3, Vector3};
    let (w, h) = (64usize, 48usize);
    let (f, cx, cy) = (60.0, w as f64 / 2.0, h as f64 / 2.0);
    let texture = |px: f64, py: f64| -> f32 {
        let v = (px * 1.7).sin() * (py * 2.3).cos() + (px * 0.37 + py * 0.71).sin();
        (128.0 + 60.0 * v) as f32
    };
    let make = |name: &str, rotation: Matrix3<f64>, translation: Vector3<f64>| {
        let mut gray = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                // The plane z = 4 in the world; intersect the pixel ray.
                let ray_cam =
                    Vector3::new((x as f64 + 0.5 - cx) / f, (y as f64 + 0.5 - cy) / f, 1.0);
                let center = -(rotation.transpose() * translation);
                let ray = rotation.transpose() * ray_cam;
                let s = (4.0 - center.z) / ray.z;
                let p = center + ray * s;
                gray[y * w + x] = texture(p.x * 3.0, p.y * 3.0);
            }
        }
        View {
            name: name.to_owned(),
            width: w,
            height: h,
            fx: f,
            fy: f,
            cx,
            cy,
            rotation,
            translation,
            rgb: gray.iter().map(|&g| [g as u8; 3]).collect(),
            gray,
            depth_range: (2.0, 8.0),
            sources: Vec::new(),
        }
    };
    let identity = Matrix3::identity();
    let mut reference = make("ref", identity, Vector3::zeros());
    let small = Rotation3::from_euler_angles(0.0, 0.03, 0.0).into_inner();
    let source1 = make("src1", small, Vector3::new(-0.4, 0.0, 0.0));
    let source2 = make("src2", identity, Vector3::new(0.0, 0.3, 0.0));
    reference.sources = vec![1, 2];
    let mut s1 = source1;
    s1.sources = vec![0, 2];
    let mut s2 = source2;
    s2.sources = vec![0, 1];
    Workspace {
        views: vec![reference, s1, s2],
    }
}

/// The exact depth map of `plane_scene`'s view `view` (tests).
#[cfg(test)]
pub(crate) fn plane_scene_truth(ws: &Workspace, view: usize) -> DepthMap {
    use nalgebra::Vector3;
    let v = &ws.views[view];
    let center = v.center();
    let normal = v.rotation * Vector3::new(0.0, 0.0, -1.0);
    let mut depth = Vec::with_capacity(v.width * v.height);
    for y in 0..v.height {
        for x in 0..v.width {
            let ray_cam = Vector3::new(
                (x as f64 + 0.5 - v.cx) / v.fx,
                (y as f64 + 0.5 - v.cy) / v.fy,
                1.0,
            );
            let ray = v.rotation.transpose() * ray_cam;
            let s = (4.0 - center.z) / ray.z;
            let point = center + ray * s;
            depth.push((v.rotation * point + v.translation).z as f32);
        }
    }
    let n = [normal.x as f32, normal.y as f32, normal.z as f32];
    DepthMap {
        width: v.width,
        height: v.height,
        normal: vec![n; depth.len()],
        cost: vec![0.0; depth.len()],
        depth,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_true_plane_matches_best() {
        let ws = plane_scene();
        let options = PatchMatchOptions::default();
        let ctx = Context::new(
            &ws.views[0],
            ws.views[0]
                .sources
                .iter()
                .map(|&s| Source::new(&ws.views[0], &ws.views[s], None))
                .collect(),
            &options,
            0,
            false,
        );
        let (x, y) = (32, 24);
        let ray = ctx.ray(x, y);
        let patch = ctx.ref_patch(x, y);
        let truth = Plane::from_depth(4.0, [0.0, 0.0, -1.0], ray);
        let wrong = Plane::from_depth(5.0, [0.0, 0.0, -1.0], ray);
        for source in &ctx.sources {
            let good = ctx.photometric_cost(&patch, source, ray, &truth);
            let bad = ctx.photometric_cost(&patch, source, ray, &wrong);
            assert!(good < 0.05, "true plane cost {good}");
            assert!(bad > good + 0.2, "wrong plane cost {bad} vs {good}");
        }
    }

    #[test]
    fn estimation_recovers_a_plane() {
        let ws = plane_scene();
        for view_selection in [ViewSelection::Confidence, ViewSelection::Sampled] {
            let options = PatchMatchOptions {
                iterations: 4,
                view_selection,
                ..PatchMatchOptions::default()
            };
            let map = estimate(&ws, 0, &options, None);
            // Away from the borders (window and parallax), most pixels
            // land within 1% of the true depth.
            let mut inside = 0;
            let mut good = 0;
            for y in 8..40 {
                for x in 12..52 {
                    inside += 1;
                    let d = map.depth[y * 64 + x];
                    if (d - 4.0).abs() < 0.04 {
                        good += 1;
                    }
                }
            }
            assert!(
                good * 10 >= inside * 9,
                "{view_selection:?}: {good} of {inside} pixels within 1%"
            );
            // Geometric consistency against the other views' maps keeps it.
            let maps: Vec<DepthMap> = (0..3).map(|v| estimate(&ws, v, &options, None)).collect();
            let refined = estimate(&ws, 0, &options, Some(&maps));
            let mut good_geo = 0;
            for y in 8..40 {
                for x in 12..52 {
                    if (refined.depth[y * 64 + x] - 4.0).abs() < 0.04 {
                        good_geo += 1;
                    }
                }
            }
            assert!(
                good_geo * 10 >= inside * 9,
                "{view_selection:?}: {good_geo} of {inside} after geometric pass"
            );
        }
    }

    #[test]
    fn a_window_without_weight_costs_the_maximum() {
        // A lone bright pixel: every window sample (the window skips the
        // centre) is 255 grey levels off, so all bilateral weights vanish.
        let mut ws = plane_scene();
        let view = &mut ws.views[0];
        view.gray.iter_mut().for_each(|g| *g = 0.0);
        let (x, y) = (32, 24);
        view.gray[y * view.width + x] = 255.0;
        let truth = plane_scene_truth(&ws, 0);
        let scores = photometric_score(&ws, 0, &PatchMatchOptions::default(), &truth);
        assert_eq!(scores[y * 64 + x], MAX_COST);
    }

    #[test]
    fn sampling_regions_take_the_other_colour() {
        for region in sampling_regions() {
            for &(dx, dy) in region {
                assert_eq!((dx + dy).rem_euclid(2), 1, "offset ({dx}, {dy})");
            }
        }
        assert_eq!(sampling_regions()[0].len(), 7);
        assert_eq!(sampling_regions()[4].len(), 11);
    }

    /// Maps round-trip through their files, with or without costs.
    #[test]
    fn depth_maps_round_trip_with_or_without_costs() {
        let ws = plane_scene();
        let mut map = plane_scene_truth(&ws, 1);
        map.cost = (0..map.depth.len()).map(|i| i as f32 / 100.0).collect();
        let path = std::env::temp_dir().join(format!("visloc_mvs_map_{}.bin", std::process::id()));
        for keep_costs in [true, false] {
            if !keep_costs {
                map.cost = Vec::new();
            }
            map.write(&path).expect("writing the map");
            let back = DepthMap::read(&path).expect("reading the map");
            assert_eq!((back.width, back.height), (map.width, map.height));
            assert_eq!(back.depth, map.depth);
            assert_eq!(back.normal, map.normal);
            assert_eq!(back.cost, map.cost);
        }
        std::fs::remove_file(&path).ok();
        map.cost = vec![0.0; 3];
        assert!(map.write(&path).is_err(), "a partial cost map is refused");
    }
}
