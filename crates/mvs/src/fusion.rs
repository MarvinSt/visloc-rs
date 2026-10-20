//! Depth-map fusion into a point cloud (after ACMM, Gipuma and COLMAP):
//! each pixel's 3D point is checked against the depth maps of its view's
//! source views and kept when enough of them agree.

use nalgebra::Vector3;
use rayon::prelude::*;

use crate::patch_match::DepthMap;
use crate::workspace::Workspace;

/// When a source view's depth agrees with a pixel's, and how many must.
///
/// By default one agreeing source view within 10° of normal angle keeps a
/// point; ACMM's and ACMP's code ask for two. Over the 13 ETH3D training
/// scenes at 3200 px, F1 at 2 cm is 83.5% with these defaults and 83.1% at
/// 15° (three scales); with one scale, 81.4%, 81.3% at 15°, 80.6% at 30°
/// and 79.3% with two views at 30°. On pipes at 1600 px, one view instead
/// of two raises F1 from 66.9% to 76.0%.
#[derive(Debug, Clone)]
pub struct FusionOptions {
    /// Pixels between the pixel and the source depth's point projected back.
    pub max_reproj_error: f64,
    /// Depth difference relative to the pixel's depth.
    pub max_relative_depth_diff: f64,
    /// Angle between the normals, degrees.
    pub max_normal_angle_deg: f64,
    /// Source views that must agree.
    pub min_consistent_views: usize,
    /// The agreeing views must also agree well on average, as in ACMMP's
    /// released code: the mean over them of `exp(-(e + 200 d + 10 a))`,
    /// `e` the reprojection error in pixels, `d` the relative depth
    /// difference and `a` the normal angle in radians, must reach this
    /// (0.3 in ACMMP's code; 0 leaves it out).
    pub min_mean_consistency: f64,
    /// A point averages the agreeing pixels; without, it is the reference
    /// pixel's (ACMMP's released code).
    pub average: bool,
}

impl Default for FusionOptions {
    fn default() -> Self {
        Self {
            max_reproj_error: 2.0,
            max_relative_depth_diff: 0.01,
            max_normal_angle_deg: 10.0,
            min_consistent_views: 1,
            min_mean_consistency: 0.15,
            average: true,
        }
    }
}

/// One fused point: the mean position, normal and colour of the agreeing
/// pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusedPoint {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [u8; 3],
}

/// Fuses `maps` (indexed like `ws.views`), consuming them. Views are
/// visited in order and pixels in raster order; a pixel used by a point is
/// not used again. Each map is dropped once no later view reads it, so the
/// maps and the points are not all held at once.
///
/// Per view, the depth checks against the source maps run in parallel
/// first; the raster-order pass that builds the points then re-checks only
/// the sources that agreed, so the points are those of a sequential fusion.
pub fn fuse(ws: &Workspace, maps: Vec<DepthMap>, options: &FusionOptions) -> Vec<FusedPoint> {
    fuse_inner(ws, maps, options, None)
}

/// Receives a view's index and depths from [`fuse_with`].
pub type Released<'a> = dyn FnMut(usize, &[f32]) + 'a;

/// [`fuse`], handing each view's depths that enough source views agree
/// with (or that are in points; 0 elsewhere) to `released` once no later
/// point can use its pixels, just before its map is dropped (e.g. to
/// [`crate::mesh::TsdfMesher`]). The points are [`fuse`]'s.
pub fn fuse_with(
    ws: &Workspace,
    maps: Vec<DepthMap>,
    options: &FusionOptions,
    released: &mut Released<'_>,
) -> Vec<FusedPoint> {
    fuse_inner(ws, maps, options, Some(released))
}

fn fuse_inner(
    ws: &Workspace,
    mut maps: Vec<DepthMap>,
    options: &FusionOptions,
    mut released: Option<&mut Released<'_>>,
) -> Vec<FusedPoint> {
    let cos_max_angle = options.max_normal_angle_deg.to_radians().cos();
    let last_read_by = ws.last_read_by();
    for map in &mut maps {
        map.cost = Vec::new();
    }
    // Per view, a bit per pixel: already in a point.
    let mut used: Vec<Vec<u64>> = maps
        .iter()
        .map(|m| vec![0; m.depth.len().div_ceil(64)])
        .collect();
    // For `released`, per view, a bit per pixel: its depth agrees with
    // enough source views, whether or not their pixels are in points (the
    // pixels in points alone leave holes where a lower-resolution source
    // pixel went to a neighbour).
    let masks = released.is_some();
    let mut consistent: Vec<Vec<u64>> = if masks { used.clone() } else { Vec::new() };
    let mut points = Vec::new();
    let mut agreeing: Vec<(usize, usize)> = Vec::new();
    let mut candidates: Vec<u64> = Vec::new();
    for (i, view) in ws.views.iter().enumerate() {
        let map = &maps[i];
        let check = Check {
            ws,
            maps: &maps,
            view: i,
            max_reproj_error: options.max_reproj_error,
            max_relative_depth_diff: options.max_relative_depth_diff,
            cos_max_angle,
        };
        // Per pixel, bit `k` set when source `k` agrees (sources past the
        // 64th are always re-checked). Bits are only ever set in `used`, so
        // a source pixel in a point now is still in one later; for the
        // consistency masks, source pixels in points count too. On one
        // thread (and no masks) every source is checked once, in the
        // raster-order pass.
        let parallel = rayon::current_num_threads() > 1 || masks;
        candidates.clear();
        candidates.resize(map.depth.len(), if parallel { 0 } else { u64::MAX });
        candidates
            .par_chunks_mut(map.width.max(1))
            .enumerate()
            .filter(|_| parallel)
            .for_each(|(y, row)| {
                for (x, bits) in row.iter_mut().enumerate() {
                    let p = y * map.width + x;
                    if is_set(&used[i], p) || map.depth[p] <= 0.0 {
                        continue;
                    }
                    let pixel = check.pixel(x, y);
                    for (k, &j) in view.sources.iter().enumerate().take(64) {
                        let in_points = (!masks).then_some(used[j].as_slice());
                        if check.source(&pixel, j, in_points).is_some() {
                            *bits |= 1 << k;
                        }
                    }
                }
            });
        let unchecked = view.sources.len().saturating_sub(64);
        for y in 0..map.height {
            for x in 0..map.width {
                let p = y * map.width + x;
                let bits = candidates[p];
                if masks
                    && map.depth[p] > 0.0
                    && (is_set(&used[i], p)
                        || (bits.count_ones() as usize + unchecked)
                            >= options.min_consistent_views.max(1))
                {
                    set(&mut consistent[i], p);
                }
                if is_set(&used[i], p)
                    || map.depth[p] <= 0.0
                    || (bits.count_ones() as usize + unchecked) < options.min_consistent_views
                {
                    continue;
                }
                let pixel = check.pixel(x, y);
                agreeing.clear();
                let mut consistency = 0.0;
                let mut sum_point = pixel.point;
                let mut sum_normal = pixel.normal;
                let mut sum_color = view.rgb[p].map(f64::from);
                for (k, &j) in view.sources.iter().enumerate() {
                    if k < 64 && bits >> k & 1 == 0 {
                        continue;
                    }
                    let Some(a) = check.source(&pixel, j, Some(&used[j])) else {
                        continue;
                    };
                    agreeing.push((j, a.q));
                    consistency += a.consistency;
                    sum_point += a.point;
                    sum_normal += a.normal;
                    for (s, c) in sum_color.iter_mut().zip(ws.views[j].rgb[a.q]) {
                        *s += f64::from(c);
                    }
                }
                let consistent = agreeing.len() >= options.min_consistent_views
                    && consistency >= options.min_mean_consistency * agreeing.len() as f64;
                if consistent {
                    let (position, normal, color) = if options.average {
                        let count = (agreeing.len() + 1) as f64;
                        (
                            sum_point / count,
                            sum_normal.normalize(),
                            sum_color.map(|s| s / count),
                        )
                    } else {
                        (pixel.point, pixel.normal, view.rgb[p].map(f64::from))
                    };
                    points.push(FusedPoint {
                        position: [position.x as f32, position.y as f32, position.z as f32],
                        normal: [normal.x as f32, normal.y as f32, normal.z as f32],
                        color: color.map(|c| c.round().clamp(0.0, 255.0) as u8),
                    });
                    set(&mut used[i], p);
                    for &(j, q) in &agreeing {
                        set(&mut used[j], q);
                    }
                }
            }
        }
        for &s in &last_read_by[i] {
            if let Some(released) = released.as_mut() {
                let agreed: Vec<f32> = maps[s]
                    .depth
                    .iter()
                    .enumerate()
                    .map(|(p, &d)| {
                        if is_set(&used[s], p) || is_set(&consistent[s], p) {
                            d
                        } else {
                            0.0
                        }
                    })
                    .collect();
                released(s, &agreed);
                consistent[s] = Vec::new();
            }
            maps[s].depth = Vec::new();
            maps[s].normal = Vec::new();
            used[s] = Vec::new();
        }
    }
    points
}

/// The depth checks of one reference view against its source views' maps.
struct Check<'a> {
    ws: &'a Workspace,
    maps: &'a [DepthMap],
    view: usize,
    max_reproj_error: f64,
    max_relative_depth_diff: f64,
    cos_max_angle: f64,
}

/// A reference pixel: its centre, depth, 3D point and world normal.
struct Pixel {
    u: f64,
    v: f64,
    depth: f64,
    point: Vector3<f64>,
    normal: Vector3<f64>,
}

/// A source pixel that agrees with a reference pixel.
struct Agreement {
    q: usize,
    point: Vector3<f64>,
    normal: Vector3<f64>,
    consistency: f64,
}

impl Check<'_> {
    fn world_normal(&self, view: usize, n: [f32; 3]) -> Vector3<f64> {
        self.ws.views[view].rotation.transpose()
            * Vector3::new(f64::from(n[0]), f64::from(n[1]), f64::from(n[2]))
    }

    #[inline(always)]
    fn pixel(&self, x: usize, y: usize) -> Pixel {
        let map = &self.maps[self.view];
        let p = y * map.width + x;
        let depth = f64::from(map.depth[p]);
        let (u, v) = (x as f64 + 0.5, y as f64 + 0.5);
        Pixel {
            u,
            v,
            depth,
            point: self.ws.views[self.view].unproject(u, v, depth),
            normal: self.world_normal(self.view, map.normal[p]),
        }
    }

    /// The pixel of source view `j` that `pixel` projects into, when its
    /// depth agrees and it is in no point yet (`used`, view `j`'s bits, if
    /// given).
    #[inline(always)]
    fn source(&self, pixel: &Pixel, j: usize, used: Option<&[u64]>) -> Option<Agreement> {
        let view = &self.ws.views[self.view];
        let source = &self.ws.views[j];
        let source_map = &self.maps[j];
        let (su, sv, sz) = source.project(&pixel.point);
        if sz <= 0.0 || su < 0.0 || sv < 0.0 {
            return None;
        }
        let (sx, sy) = (su as usize, sv as usize);
        if sx >= source_map.width || sy >= source_map.height {
            return None;
        }
        let q = sy * source_map.width + sx;
        let source_depth = f64::from(source_map.depth[q]);
        if used.is_some_and(|used| is_set(used, q)) || source_depth <= 0.0 {
            return None;
        }
        let source_point = source.unproject(sx as f64 + 0.5, sy as f64 + 0.5, source_depth);
        let (ru, rv, rz) = view.project(&source_point);
        let reproj = ((ru - pixel.u).powi(2) + (rv - pixel.v).powi(2)).sqrt();
        let depth_diff = (rz - pixel.depth).abs() / pixel.depth;
        let source_normal = self.world_normal(j, source_map.normal[q]);
        let cos_angle = pixel.normal.dot(&source_normal);
        if reproj <= self.max_reproj_error
            && depth_diff <= self.max_relative_depth_diff
            && cos_angle >= self.cos_max_angle
        {
            let angle = cos_angle.clamp(-1.0, 1.0).acos();
            Some(Agreement {
                q,
                point: source_point,
                normal: source_normal,
                consistency: (-(reproj + 200.0 * depth_diff + 10.0 * angle)).exp(),
            })
        } else {
            None
        }
    }
}

fn is_set(bits: &[u64], i: usize) -> bool {
    bits[i / 64] >> (i % 64) & 1 == 1
}

fn set(bits: &mut [u64], i: usize) {
    bits[i / 64] |= 1 << (i % 64);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch_match::{plane_scene, plane_scene_truth};

    #[test]
    fn exact_maps_fuse_onto_the_plane() {
        let ws = plane_scene();
        let maps: Vec<DepthMap> = (0..ws.views.len())
            .map(|v| plane_scene_truth(&ws, v))
            .collect();
        let points = fuse(&ws, maps.clone(), &FusionOptions::default());
        // Most reference pixels see the plane in both other views.
        assert!(points.len() > 2000, "{} points", points.len());
        for p in &points {
            assert!((p.position[2] - 4.0).abs() < 1e-3, "{:?}", p.position);
            assert!(p.normal[2] < -0.999, "{:?}", p.normal);
        }
        // Requiring a third agreeing view (there are only two others) keeps
        // nothing.
        let strict = FusionOptions {
            min_consistent_views: 3,
            ..FusionOptions::default()
        };
        assert!(fuse(&ws, maps, &strict).is_empty());
    }

    /// Dropping maps once no later view reads them changes nothing: the
    /// same points as a fusion that keeps every map to the end, which a
    /// last view without depths, reading all the others, forces.
    #[test]
    fn released_maps_fuse_the_same_points() {
        let mut ws = plane_scene();
        // View 0's map is dropped after view 1.
        ws.views[2].sources = vec![1];
        let maps: Vec<DepthMap> = (0..ws.views.len())
            .map(|v| plane_scene_truth(&ws, v))
            .collect();
        let released = fuse(&ws, maps.clone(), &FusionOptions::default());
        assert!(!released.is_empty());

        let mut kept_ws = ws.clone();
        let mut last = kept_ws.views[0].clone();
        last.sources = vec![0, 1, 2];
        kept_ws.views.push(last);
        let mut kept_maps = maps;
        let mut no_depth = kept_maps[0].clone();
        no_depth.depth.iter_mut().for_each(|d| *d = 0.0);
        kept_maps.push(no_depth);
        let kept = fuse(&kept_ws, kept_maps, &FusionOptions::default());
        assert_eq!(released, kept);
    }
}
