//! Depth-map fusion into a point cloud (after ACMM, Gipuma and COLMAP):
//! each pixel's 3D point is checked against the depth maps of its view's
//! source views and kept when enough of them agree.

use nalgebra::Vector3;

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
    /// ACMMP's released code also asks the agreeing views to agree well on
    /// average: the mean over them of `exp(-(e + 200 d + 10 a))`, `e` the
    /// reprojection error in pixels, `d` the relative depth difference and
    /// `a` the normal angle in radians, must exceed this (0.3 there); 0
    /// leaves it out.
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
            min_mean_consistency: 0.0,
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
pub fn fuse(ws: &Workspace, mut maps: Vec<DepthMap>, options: &FusionOptions) -> Vec<FusedPoint> {
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
    let mut points = Vec::new();
    let world_normal = |view: usize, n: [f32; 3]| -> Vector3<f64> {
        ws.views[view].rotation.transpose()
            * Vector3::new(f64::from(n[0]), f64::from(n[1]), f64::from(n[2]))
    };
    let mut agreeing: Vec<(usize, usize)> = Vec::new();
    for (i, view) in ws.views.iter().enumerate() {
        let map = &maps[i];
        for y in 0..map.height {
            for x in 0..map.width {
                let p = y * map.width + x;
                let depth = f64::from(map.depth[p]);
                if is_set(&used[i], p) || depth <= 0.0 {
                    continue;
                }
                let (u, v) = (x as f64 + 0.5, y as f64 + 0.5);
                let point = view.unproject(u, v, depth);
                let normal = world_normal(i, map.normal[p]);
                agreeing.clear();
                let mut consistency = 0.0;
                let mut sum_point = point;
                let mut sum_normal = normal;
                let mut sum_color = view.rgb[p].map(f64::from);
                for &j in &view.sources {
                    let source = &ws.views[j];
                    let source_map = &maps[j];
                    let (su, sv, sz) = source.project(&point);
                    if sz <= 0.0 || su < 0.0 || sv < 0.0 {
                        continue;
                    }
                    let (sx, sy) = (su as usize, sv as usize);
                    if sx >= source_map.width || sy >= source_map.height {
                        continue;
                    }
                    let q = sy * source_map.width + sx;
                    let source_depth = f64::from(source_map.depth[q]);
                    if is_set(&used[j], q) || source_depth <= 0.0 {
                        continue;
                    }
                    let source_point =
                        source.unproject(sx as f64 + 0.5, sy as f64 + 0.5, source_depth);
                    let (ru, rv, rz) = view.project(&source_point);
                    let reproj = ((ru - u).powi(2) + (rv - v).powi(2)).sqrt();
                    let depth_diff = (rz - depth).abs() / depth;
                    let source_normal = world_normal(j, source_map.normal[q]);
                    let cos_angle = normal.dot(&source_normal);
                    if reproj <= options.max_reproj_error
                        && depth_diff <= options.max_relative_depth_diff
                        && cos_angle >= cos_max_angle
                    {
                        agreeing.push((j, q));
                        let angle = cos_angle.clamp(-1.0, 1.0).acos();
                        consistency += (-(reproj + 200.0 * depth_diff + 10.0 * angle)).exp();
                        sum_point += source_point;
                        sum_normal += source_normal;
                        for (s, c) in sum_color.iter_mut().zip(source.rgb[q]) {
                            *s += f64::from(c);
                        }
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
                        (point, normal, view.rgb[p].map(f64::from))
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
            maps[s].depth = Vec::new();
            maps[s].normal = Vec::new();
            used[s] = Vec::new();
        }
    }
    points
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
