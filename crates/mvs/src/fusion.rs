//! Depth-map fusion into a point cloud (after ACMM, Gipuma and COLMAP):
//! each pixel's 3D point is checked against the depth maps of its view's
//! source views and kept when enough of them agree.

use nalgebra::Vector3;

use crate::patch_match::DepthMap;
use crate::workspace::Workspace;

/// When a source view's depth agrees with a pixel's, and how many must.
///
/// By default one agreeing source view within 15° of normal angle keeps a
/// point; ACMM's code asks for two (and 10°). Over the 13 ETH3D training
/// scenes at 3200 px (multi-scale), F1 at 2 cm is 78.0% with these
/// defaults, 77.3% at 30°, 76.4% without the normal check and 76.0% with
/// two views at 30°. On pipes at 1600 px, one view instead of two raises F1
/// from 66.9% to 76.0%.
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
}

impl Default for FusionOptions {
    fn default() -> Self {
        Self {
            max_reproj_error: 2.0,
            max_relative_depth_diff: 0.01,
            max_normal_angle_deg: 15.0,
            min_consistent_views: 1,
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

/// Fuses `maps` (indexed like `ws.views`). Views are visited in order and
/// pixels in raster order; a pixel used by a point is not used again.
pub fn fuse(ws: &Workspace, maps: &[DepthMap], options: &FusionOptions) -> Vec<FusedPoint> {
    let cos_max_angle = options.max_normal_angle_deg.to_radians().cos();
    let mut used: Vec<Vec<bool>> = maps.iter().map(|m| vec![false; m.depth.len()]).collect();
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
                if used[i][p] || depth <= 0.0 {
                    continue;
                }
                let (u, v) = (x as f64 + 0.5, y as f64 + 0.5);
                let point = view.unproject(u, v, depth);
                let normal = world_normal(i, map.normal[p]);
                agreeing.clear();
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
                    if used[j][q] || source_depth <= 0.0 {
                        continue;
                    }
                    let source_point =
                        source.unproject(sx as f64 + 0.5, sy as f64 + 0.5, source_depth);
                    let (ru, rv, rz) = view.project(&source_point);
                    let reproj = ((ru - u).powi(2) + (rv - v).powi(2)).sqrt();
                    let depth_diff = (rz - depth).abs() / depth;
                    let source_normal = world_normal(j, source_map.normal[q]);
                    if reproj <= options.max_reproj_error
                        && depth_diff <= options.max_relative_depth_diff
                        && normal.dot(&source_normal) >= cos_max_angle
                    {
                        agreeing.push((j, q));
                        sum_point += source_point;
                        sum_normal += source_normal;
                        for (s, c) in sum_color.iter_mut().zip(source.rgb[q]) {
                            *s += f64::from(c);
                        }
                    }
                }
                if agreeing.len() >= options.min_consistent_views {
                    let count = (agreeing.len() + 1) as f64;
                    let position = sum_point / count;
                    let normal = sum_normal.normalize();
                    points.push(FusedPoint {
                        position: [position.x as f32, position.y as f32, position.z as f32],
                        normal: [normal.x as f32, normal.y as f32, normal.z as f32],
                        color: sum_color.map(|s| (s / count).round().clamp(0.0, 255.0) as u8),
                    });
                    used[i][p] = true;
                    for &(j, q) in &agreeing {
                        used[j][q] = true;
                    }
                }
            }
        }
    }
    points
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
        let points = fuse(&ws, &maps, &FusionOptions::default());
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
        assert!(fuse(&ws, &maps, &strict).is_empty());
    }
}
