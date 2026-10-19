//! A mesh from the fused depth maps: each view's depths that other views
//! agree with, fused into truncated signed distance fields and extracted
//! with surface nets ([`visloc_gsplat_core::mesh`]).
//!
//! A pixel's depth goes to the level whose voxel is closest to its
//! footprint (levels four times coarser each), so far, noisier surfaces
//! fuse at a coarser voxel; a coarser level's triangles near a finer
//! level's surface are dropped. Pixels without a depth are unknown, not
//! empty, so they carve no free space.

use std::collections::HashMap;

use nalgebra::Matrix3;
use visloc_gsplat_core::camera::{CameraView, PinholeCamera};
use visloc_gsplat_core::mesh::{DepthFrame, Tsdf};

pub use visloc_gsplat_core::mesh::Mesh;

use crate::workspace::{View, Workspace};

/// Each level's voxel is this many times the previous one's.
const LEVEL_RATIO: f64 = 4.0;

/// TSDF and extraction settings.
#[derive(Debug, Clone)]
pub struct MeshOptions {
    /// The finest level's voxel edge in world units; `None` takes
    /// `voxel_pixels` pixels' footprint at the median over the views of
    /// their sparse points' median depth.
    pub voxel: Option<f64>,
    /// A pixel's depth goes to the level whose voxel is nearest (in
    /// ratio) to this many times its footprint.
    pub voxel_pixels: f64,
    /// TSDF levels, each 4 times coarser than the one before.
    pub levels: usize,
    /// Truncation distance, in voxels.
    pub trunc_voxels: f32,
    /// Voxels fused from fewer views are left out of the surface.
    pub min_weight: f32,
    /// Connected components with fewer triangles (at the finest level;
    /// coarser levels scale it by their area per triangle) are dropped.
    pub min_component: usize,
    /// Voxels far in front of a view's surface count as free space.
    pub free_space: bool,
    /// Cells with unknown corners still get a vertex from their known
    /// edges.
    pub partial_cells: bool,
}

impl Default for MeshOptions {
    fn default() -> Self {
        Self {
            voxel: None,
            voxel_pixels: 1.0,
            levels: 1,
            trunc_voxels: 3.0,
            min_weight: 1.0,
            min_component: 50,
            free_space: true,
            partial_cells: true,
        }
    }
}

/// Fuses views' depth maps one at a time; [`TsdfMesher::finish`] extracts
/// the mesh.
pub struct TsdfMesher {
    levels: Vec<Tsdf>,
    voxel: f64,
    voxel_pixels: f64,
    min_component: usize,
    views: usize,
}

impl TsdfMesher {
    pub fn new(ws: &Workspace, options: &MeshOptions) -> Self {
        let voxel = options.voxel.unwrap_or_else(|| {
            let mut footprints: Vec<f64> = ws
                .views
                .iter()
                .map(|v| f64::from(v.median_depth) / v.fx)
                .collect();
            footprints.sort_by(f64::total_cmp);
            options.voxel_pixels * footprints.get(footprints.len() / 2).copied().unwrap_or(1.0)
        });
        let levels = (0..options.levels.max(1))
            .map(|k| {
                let v = (voxel * LEVEL_RATIO.powi(k as i32)) as f32;
                let mut tsdf = Tsdf::new(v, v * options.trunc_voxels, f32::INFINITY);
                tsdf.carve = false;
                tsdf.free_space = options.free_space;
                tsdf.partial_cells = options.partial_cells;
                tsdf.min_weight = options.min_weight;
                tsdf
            })
            .collect();
        Self {
            levels,
            voxel,
            voxel_pixels: options.voxel_pixels,
            min_component: options.min_component,
            views: 0,
        }
    }

    /// The finest level's voxel edge in world units.
    pub fn voxel(&self) -> f64 {
        self.voxel
    }

    /// Fuses `view`'s `depth` (camera z per pixel, 0 where unknown).
    pub fn integrate(&mut self, view: &View, depth: &[f32]) {
        let rotation: Matrix3<f32> = view.rotation.cast();
        let camera = PinholeCamera::new(
            view.width as u32,
            view.height as u32,
            view.fx as f32,
            view.fy as f32,
            view.cx as f32,
            view.cy as f32,
        );
        let camera_view = CameraView::new(rotation, view.translation.cast(), camera);
        let rgb: Vec<[f32; 3]> = view
            .rgb
            .iter()
            .map(|c| c.map(|v| f32::from(v) / 255.0))
            .collect();
        let top = self.levels.len() - 1;
        let level_of = |d: f32| -> usize {
            let target = self.voxel_pixels * f64::from(d) / view.fx;
            ((target / self.voxel).log(LEVEL_RATIO) + 0.5)
                .floor()
                .clamp(0.0, top as f64) as usize
        };
        for (k, tsdf) in self.levels.iter_mut().enumerate() {
            let level_depth: Vec<f32> = depth
                .iter()
                .map(|&d| if d > 0.0 && level_of(d) == k { d } else { 0.0 })
                .collect();
            if level_depth.iter().all(|&d| d == 0.0) {
                continue;
            }
            tsdf.integrate(&DepthFrame {
                view: &camera_view,
                depth: &level_depth,
                rgb: &rgb,
            });
        }
        self.views += 1;
    }

    /// The surface, without small components, and a one-line summary.
    pub fn finish(self) -> (Mesh, String) {
        let mut mesh = Mesh::default();
        let mut parts = Vec::new();
        // Vertices of the finer levels, on a grid of the current level's
        // voxel: a coarser triangle with a vertex within one of its voxels
        // of a finer vertex is dropped.
        let mut finer: Vec<[f32; 3]> = Vec::new();
        for (k, tsdf) in self.levels.iter().enumerate() {
            let mut level = tsdf.extract();
            let area = LEVEL_RATIO.powi(2 * k as i32);
            level.remove_small_components(
                ((self.min_component as f64 / area).ceil() as usize).max(1),
            );
            let v = self.voxel * LEVEL_RATIO.powi(k as i32);
            let cell = |p: [f32; 3]| p.map(|c| (f64::from(c) / v).floor() as i64);
            let mut grid: HashMap<[i64; 3], Vec<[f32; 3]>> = HashMap::new();
            for &p in &finer {
                grid.entry(cell(p)).or_default().push(p);
            }
            let r2 = (v * v) as f32;
            let near_finer = |q: [f32; 3]| {
                let c = cell(q);
                (-1..=1).any(|dx| {
                    (-1..=1).any(|dy| {
                        (-1..=1).any(|dz| {
                            grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz])
                                .is_some_and(|ps| {
                                    ps.iter().any(|p| {
                                        (0..3).map(|i| (p[i] - q[i]).powi(2)).sum::<f32>() <= r2
                                    })
                                })
                        })
                    })
                })
            };
            let before = level.triangles.len();
            if !finer.is_empty() {
                let keep: Vec<bool> = level.vertices.iter().map(|&q| !near_finer(q)).collect();
                level
                    .triangles
                    .retain(|t| t.iter().all(|&i| keep[i as usize]));
            }
            parts.push(format!(
                "level {k} (voxel {v:.4}): {} blocks, {before} -> {} triangles",
                tsdf.num_blocks(),
                level.triangles.len()
            ));
            let offset = mesh.vertices.len() as u32;
            for t in &level.triangles {
                for &i in t {
                    finer.push(level.vertices[i as usize]);
                }
            }
            mesh.vertices.extend_from_slice(&level.vertices);
            mesh.colors.extend_from_slice(&level.colors);
            mesh.triangles
                .extend(level.triangles.iter().map(|t| t.map(|i| i + offset)));
        }
        let summary = format!(
            "mesh: {} views; {}; {} triangles",
            self.views,
            parts.join("; "),
            mesh.triangles.len()
        );
        (mesh, summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::{fuse_with, FusionOptions};
    use crate::patch_match::{plane_scene, plane_scene_truth, DepthMap};

    /// The plane scene's exact maps mesh onto the plane z = 4, through the
    /// depths fusion hands over.
    #[test]
    fn exact_maps_mesh_onto_the_plane() {
        let ws = plane_scene();
        let maps: Vec<DepthMap> = (0..ws.views.len())
            .map(|v| plane_scene_truth(&ws, v))
            .collect();
        let mut mesher = TsdfMesher::new(
            &ws,
            &MeshOptions {
                voxel: Some(0.02),
                min_component: 10,
                ..MeshOptions::default()
            },
        );
        let points = fuse_with(&ws, maps, &FusionOptions::default(), &mut |v, depth| {
            mesher.integrate(&ws.views[v], depth)
        });
        assert!(!points.is_empty());
        let (mesh, _) = mesher.finish();
        assert!(
            mesh.triangles.len() > 100,
            "{} triangles",
            mesh.triangles.len()
        );
        for t in &mesh.triangles {
            for &i in t {
                let z = mesh.vertices[i as usize][2];
                assert!((z - 4.0).abs() < 0.01, "vertex at z = {z}");
            }
        }
    }
}
