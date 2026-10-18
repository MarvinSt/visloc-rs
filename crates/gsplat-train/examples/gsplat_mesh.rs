//! Extract a coloured triangle mesh from a trained splat.
//!
//! ```text
//! cargo run --release -p visloc-gsplat-train --features gpu --example gsplat_mesh -- \
//!     --ply ours.ply --data <colmap_root> --out mesh.ply \
//!     [--voxel V] [--trunc-voxels 4] [--max-depth D] [--min-component 500] \
//!     [--min-weight 3] [--no-carve] [--support-voxels 10]
//! ```
//!
//! Renders the median depth of the splat from every dataset view, fuses the
//! depths into a sparse TSDF and extracts the surface with surface nets (see
//! `visloc_gsplat_train::mesh`). Pixels with no opaque surface carve free
//! space, and voxels seen from fewer than `--min-weight` views are dropped.
//! Triangles farther than `--support-voxels` from every SfM point of the
//! dataset are dropped too (sky and other invented surfaces; 0 keeps them).
//! Defaults are relative to the scene: `scale` = the median distance of the
//! camera centres from their centroid, or half the 90th percentile depth of
//! the SfM points the views see if larger (cameras close together looking
//! out); voxel = scale / 256, max depth = 2 * scale.

use std::path::PathBuf;

use visloc_gsplat_core::ply::load_ply;
use visloc_gsplat_render::GpuContext;
use visloc_gsplat_train::dataset::load_colmap_dataset;
use visloc_gsplat_train::mesh::{extract_from_splat, rig_scale, scene_scale, MeshOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut ply: Option<PathBuf> = None;
    let mut data: Option<PathBuf> = None;
    let mut out = PathBuf::from("mesh.ply");
    let mut voxel: Option<f32> = None;
    let mut trunc_voxels = 4.0f32;
    let mut max_depth: Option<f32> = None;
    let mut min_component = 500usize;
    let mut carve = true;
    let mut min_weight = 3.0f32;
    let mut support_voxels = 10.0f32;
    let num = |v: Option<String>, flag: &str| -> Result<f32, String> {
        v.and_then(|s| s.parse().ok())
            .ok_or_else(|| format!("{flag} needs a number"))
    };
    while let Some(a) = args.next() {
        match a.as_str() {
            "--ply" => ply = args.next().map(PathBuf::from),
            "--data" => data = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from).ok_or("--out needs a path")?,
            "--voxel" => voxel = Some(num(args.next(), "--voxel")?),
            "--trunc-voxels" => trunc_voxels = num(args.next(), "--trunc-voxels")?,
            "--max-depth" => max_depth = Some(num(args.next(), "--max-depth")?),
            "--min-component" => min_component = num(args.next(), "--min-component")? as usize,
            "--no-carve" => carve = false,
            "--min-weight" => min_weight = num(args.next(), "--min-weight")?,
            "--support-voxels" => support_voxels = num(args.next(), "--support-voxels")?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let ply = ply.ok_or("--ply <path> is required")?;
    let data = data.ok_or("--data <colmap root> is required")?;

    let scene = load_ply(&ply)?;
    let dataset = load_colmap_dataset(&data, None)?;
    let views: Vec<_> = dataset
        .train
        .iter()
        .chain(&dataset.eval)
        .map(|v| &v.camera)
        .collect();
    let support: Vec<[f32; 3]> = dataset
        .init
        .gaussians
        .iter()
        .map(|g| [g.mean.x, g.mean.y, g.mean.z])
        .collect();
    let opts = MeshOptions {
        voxel,
        trunc_voxels,
        max_depth,
        carve,
        min_weight,
        support_voxels,
        min_component,
    };
    println!(
        "{} gaussians, {} views, {} SfM points; rig scale {:.3}, scene scale {:.3}",
        scene.len(),
        views.len(),
        support.len(),
        rig_scale(&views),
        scene_scale(&views, &support)
    );
    let (mesh, summary, _) =
        extract_from_splat(GpuContext::new()?, &scene, &views, &support, &opts)?;
    println!("{summary}");
    mesh.write_ply(&out)?;
    println!("wrote {}", out.display());
    Ok(())
}
