//! Mesh extraction from a trained splat: its median depth rendered from
//! every view, fused by [`visloc_gsplat_core::mesh`]'s TSDF and extracted
//! with surface nets.

use visloc_gsplat_core::camera::CameraView;
pub use visloc_gsplat_core::mesh::{rig_scale, DepthFrame, Mesh, Tsdf};

/// Settings for [`extract_from_splat`]; lengths in voxels unless noted.
#[derive(Debug, Clone)]
pub struct MeshOptions {
    /// Voxel edge (world units); `None`: rig scale / 256.
    pub voxel: Option<f32>,
    pub trunc_voxels: f32,
    /// Ignore depths beyond this (world units); `None`: 2 x rig scale.
    pub max_depth: Option<f32>,
    pub carve: bool,
    pub min_weight: f32,
    /// Drop triangles farther than this from every support point (0: off).
    pub support_voxels: f32,
    pub min_component: usize,
}

impl Default for MeshOptions {
    fn default() -> Self {
        Self {
            voxel: None,
            trunc_voxels: 4.0,
            max_depth: None,
            carve: true,
            min_weight: 3.0,
            support_voxels: 10.0,
            min_component: 500,
        }
    }
}

/// Render `scene`'s median depth from every view, fuse, extract, filter by
/// `support` (e.g. SfM points) and drop small components. Returns the mesh
/// and a one-line summary.
pub fn extract_from_splat(
    ctx: visloc_gsplat_render::GpuContext,
    scene: &visloc_gsplat_core::gaussian::Scene,
    views: &[&CameraView],
    support: &[[f32; 3]],
    opts: &MeshOptions,
) -> Result<(Mesh, String, visloc_gsplat_render::GpuContext), visloc_gsplat_render::GpuError> {
    let scale = rig_scale(views);
    let voxel = opts.voxel.unwrap_or(scale / 256.0);
    let max_depth = opts.max_depth.unwrap_or(2.0 * scale);
    let mut tsdf = Tsdf::new(voxel, voxel * opts.trunc_voxels, max_depth);
    tsdf.carve = opts.carve;
    tsdf.min_weight = opts.min_weight;
    let mut ctx = Some(ctx);
    let mut renderer: Option<(u32, u32, visloc_gsplat_render::Renderer)> = None;
    let t0 = std::time::Instant::now();
    let (mut render_s, mut integrate_s) = (0.0f64, 0.0f64);
    for view in views {
        let (w, h) = (view.camera.width, view.camera.height);
        if renderer.as_ref().map(|r| (r.0, r.1)) != Some((w, h)) {
            let c = match renderer.take() {
                Some((_, _, r)) => r.into_context(),
                None => ctx.take().expect("context"),
            };
            renderer = Some((w, h, visloc_gsplat_render::Renderer::new(c, scene, w, h)?));
        }
        let (_, _, r) = renderer.as_mut().expect("renderer set above");
        let t = std::time::Instant::now();
        let (image, depth, _) = r.render_depth(view, [0.0, 0.0, 0.0]);
        render_s += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        tsdf.integrate(&DepthFrame {
            view,
            depth: &depth,
            rgb: &image.rgb,
        });
        integrate_s += t.elapsed().as_secs_f64();
    }
    let fused = t0.elapsed().as_secs_f64();
    let mut mesh = tsdf.extract();
    let raw = mesh.triangles.len();
    if opts.support_voxels > 0.0 && !support.is_empty() {
        mesh.keep_supported(support, opts.support_voxels * voxel);
    }
    mesh.remove_small_components(opts.min_component);
    let summary = format!(
        "mesh: {} views fused in {fused:.1} s (render {render_s:.1} s, integrate {integrate_s:.1} s; voxel {voxel:.4}, {} blocks), {raw} -> {} triangles",
        views.len(),
        tsdf.num_blocks(),
        mesh.triangles.len()
    );
    let ctx = match renderer {
        Some((_, _, r)) => r.into_context(),
        None => ctx.take().expect("context"),
    };
    Ok((mesh, summary, ctx))
}
