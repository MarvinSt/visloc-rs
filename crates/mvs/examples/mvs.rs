//! Multi-view stereo on a COLMAP sparse model with undistorted (pinhole)
//! images: PatchMatch depth maps, geometric-consistency passes, optionally
//! on an image pyramid, then fusion into `fused.ply`.
//!
//! ```text
//! cargo run --release -p visloc-mvs --example mvs -- \
//!     --sparse <dir with cameras.txt/images.txt/points3D.txt> \
//!     --images <image dir> --out <dir> [--max-image-size 3200] \
//!     [--sources 10] [--iterations 3] [--geom-passes 1] [--scales 1] [--gpu] \
//!     [--view-selection confidence|sampled] [--init-top-k 3] \
//!     [--save-depth] [--save-maps | --load-maps] [--min-consistent 2] \
//!     [--max-reproj 2] [--max-depth-diff 0.01] [--max-normal-angle 30]
//! ```
//!
//! `--scales 3 --geom-passes 2` runs ACMM's multi-scale scheme; `--gpu`
//! (a `--features gpu` build) runs PatchMatch on the GPU.

use std::path::PathBuf;
use std::time::Instant;

use visloc_mvs::{
    estimate, fuse, multi_scale_depth_maps, read_text_model, write_ply, FusionOptions,
    MultiScaleOptions, PatchMatchOptions, ViewSelection, Workspace, WorkspaceOptions,
};

fn main() {
    let mut sparse = None;
    let mut images = None;
    let mut out = None;
    let mut ws_options = WorkspaceOptions::default();
    let mut pm_options = PatchMatchOptions::default();
    let fusion_options = FusionOptions::default();
    let mut geom_passes = 1usize;
    let mut scales = 1usize;
    let mut use_gpu = false;
    let mut save_depth = false;
    let mut save_maps = false;
    let mut load_maps = false;
    let mut fusion_options = fusion_options;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--sparse" => sparse = Some(PathBuf::from(value())),
            "--images" => images = Some(PathBuf::from(value())),
            "--out" => out = Some(PathBuf::from(value())),
            "--max-image-size" => ws_options.max_image_size = Some(value().parse().unwrap()),
            "--sources" => ws_options.num_sources = value().parse().unwrap(),
            "--iterations" => pm_options.iterations = value().parse().unwrap(),
            "--view-selection" => {
                pm_options.view_selection = match value().as_str() {
                    "confidence" => ViewSelection::Confidence,
                    "sampled" => ViewSelection::Sampled,
                    other => panic!("--view-selection: confidence or sampled, not {other}"),
                }
            }
            "--init-top-k" => pm_options.init_top_k = value().parse().unwrap(),
            "--geom-passes" => geom_passes = value().parse().unwrap(),
            "--scales" => scales = value().parse().unwrap(),
            "--gpu" => use_gpu = true,
            "--save-depth" => save_depth = true,
            "--save-maps" => save_maps = true,
            "--load-maps" => load_maps = true,
            "--min-consistent" => fusion_options.min_consistent_views = value().parse().unwrap(),
            "--max-reproj" => fusion_options.max_reproj_error = value().parse().unwrap(),
            "--max-depth-diff" => fusion_options.max_relative_depth_diff = value().parse().unwrap(),
            "--max-normal-angle" => fusion_options.max_normal_angle_deg = value().parse().unwrap(),
            other => panic!("unknown argument {other}"),
        }
    }
    let sparse = sparse.expect("--sparse is required");
    let images = images.expect("--images is required");
    let out = out.expect("--out is required");
    std::fs::create_dir_all(&out).expect("creating the output directory");

    let started = Instant::now();
    let model = read_text_model(&sparse).expect("reading the sparse model");
    let ws = Workspace::load(&model, &images, &ws_options).expect("loading the workspace");
    eprintln!(
        "workspace: {} views at {}x{} in {:.1} s",
        ws.views.len(),
        ws.views.first().map_or(0, |v| v.width),
        ws.views.first().map_or(0, |v| v.height),
        started.elapsed().as_secs_f64()
    );

    #[cfg(feature = "gpu")]
    let gpu = use_gpu.then(|| {
        let gpu = visloc_mvs::GpuPatchMatch::new().expect("creating the GPU device");
        eprintln!("GPU: {}", gpu.adapter());
        gpu
    });
    #[cfg(not(feature = "gpu"))]
    assert!(!use_gpu, "--gpu needs a build with --features gpu");
    let run = |ws: &Workspace,
               view: usize,
               options: &PatchMatchOptions,
               prior: Option<&[visloc_mvs::DepthMap]>| {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &gpu {
            return gpu.estimate(ws, view, options, prior);
        }
        estimate(ws, view, options, prior)
    };
    let map_path = |v: usize| out.join(format!("map_{v:04}.bin"));
    let maps = if load_maps {
        (0..ws.views.len())
            .map(|v| visloc_mvs::DepthMap::read(&map_path(v)).expect("reading a saved depth map"))
            .collect()
    } else if scales > 1 {
        let options = MultiScaleOptions {
            patch_match: pm_options,
            scales,
            geom_passes,
            ..MultiScaleOptions::default()
        };
        multi_scale_depth_maps(&ws, &options, &run, &mut |line| eprintln!("{line}"))
    } else {
        let t = Instant::now();
        let mut maps: Vec<_> = (0..ws.views.len())
            .map(|v| run(&ws, v, &pm_options, None))
            .collect();
        eprintln!("photometric pass: {:.1} s", t.elapsed().as_secs_f64());
        for pass in 0..geom_passes {
            let t = Instant::now();
            maps = (0..ws.views.len())
                .map(|v| run(&ws, v, &pm_options, Some(&maps)))
                .collect();
            eprintln!(
                "geometric pass {}: {:.1} s",
                pass + 1,
                t.elapsed().as_secs_f64()
            );
        }
        maps
    };

    if save_maps && !load_maps {
        for (v, map) in maps.iter().enumerate() {
            map.write(&map_path(v)).expect("writing a depth map");
        }
    }
    if save_depth {
        // Grey depth images: near (the view's minimum depth) white, far
        // black, no depth black.
        for (view, map) in ws.views.iter().zip(&maps) {
            let (lo, hi) = view.depth_range;
            let pixels: Vec<u8> = map
                .depth
                .iter()
                .map(|&d| {
                    if d <= 0.0 {
                        0
                    } else {
                        (255.0 * (1.0 - (d - lo) / (hi - lo))).clamp(1.0, 255.0) as u8
                    }
                })
                .collect();
            let name = std::path::Path::new(&view.name)
                .file_stem()
                .map_or_else(|| "view".to_owned(), |s| s.to_string_lossy().into_owned());
            image::GrayImage::from_raw(map.width as u32, map.height as u32, pixels)
                .expect("depth image size")
                .save(out.join(format!("depth_{name}.png")))
                .expect("writing a depth image");
        }
    }

    let t = Instant::now();
    let points = fuse(&ws, &maps, &fusion_options);
    eprintln!(
        "fusion: {} points in {:.1} s",
        points.len(),
        t.elapsed().as_secs_f64()
    );
    let ply = out.join("fused.ply");
    write_ply(&ply, &points).expect("writing fused.ply");
    eprintln!(
        "wrote {} in {:.1} s total",
        ply.display(),
        started.elapsed().as_secs_f64()
    );
}
