//! Compare the VLFeat-compatible GPU SIFT with the CPU `extract_sift` on a
//! directory of images, with `gsplat_photos`' SIFT settings: wall time,
//! keypoint agreement and descriptor byte differences.
//!
//! ```text
//! cargo run --release -p visloc-sift-gpu --features gpu --example sift_vlfeat_gpu_bench -- \
//!     --images <dir> [--max-images 5] [--max-size 1024] [--no-cpu]
//! ```
//!
//! Images are resized so that their longer side is at most `--max-size`
//! and converted to grey as `gsplat_photos` does.

use std::path::PathBuf;
use std::time::Instant;

use visloc_sift_gpu::{GpuContext, VlfeatSiftGpu};
use visloc_vision::features::sift::{
    extract_sift, GrayImage, SiftConfig, SiftKeypoint, SiftNormalization,
};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn angle_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(std::f64::consts::TAU);
    d.min(std::f64::consts::TAU - d)
}

/// Keypoints of `a` with a counterpart in `b` (position within 0.01 px,
/// scale within 0.1%, orientation within 0.01 rad), as index pairs.
fn pair_up(a: &[SiftKeypoint], b: &[SiftKeypoint]) -> Vec<(usize, usize)> {
    // Bucket b by rounded position.
    let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> = Default::default();
    for (j, q) in b.iter().enumerate() {
        grid.entry((q.x.round() as i64, q.y.round() as i64))
            .or_default()
            .push(j);
    }
    let mut used = vec![false; b.len()];
    let mut pairs = Vec::new();
    for (i, p) in a.iter().enumerate() {
        let (cx, cy) = (p.x.round() as i64, p.y.round() as i64);
        let mut hit = None;
        'search: for gx in cx - 1..=cx + 1 {
            for gy in cy - 1..=cy + 1 {
                for &j in grid.get(&(gx, gy)).map_or(&[][..], |v| v.as_slice()) {
                    let q = &b[j];
                    if !used[j]
                        && (p.x - q.x).abs() < 0.01
                        && (p.y - q.y).abs() < 0.01
                        && (p.sigma - q.sigma).abs() < 1e-3 * p.sigma
                        && angle_diff(p.orientation, q.orientation) < 0.01
                    {
                        hit = Some(j);
                        break 'search;
                    }
                }
            }
        }
        if let Some(j) = hit {
            used[j] = true;
            pairs.push((i, j));
        }
    }
    pairs
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(arg(&args, "--images").expect("--images <dir>"));
    let max_images: usize = arg(&args, "--max-images").map_or(5, |v| v.parse().unwrap());
    let max_size: u32 = arg(&args, "--max-size").map_or(1024, |v| v.parse().unwrap());
    let run_cpu = !args.iter().any(|a| a == "--no-cpu");
    let config = SiftConfig {
        max_keypoints: 8192,
        normalization: SiftNormalization::L1Root,
        descriptor_magnification: 3.0,
        max_orientations: 2,
        prefer_larger_scale: true,
        vlfeat_compatible_detector: true,
        vlfeat_compatible_descriptor: true,
        vlfeat_bilinear_orientations: true,
        ..SiftConfig::default()
    };
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|e| {
                let e = e.to_string_lossy().to_ascii_lowercase();
                e == "png" || e == "jpg" || e == "jpeg"
            })
        })
        .collect();
    paths.sort();
    paths.truncate(max_images);

    let t = Instant::now();
    let mut gpu = VlfeatSiftGpu::new(GpuContext::new().expect("gpu"));
    eprintln!(
        "gpu: {} (init {:.2}s)",
        gpu.context().adapter_info.name,
        t.elapsed().as_secs_f64()
    );
    let (mut t_cpu, mut t_gpu) = (0.0, 0.0);
    let (mut n_cpu, mut n_gpu, mut n_pair) = (0usize, 0usize, 0usize);
    let (mut worst, mut diff_sum, mut diff_n) = (0.0f32, 0.0f64, 0usize);
    for path in &paths {
        let img = image::open(path).expect("image").to_rgb8();
        let (w, h) = img.dimensions();
        let scale = (max_size as f64 / w.max(h) as f64).min(1.0);
        let img = if scale < 1.0 {
            let (nw, nh) = (
                (w as f64 * scale).round() as u32,
                (h as f64 * scale).round() as u32,
            );
            image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Triangle)
        } else {
            img
        };
        let (w, h) = img.dimensions();
        let px: Vec<f32> = img
            .pixels()
            .map(|p| (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) / 255.0)
            .collect();
        let gray = GrayImage::new(w as usize, h as usize, &px).unwrap();
        let t = Instant::now();
        let (gk, gd) = gpu.extract(&gray, &config).expect("gpu sift");
        let tg = t.elapsed().as_secs_f64();
        t_gpu += tg;
        n_gpu += gk.len();
        let name = path.file_name().unwrap().to_string_lossy();
        if !run_cpu {
            println!("{name} {w}x{h}: gpu {} keypoints in {tg:.3} s", gk.len());
            continue;
        }
        let t = Instant::now();
        let (ck, cd) = extract_sift(&gray, &config).expect("cpu sift");
        let tc = t.elapsed().as_secs_f64();
        t_cpu += tc;
        n_cpu += ck.len();
        let pairs = pair_up(&ck, &gk);
        n_pair += pairs.len();
        let mut img_worst = 0.0f32;
        for &(i, j) in &pairs {
            for (a, b) in cd[i].iter().zip(&gd[j]) {
                let d = (a - b).abs();
                img_worst = img_worst.max(d);
                diff_sum += d as f64;
                diff_n += 1;
            }
        }
        worst = worst.max(img_worst);
        println!(
            "{name} {w}x{h}: cpu {} in {tc:.3} s, gpu {} in {tg:.3} s, paired {} ({:.2}% of cpu, {:.2}% of gpu), worst byte diff {img_worst}",
            ck.len(),
            gk.len(),
            pairs.len(),
            100.0 * pairs.len() as f64 / ck.len().max(1) as f64,
            100.0 * pairs.len() as f64 / gk.len().max(1) as f64,
        );
    }
    let n = paths.len().max(1) as f64;
    println!("gpu: {n_gpu} keypoints, {:.3} s per image", t_gpu / n);
    if run_cpu {
        println!(
            "cpu: {n_cpu} keypoints, {:.3} s per image; paired {n_pair} ({:.2}% of cpu, {:.2}% of gpu); descriptor bytes: worst diff {worst}, mean {:.5}",
            t_cpu / n,
            100.0 * n_pair as f64 / n_cpu.max(1) as f64,
            100.0 * n_pair as f64 / n_gpu.max(1) as f64,
            diff_sum / diff_n.max(1) as f64
        );
    }
}
