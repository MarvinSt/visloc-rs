//! Time `extract_sift` with `gsplat_photos`' SIFT settings on a binary PGM.
//!
//! ```text
//! cargo run --release -p visloc-vision --example sift_vlfeat_bench -- <image.pgm> [runs]
//! ```

use std::time::Instant;

use visloc_vision::features::sift::{extract_sift, GrayImage, SiftConfig, SiftNormalization};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: sift_vlfeat_bench <image.pgm> [runs]");
    let runs: usize = args.next().map_or(3, |r| r.parse().expect("runs"));
    let bytes = std::fs::read(&path).expect("read pgm");
    // Binary PGM: "P5\n<w> <h>\n255\n" then w*h bytes.
    let mut fields = Vec::new();
    let mut i = 0;
    while fields.len() < 4 {
        while bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let start = i;
        while !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        fields.push(String::from_utf8_lossy(&bytes[start..i]).into_owned());
    }
    assert_eq!(fields[0], "P5");
    let (w, h): (usize, usize) = (fields[1].parse().unwrap(), fields[2].parse().unwrap());
    let pixels: Vec<f32> = bytes[i + 1..i + 1 + w * h]
        .iter()
        .map(|&b| b as f32 / 255.0)
        .collect();
    let image = GrayImage::new(w, h, &pixels).expect("image");
    let config = SiftConfig {
        max_keypoints: 8000,
        normalization: SiftNormalization::L1Root,
        descriptor_magnification: 3.0,
        max_orientations: 2,
        prefer_larger_scale: true,
        vlfeat_compatible_detector: true,
        vlfeat_compatible_descriptor: true,
        vlfeat_bilinear_orientations: true,
        ..SiftConfig::default()
    };
    for _ in 0..runs {
        let t = Instant::now();
        let (kps, desc) = extract_sift(&image, &config).expect("sift");
        let secs = t.elapsed().as_secs_f64();
        // FNV-1a over every keypoint field and descriptor value's bits, to
        // check that an optimization leaves the output bit-identical.
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |bits: u64| {
            hash ^= bits;
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        };
        for k in &kps {
            for v in [k.x, k.y, k.sigma, k.orientation] {
                eat(v.to_bits());
            }
        }
        for d in &desc {
            for v in d {
                eat(u64::from(v.to_bits()));
            }
        }
        println!(
            "{w}x{h}: {} keypoints in {secs:.3} s, checksum {hash:016x}",
            kps.len()
        );
    }
}
