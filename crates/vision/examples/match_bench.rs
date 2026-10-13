//! Times `BruteForceMatcher::match_descriptors_cross_checked` on synthetic
//! SIFT-like byte descriptors (values k / 512, as `gsplat_photos` keeps
//! them), through the integer kernel and, with one value nudged off the byte
//! grid, through the f32 matrix product.
//!
//! ```text
//! cargo run --release -p visloc-vision --example match_bench -- [n] [reps]
//! ```

use std::time::Instant;

use visloc_vision::matching::BruteForceMatcher;

fn descriptors(n: usize, seed: u64, base: Option<&[Vec<f32>]>) -> Vec<Vec<f32>> {
    let mut state = seed;
    let mut random = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    (0..n)
        .map(|i| {
            let raw: Vec<f32> = (0..128)
                .map(|k| match base {
                    // A third of the rows are noisy copies, so matches exist.
                    Some(base) if i % 3 == 0 => base[i][k] + 0.05 * random(),
                    _ => random().powi(3),
                })
                .collect();
            let l2 = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
            raw.iter()
                .map(|v| (v / l2 * 512.0).round().min(255.0) / 512.0)
                .collect()
        })
        .collect()
}

fn time(reps: usize, mut run: impl FnMut() -> usize) -> (f64, usize) {
    let mut best = f64::MAX;
    let mut count = 0;
    for _ in 0..reps {
        let start = Instant::now();
        count = run();
        best = best.min(start.elapsed().as_secs_f64());
    }
    (best, count)
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let n = args.first().copied().unwrap_or(11000);
    let reps = args.get(1).copied().unwrap_or(3);
    let query = descriptors(n, 0x9e37_79b9_7f4a_7c15, None);
    let train = descriptors(n, 0x0123_4567, Some(&query));
    let mut off_grid = train.clone();
    off_grid[0][0] += 1e-6;
    let matcher = BruteForceMatcher { ratio: Some(0.8) };
    let (int_s, int_n) = time(reps, || {
        matcher
            .match_descriptors_cross_checked(&query, &train)
            .len()
    });
    let (f32_s, f32_n) = time(reps, || {
        matcher
            .match_descriptors_cross_checked(&query, &off_grid)
            .len()
    });
    println!(
        "{n} x {n}: integer kernel {int_s:.3} s ({int_n} matches), \
         f32 product {f32_s:.3} s ({f32_n} matches), {:.2}x",
        f32_s / int_s
    );
}
