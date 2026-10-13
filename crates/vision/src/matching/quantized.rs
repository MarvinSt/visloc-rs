//! Integer kernel for [`super::BruteForceMatcher::match_descriptors_cross_checked`]
//! on byte-quantized descriptors.
//!
//! SIFT descriptors stored as bytes over a power-of-two scale (COLMAP's
//! `L1_ROOT` bytes / 512, as `gsplat_photos` keeps them, or the raw bytes)
//! have exact dot products in `f32` as long as every partial sum stays below
//! 2^24 units — true for up to 258 dimensions of values in `[0, 255]`. The
//! f32 matrix product therefore computes them exactly, and so does a 16-bit
//! integer multiply-add (`pmaddwd`, 16 lanes on AVX2), which does twice the
//! work per instruction. The top-2 scan then runs on 4 x 16 tiles straight
//! out of the kernel instead of over a stored `n_query x n_train` matrix.
//! Scores and visit order match the f32 path exactly, so the matches are
//! bit-identical.

use wide::{f32x8, i16x16, i32x8, CmpLt};

use super::Top2;

/// Query rows and train rows per kernel tile.
const QUERY_TILE: usize = 4;
const TRAIN_TILE: usize = 16;

/// The scale that maps every value to an integer in `[0, 255]`, if one of
/// the supported ones does and the dimension keeps dot products exact.
pub(super) fn byte_scale(query: &[Vec<f32>], train: &[Vec<f32>], dim: usize) -> Option<f32> {
    if dim == 0 || !dim.is_multiple_of(2) || dim > 258 {
        return None;
    }
    [512.0f32, 1.0].into_iter().find(|&scale| {
        query.iter().chain(train).all(|row| {
            row.iter().all(|&value| {
                let scaled = value * scale;
                (0.0..=255.0).contains(&scaled) && scaled.fract() == 0.0
            })
        })
    })
}

/// Top-2 scores in both directions: forward `train_norm_sq[t] - 2 dot` per
/// query row, reverse `query_norm_sq[q] - 2 dot` per train row, with the
/// strict-`<` first-wins order of the f32 path (each query sees train rows in
/// increasing order and vice versa).
pub(super) fn top2(
    query: &[Vec<f32>],
    train: &[Vec<f32>],
    dim: usize,
    scale: f32,
    query_norm_sq: &[f32],
    train_norm_sq: &[f32],
) -> Top2 {
    let (n_query, n_train) = (query.len(), train.len());
    let pairs = dim / 2;
    let query_rows = quantize_rows(query, dim, scale, n_query.next_multiple_of(QUERY_TILE));
    let packed_train = pack_train(train, dim, scale, n_train.next_multiple_of(TRAIN_TILE));
    let block_len = pairs * TRAIN_TILE;
    let inv_scale_sq = 1.0 / (scale * scale);

    let mut forward_best = vec![(usize::MAX, f32::INFINITY); n_query];
    let mut forward_second = vec![(usize::MAX, f32::INFINITY); n_query];
    let mut reverse_best = vec![(usize::MAX, f32::INFINITY); n_train];
    let mut reverse_second = vec![f32::INFINITY; n_train];
    let train_norm_padded: Vec<f32> = (0..n_train.next_multiple_of(TRAIN_TILE))
        .map(|t| train_norm_sq.get(t).copied().unwrap_or(f32::INFINITY))
        .collect();

    for q0 in (0..n_query).step_by(QUERY_TILE) {
        let rows: [&[i16]; QUERY_TILE] =
            std::array::from_fn(|i| &query_rows[(q0 + i) * dim..(q0 + i + 1) * dim]);
        for (tile, t0) in (0..n_train).step_by(TRAIN_TILE).enumerate() {
            let block = &packed_train[tile * 2 * block_len..(tile + 1) * 2 * block_len];
            let dots = kernel(rows, &block[..block_len], &block[block_len..], pairs);
            for (i, halves) in dots.iter().enumerate() {
                let q = q0 + i;
                if q >= n_query {
                    break;
                }
                for (h, dot) in halves.iter().enumerate() {
                    let tb = t0 + 8 * h;
                    if tb >= n_train {
                        break;
                    }
                    let dot = dot.round_float() * inv_scale_sq;
                    let twice = dot * 2.0;
                    let lanes = (n_train - tb).min(8);

                    // Forward: scores of query q against train rows tb..tb+8.
                    let norms: [f32; 8] = train_norm_padded[tb..tb + 8].try_into().unwrap();
                    let forward = f32x8::new(norms) - twice;
                    if forward.cmp_lt(f32x8::splat(forward_second[q].1)).any() {
                        let scores = forward.to_array();
                        for (l, &score) in scores.iter().enumerate().take(lanes) {
                            let (best, second) = (&mut forward_best[q], &mut forward_second[q]);
                            if score < best.1 {
                                *second = *best;
                                *best = (tb + l, score);
                            } else if score < second.1 {
                                *second = (tb + l, score);
                            }
                        }
                    }

                    // Reverse: scores of train rows tb..tb+8 against query q.
                    let reverse = f32x8::splat(query_norm_sq[q]) - twice;
                    let mut seconds = [f32::INFINITY; 8];
                    seconds[..lanes].copy_from_slice(&reverse_second[tb..tb + lanes]);
                    if reverse.cmp_lt(f32x8::new(seconds)).any() {
                        let scores = reverse.to_array();
                        for (l, &score) in scores.iter().enumerate().take(lanes) {
                            let t = tb + l;
                            if score < reverse_best[t].1 {
                                reverse_second[t] = reverse_best[t].1;
                                reverse_best[t] = (q, score);
                            } else if score < reverse_second[t] {
                                reverse_second[t] = score;
                            }
                        }
                    }
                }
            }
        }
    }

    let found = |(index, score): (usize, f32)| (index != usize::MAX).then_some((index, score));
    Top2 {
        forward_best: forward_best.into_iter().map(found).collect(),
        forward_second: forward_second.into_iter().map(found).collect(),
        reverse_best: reverse_best.into_iter().map(found).collect(),
        reverse_second: reverse_second
            .into_iter()
            .map(|score| (score != f32::INFINITY).then_some(score))
            .collect(),
    }
}

/// Rows as contiguous `i16`, zero rows appended up to `padded`.
fn quantize_rows(rows: &[Vec<f32>], dim: usize, scale: f32, padded: usize) -> Vec<i16> {
    let mut out = vec![0i16; padded * dim];
    for (row, chunk) in rows.iter().zip(out.chunks_exact_mut(dim)) {
        for (value, slot) in row.iter().zip(chunk) {
            *slot = (value * scale) as i16;
        }
    }
    out
}

/// Train rows in blocks of 8 with dimensions interleaved in pairs, so one
/// 16-lane load holds dimensions `2p, 2p + 1` of eight rows: row `r`,
/// dimension `2p + e` sits at `((r / 8 * pairs + p) * 8 + r % 8) * 2 + e`.
fn pack_train(rows: &[Vec<f32>], dim: usize, scale: f32, padded: usize) -> Vec<i16> {
    let pairs = dim / 2;
    let mut out = vec![0i16; padded * dim];
    for (r, row) in rows.iter().enumerate() {
        let (block, lane) = (r / 8, r % 8);
        for p in 0..pairs {
            let at = ((block * pairs + p) * 8 + lane) * 2;
            out[at] = (row[2 * p] * scale) as i16;
            out[at + 1] = (row[2 * p + 1] * scale) as i16;
        }
    }
    out
}

/// Dot products of four query rows with two packed blocks of eight train
/// rows: `[query][block]`, one lane per train row.
#[inline]
fn kernel(
    rows: [&[i16]; QUERY_TILE],
    block0: &[i16],
    block1: &[i16],
    pairs: usize,
) -> [[i32x8; 2]; QUERY_TILE] {
    let mut acc = [[i32x8::ZERO; 2]; QUERY_TILE];
    for p in 0..pairs {
        let t0 = i16x16::new(block0[p * 16..p * 16 + 16].try_into().unwrap());
        let t1 = i16x16::new(block1[p * 16..p * 16 + 16].try_into().unwrap());
        for (row, acc) in rows.iter().zip(acc.iter_mut()) {
            let (a, b) = (row[2 * p], row[2 * p + 1]);
            let pair = i16x16::new([a, b, a, b, a, b, a, b, a, b, a, b, a, b, a, b]);
            acc[0] += pair.dot(t0);
            acc[1] += pair.dot(t1);
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::{byte_scale, top2};
    use crate::matching::BruteForceMatcher;

    /// Byte descriptors over `scale`; a small `range` forces score ties.
    fn rows(count: usize, dim: usize, salt: usize, range: usize, scale: f32) -> Vec<Vec<f32>> {
        (0..count)
            .map(|row| {
                (0..dim)
                    .map(|k| ((row * 131 + k * 17 + salt * 29) % 251 % range) as f32 / scale)
                    .collect()
            })
            .collect()
    }

    fn norms(rows: &[Vec<f32>]) -> Vec<f32> {
        rows.iter()
            .map(|row| row.iter().map(|value| value * value).sum())
            .collect()
    }

    #[test]
    fn integer_kernel_finds_the_f32_scores() {
        for (n_query, n_train, dim, range, scale) in [
            (1, 1, 2, 256, 512.0),
            (5, 3, 4, 3, 512.0),
            (13, 37, 128, 256, 512.0),
            (17, 16, 128, 4, 512.0),
            (40, 33, 64, 256, 1.0),
            (9, 70, 258, 256, 1.0),
        ] {
            let query = rows(n_query, dim, 3, range, scale);
            let train = rows(n_train, dim, 7, range, scale);
            assert_eq!(byte_scale(&query, &train, dim), Some(scale));
            let (qn, tn) = (norms(&query), norms(&train));
            assert_eq!(
                top2(&query, &train, dim, scale, &qn, &tn),
                BruteForceMatcher::top2_f32(&query, &train, dim, &qn, &tn),
                "{n_query}x{n_train}, dim {dim}, range {range}"
            );
        }
    }

    #[test]
    fn only_exact_byte_descriptors_take_the_integer_kernel() {
        let bytes = rows(4, 8, 1, 256, 512.0);
        assert_eq!(byte_scale(&bytes, &bytes, 8), Some(512.0));
        let mut off_grid = bytes.clone();
        off_grid[2][3] += 0.25 / 512.0;
        assert_eq!(byte_scale(&bytes, &off_grid, 8), None);
        let mut negative = bytes.clone();
        negative[0][0] = -1.0 / 512.0;
        assert_eq!(byte_scale(&negative, &bytes, 8), None);
        let odd = rows(4, 7, 1, 256, 512.0);
        assert_eq!(byte_scale(&odd, &odd, 7), None);
        let wide = rows(2, 260, 1, 256, 1.0);
        assert_eq!(byte_scale(&wide, &wide, 260), None);
    }
}
