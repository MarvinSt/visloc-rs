//! Two-view geometry of spherical cameras (e.g.
//! [`visloc_core::types::CameraModel::Equirectangular`]), on unit bearing
//! vectors.
//!
//! The pinhole verifier ([`super::colmap_verification`]) measures
//! fundamental-matrix and homography errors in pixels and essential-matrix
//! errors on the `z = 1` plane, none of which exist for rays behind the
//! camera. Here only the essential matrix is fitted (a spherical camera
//! is calibrated by construction), from the five-point solver on unit
//! rays, and each correspondence is scored by the angle between its rays
//! and their epipolar planes.

use nalgebra::{Matrix3, Vector3};
use rand::rngs::SmallRng;
use rand::seq::index::sample;
use rand::SeedableRng;

use super::colmap_verification::pose_from_essential_matrix;
use super::five_point::relpose_5pt;

/// Settings for [`estimate_spherical_relative_pose`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalRansacConfig {
    /// Largest angle, in radians, between a ray and its epipolar plane (in
    /// both images) for an inlier.
    pub max_angle: f64,
    pub iterations: usize,
    pub seed: u64,
}

impl Default for SphericalRansacConfig {
    fn default() -> Self {
        Self {
            max_angle: 4.0e-3,
            iterations: 256,
            seed: 7,
        }
    }
}

/// [`estimate_spherical_relative_pose`]'s result: `x2 = rotation * x1 +
/// translation` up to scale (`|translation| = 1`), and the
/// correspondences that fit the essential matrix and triangulate in front
/// of both rays.
#[derive(Debug, Clone, PartialEq)]
pub struct SphericalRelativePose {
    pub essential: Matrix3<f64>,
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    pub inliers: Vec<usize>,
}

/// Sine of the larger of the two angles between a ray and the epipolar
/// plane of its partner (`x2ᵀ E x1 = 0`).
pub fn epipolar_angle(essential: &Matrix3<f64>, x1: &Vector3<f64>, x2: &Vector3<f64>) -> f64 {
    let n2 = essential * x1;
    let n1 = essential.transpose() * x2;
    let residual = x2.dot(&n2).abs();
    let (l1, l2) = (n1.norm() * x1.norm(), n2.norm() * x2.norm());
    if l1 < f64::EPSILON || l2 < f64::EPSILON {
        return f64::INFINITY;
    }
    (residual / l1).max(residual / l2)
}

/// Essential-matrix RANSAC on unit rays (`rays1[i]` in camera 1 matches
/// `rays2[i]` in camera 2), refitted on the inliers of the best minimal
/// model, then decomposed with the cheirality test on the rays. `None`
/// with fewer than five correspondences or no model.
pub fn estimate_spherical_relative_pose(
    rays1: &[Vector3<f64>],
    rays2: &[Vector3<f64>],
    config: &SphericalRansacConfig,
) -> Option<SphericalRelativePose> {
    let n = rays1.len();
    if n < 5 || rays2.len() != n {
        return None;
    }
    let threshold = config.max_angle.sin();
    let inliers_of = |e: &Matrix3<f64>| -> Vec<usize> {
        (0..n)
            .filter(|&i| epipolar_angle(e, &rays1[i], &rays2[i]) <= threshold)
            .collect()
    };
    let mut rng = SmallRng::seed_from_u64(config.seed);
    let mut best: Option<(Matrix3<f64>, Vec<usize>)> = None;
    let (mut s1, mut s2) = (Vec::with_capacity(5), Vec::with_capacity(5));
    for _ in 0..config.iterations.max(1) {
        s1.clear();
        s2.clear();
        for i in sample(&mut rng, n, 5) {
            s1.push(rays1[i]);
            s2.push(rays2[i]);
        }
        for e in relpose_5pt(&s1, &s2) {
            let inliers = inliers_of(&e);
            if best.as_ref().is_none_or(|b| inliers.len() > b.1.len()) {
                best = Some((e, inliers));
            }
        }
    }
    let (mut essential, mut inliers) = best?;
    // Refit on every inlier while that keeps or grows the inlier set.
    for _ in 0..3 {
        if inliers.len() <= 5 {
            break;
        }
        let r1: Vec<_> = inliers.iter().map(|&i| rays1[i]).collect();
        let r2: Vec<_> = inliers.iter().map(|&i| rays2[i]).collect();
        let Some((e, refit)) = relpose_5pt(&r1, &r2)
            .into_iter()
            .map(|e| {
                let refit = inliers_of(&e);
                (e, refit)
            })
            .max_by_key(|(_, refit)| refit.len())
        else {
            break;
        };
        if refit.len() < inliers.len() {
            break;
        }
        let grew = refit.len() > inliers.len();
        (essential, inliers) = (e, refit);
        if !grew {
            break;
        }
    }
    let r1: Vec<_> = inliers.iter().map(|&i| rays1[i]).collect();
    let r2: Vec<_> = inliers.iter().map(|&i| rays2[i]).collect();
    let (rotation, translation, valid) = pose_from_essential_matrix(&essential, &r1, &r2);
    if valid.is_empty() {
        return None;
    }
    Some(SphericalRelativePose {
        essential,
        rotation,
        translation,
        inliers: valid.into_iter().map(|k| inliers[k]).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Vector3};

    /// Rays all around both cameras, including behind them, with 20%
    /// outliers: the pose is recovered and the outliers rejected.
    #[test]
    fn recovers_the_pose_from_rays_over_the_whole_sphere() {
        let rotation = Rotation3::from_euler_angles(0.1, 2.5, -0.2).into_inner();
        let centre2 = Vector3::new(0.8, 0.1, -0.3);
        // x2 = R (X - c2), so t = -R c2.
        let translation = -(rotation * centre2);
        let mut state = 12345u64;
        let mut uniform = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let (mut rays1, mut rays2, mut outlier) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..200 {
            let point = Vector3::new(uniform(), uniform(), uniform()) * 6.0;
            if point.norm() < 1.5 || (point - centre2).norm() < 1.5 {
                continue;
            }
            rays1.push(point.normalize());
            let x2 = if i % 5 == 0 {
                Vector3::new(uniform(), uniform(), uniform()).normalize()
            } else {
                (rotation * (point - centre2)).normalize()
            };
            rays2.push(x2);
            outlier.push(i % 5 == 0);
        }
        assert!(rays1.iter().any(|r| r.z < -0.5), "rays behind camera 1");
        let pose = estimate_spherical_relative_pose(&rays1, &rays2, &Default::default()).unwrap();
        let angle = Rotation3::from_matrix_unchecked(pose.rotation.transpose() * rotation).angle();
        assert!(angle < 1e-9, "rotation off by {angle}");
        assert!((pose.translation - translation.normalize()).norm() < 1e-9);
        assert!(pose.inliers.iter().all(|&i| !outlier[i]), "an outlier kept");
        let good = outlier.iter().filter(|&&o| !o).count();
        assert_eq!(pose.inliers.len(), good);
    }
}
