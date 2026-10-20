//! Binary PLY output of fused point clouds.

use std::io::Write;
use std::path::Path;

use crate::fusion::FusedPoint;

/// Writes `points` as a binary little-endian PLY with positions, normals
/// and colours (the layout COLMAP's `fused.ply` uses).
pub fn write_ply(path: &Path, points: &[FusedPoint]) -> std::io::Result<()> {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    write!(
        out,
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\n\
         property float x\nproperty float y\nproperty float z\n\
         property float nx\nproperty float ny\nproperty float nz\n\
         property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",
        points.len()
    )?;
    let mut record = Vec::with_capacity(27);
    for point in points {
        record.clear();
        for v in point.position.iter().chain(&point.normal) {
            record.extend_from_slice(&v.to_le_bytes());
        }
        record.extend_from_slice(&point.color);
        out.write_all(&record)?;
    }
    out.flush()
}
