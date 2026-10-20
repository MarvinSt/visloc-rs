//! COLMAP text sparse models (`cameras.txt`, `images.txt`, `points3D.txt`)
//! as multi-view stereo reads them: each image's name and pose, and which
//! 3D points it observes.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};
use thiserror::Error;

/// Why a sparse model could not be read.
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("reading {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{file}: {message}: {line:?}")]
    Parse {
        file: &'static str,
        line: String,
        message: &'static str,
    },
}

/// One camera of `cameras.txt`.
#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    pub id: u32,
    pub model: String,
    pub width: u32,
    pub height: u32,
    pub params: Vec<f64>,
}

impl Camera {
    /// `(fx, fy, cx, cy)` of a `PINHOLE` or `SIMPLE_PINHOLE` camera (COLMAP's
    /// convention: the upper-left pixel's centre is at `(0.5, 0.5)`).
    pub fn pinhole(&self) -> Option<(f64, f64, f64, f64)> {
        match (self.model.as_str(), self.params.as_slice()) {
            ("PINHOLE", &[fx, fy, cx, cy]) => Some((fx, fy, cx, cy)),
            ("SIMPLE_PINHOLE", &[f, cx, cy]) => Some((f, f, cx, cy)),
            _ => None,
        }
    }
}

/// One image of `images.txt`.
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    pub id: u32,
    pub camera_id: u32,
    pub name: String,
    /// World-to-camera rotation as `(w, x, y, z)`.
    pub qvec: [f64; 4],
    /// World-to-camera translation.
    pub tvec: [f64; 3],
    /// The 3D point each keypoint observes, if any.
    pub point3d_ids: Vec<Option<u64>>,
}

impl Image {
    /// World-to-camera rotation matrix.
    pub fn rotation(&self) -> Matrix3<f64> {
        let [w, x, y, z] = self.qvec;
        UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z))
            .to_rotation_matrix()
            .into_inner()
    }

    /// World-to-camera translation.
    pub fn translation(&self) -> Vector3<f64> {
        Vector3::from(self.tvec)
    }

    /// The camera centre in world coordinates.
    pub fn center(&self) -> Vector3<f64> {
        -(self.rotation().transpose() * self.translation())
    }
}

/// One point of `points3D.txt`.
#[derive(Debug, Clone, PartialEq)]
pub struct Point3D {
    pub id: u64,
    pub xyz: Vector3<f64>,
}

/// A COLMAP sparse model; images in file order.
#[derive(Debug, Clone, Default)]
pub struct SparseModel {
    pub cameras: HashMap<u32, Camera>,
    pub images: Vec<Image>,
    pub points: HashMap<u64, Point3D>,
}

/// Reads `cameras.txt`, `images.txt` and `points3D.txt` from `dir`.
pub fn read_text_model(dir: &Path) -> Result<SparseModel, ModelError> {
    let read = |name: &str| {
        let path = dir.join(name);
        std::fs::read_to_string(&path).map_err(|source| ModelError::Io { path, source })
    };
    Ok(SparseModel {
        cameras: parse_cameras(&read("cameras.txt")?)?,
        images: parse_images(&read("images.txt")?)?,
        points: parse_points(&read("points3D.txt")?)?,
    })
}

fn data_lines(contents: &str) -> impl Iterator<Item = &str> {
    contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
}

fn field<T: std::str::FromStr>(
    tokens: &[&str],
    index: usize,
    file: &'static str,
    line: &str,
) -> Result<T, ModelError> {
    tokens
        .get(index)
        .and_then(|token| token.parse().ok())
        .ok_or_else(|| ModelError::Parse {
            file,
            line: line.to_owned(),
            message: "missing or malformed field",
        })
}

fn parse_cameras(contents: &str) -> Result<HashMap<u32, Camera>, ModelError> {
    const FILE: &str = "cameras.txt";
    let mut cameras = HashMap::new();
    for line in data_lines(contents) {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let camera = Camera {
            id: field(&tokens, 0, FILE, line)?,
            model: field(&tokens, 1, FILE, line)?,
            width: field(&tokens, 2, FILE, line)?,
            height: field(&tokens, 3, FILE, line)?,
            params: (4..tokens.len())
                .map(|i| field(&tokens, i, FILE, line))
                .collect::<Result<_, _>>()?,
        };
        cameras.insert(camera.id, camera);
    }
    Ok(cameras)
}

fn parse_images(contents: &str) -> Result<Vec<Image>, ModelError> {
    const FILE: &str = "images.txt";
    let mut images = Vec::new();
    // Each image is a header line and a (possibly empty) keypoint line, so
    // empty lines are kept here.
    let mut lines = contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with('#'));
    while let Some(header) = lines.next() {
        if header.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = header.split_whitespace().collect();
        if tokens.len() < 10 {
            return Err(ModelError::Parse {
                file: FILE,
                line: header.to_owned(),
                message: "expected IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME",
            });
        }
        let points_line = lines.next().unwrap_or("");
        let point_tokens: Vec<&str> = points_line.split_whitespace().collect();
        if !point_tokens.len().is_multiple_of(3) {
            return Err(ModelError::Parse {
                file: FILE,
                line: points_line.to_owned(),
                message: "expected (X, Y, POINT3D_ID) triples",
            });
        }
        let point3d_ids = point_tokens
            .chunks_exact(3)
            .map(|chunk| {
                let id: i64 = chunk[2].parse().map_err(|_| ModelError::Parse {
                    file: FILE,
                    line: points_line.to_owned(),
                    message: "malformed POINT3D_ID",
                })?;
                Ok(u64::try_from(id).ok())
            })
            .collect::<Result<_, ModelError>>()?;
        images.push(Image {
            id: field(&tokens, 0, FILE, header)?,
            qvec: [
                field(&tokens, 1, FILE, header)?,
                field(&tokens, 2, FILE, header)?,
                field(&tokens, 3, FILE, header)?,
                field(&tokens, 4, FILE, header)?,
            ],
            tvec: [
                field(&tokens, 5, FILE, header)?,
                field(&tokens, 6, FILE, header)?,
                field(&tokens, 7, FILE, header)?,
            ],
            camera_id: field(&tokens, 8, FILE, header)?,
            // Names may contain spaces: everything after the camera id.
            name: tokens[9..].join(" "),
            point3d_ids,
        });
    }
    Ok(images)
}

fn parse_points(contents: &str) -> Result<HashMap<u64, Point3D>, ModelError> {
    const FILE: &str = "points3D.txt";
    let mut points = HashMap::new();
    for line in data_lines(contents) {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let point = Point3D {
            id: field(&tokens, 0, FILE, line)?,
            xyz: Vector3::new(
                field(&tokens, 1, FILE, line)?,
                field(&tokens, 2, FILE, line)?,
                field(&tokens, 3, FILE, line)?,
            ),
        };
        points.insert(point.id, point);
    }
    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_small_model() {
        let cameras = parse_cameras(
            "# comment\n1 PINHOLE 640 480 500 501 320 240\n2 SIMPLE_PINHOLE 100 80 90 50 40\n",
        )
        .unwrap();
        assert_eq!(cameras[&1].pinhole(), Some((500.0, 501.0, 320.0, 240.0)));
        assert_eq!(cameras[&2].pinhole(), Some((90.0, 90.0, 50.0, 40.0)));

        let images = parse_images(
            "# header\n3 1 0 0 0 0.5 0 0 1 dir/a b.jpg\n10 20 7 30 40 -1\n4 1 0 0 0 0 0 0 2 c.png\n\n",
        )
        .unwrap();
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].name, "dir/a b.jpg");
        assert_eq!(images[0].point3d_ids, vec![Some(7), None]);
        assert_eq!(images[0].center(), Vector3::new(-0.5, 0.0, 0.0));
        assert!(images[1].point3d_ids.is_empty());

        let points = parse_points("7 1 2 3 255 0 0 0.5 3 0\n").unwrap();
        assert_eq!(points[&7].xyz, Vector3::new(1.0, 2.0, 3.0));
    }
}
