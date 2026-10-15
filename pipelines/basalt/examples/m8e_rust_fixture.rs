//! Run the portable M8e fixture emitted by `build_m8e_rust_fixture.py`.
use nalgebra::{Matrix2, Matrix3, Matrix6, Point2, Vector2, Vector6};
use serde_json::Value;
use std::{collections::BTreeMap, env, fs};
use visloc_basalt::vio::margdata::{
    FramePoseData, MargData, MatrixData, MARGDATA_SCHEMA_VERSION_V3,
};
use visloc_basalt::{
    mapper::{
        self, GlobalBaConfig, MapperFactors, MapperLandmark, MapperObservation, RelativePoseFactor,
        RollPitchFactor, TimeCamId,
    },
    vio::StereographicDirection,
    BasaltCalibration,
};

fn u(v: &Value) -> u64 {
    v.as_u64().unwrap()
}
fn f(v: &Value) -> f64 {
    v.as_f64()
        .or_else(|| v.as_i64().map(|x| x as f64))
        .unwrap_or_else(|| panic!("not numeric: {v}"))
}
fn a<const N: usize>(v: &Value) -> [f64; N] {
    std::array::from_fn(|i| f(&v[i]))
}
fn flat(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .flat_map(|x| if x.is_array() { flat(x) } else { vec![f(x)] })
        .collect()
}
fn id(v: &Value) -> TimeCamId {
    TimeCamId::new(u(&v["frame_id"]), v["cam_id"].as_u64().unwrap() as u16)
}

fn main() {
    let root = env::args()
        .nth(1)
        .unwrap_or_else(|| "target/m8e_rust_fixture.json".into());
    let value: Value = serde_json::from_str(&fs::read_to_string(&root).unwrap()).unwrap();
    let mut frame_poses = Vec::new();
    for p in value["poses"].as_array().unwrap() {
        let t = a::<3>(&p["translation"]);
        let q = a::<4>(&p["quaternion_xyzw"]);
        frame_poses.push(FramePoseData {
            frame_id: u(&p["frame_id"]),
            timestamp_ns: u(&p["frame_id"]) as i64,
            pose: [t[0], t[1], t[2], q[3], q[0], q[1], q[2]],
            is_keyframe: true,
        });
    }
    let data = MargData {
        // This synthetic fixture carries only compatibility poses; it has no
        // upstream FEJ sidecars.  Keep it an explicit schema-3 record instead
        // of fabricating current/linearized values for schema 4.
        schema_version: MARGDATA_SCHEMA_VERSION_V3,
        aom_sqrt_jacobian: MatrixData::new(0, 0, vec![]).unwrap(),
        aom_sqrt_rhs: vec![],
        aom_abs_h: None,
        aom_abs_b: None,
        frame_poses,
        frame_states: vec![],
        keyframes: vec![],
        kf_to_marg: vec![],
        kfs_all: vec![],
        kfs_to_marg: vec![],
        aom_order: vec![],
        marginalization: Default::default(),
        prior: None,
        row_counts: [0; 4],
        of_observations: vec![],
        of_images: vec![],
        used_imu: false,
        provenance_version: "basalt-m8e-rust-fixture".into(),
        frame_poses_fej: BTreeMap::new(),
        frame_states_fej: BTreeMap::new(),
        fej_complete: false,
        measurement_poses: Vec::new(),
    };
    let pose_map: BTreeMap<u64, [f64; 7]> = data
        .frame_poses
        .iter()
        .map(|p| (p.frame_id, p.pose))
        .collect();
    let mut landmarks = BTreeMap::new();
    for x in value["landmarks"].as_array().unwrap() {
        let mut obs = Vec::new();
        for o in x["observations"].as_array().unwrap() {
            let p = a::<2>(&o["pixel"]);
            obs.push(MapperObservation {
                image: id(&o["image"]),
                feature_id: u(&o["feature_id"]),
                pixel: Point2::new(p[0], p[1]),
            });
        }
        let d = a::<2>(&x["direction"]);
        landmarks.insert(
            u(&x["track_id"]),
            MapperLandmark {
                track_id: u(&x["track_id"]),
                host: id(&x["host"]),
                second: id(&x["second"]),
                direction: StereographicDirection {
                    xy: Point2::new(d[0], d[1]),
                },
                inverse_distance: f(&x["inverse_distance"]),
                observations: obs,
            },
        );
    }
    let mut relative_pose = Vec::new();
    for x in value["factors"]["relative_pose"].as_array().unwrap() {
        let q = a::<4>(&x["rotation"]);
        relative_pose.push(RelativePoseFactor {
            from: u(&x["from"]),
            to: u(&x["to"]),
            translation: a::<3>(&x["translation"]),
            rotation: [q[3], q[0], q[1], q[2]],
            information: flat(&x["information"]),
            weight: 1.0,
        });
    }
    let mut roll_pitch = Vec::new();
    for x in value["factors"]["roll_pitch"].as_array().unwrap() {
        let r = a::<9>(&x["measured_rotation"]);
        let m = Matrix3::from_row_slice(&r);
        roll_pitch.push(RollPitchFactor {
            frame_id: u(&x["frame_id"]),
            roll: 0.0,
            pitch: 0.0,
            information: a::<4>(&x["information"]),
            weight: 1.0,
            measured_rotation: [
                m[(0, 0)],
                m[(0, 1)],
                m[(0, 2)],
                m[(1, 0)],
                m[(1, 1)],
                m[(1, 2)],
                m[(2, 0)],
                m[(2, 1)],
                m[(2, 2)],
            ],
        });
    }
    let factors = MapperFactors {
        imu_relative_pose: Vec::new(),
        provenance_version: "m8a-m8e-pinned".into(),
        relative_pose,
        roll_pitch,
        ba_covisibility: vec![],
    };
    let rel_costs: Vec<f64> = factors
        .relative_pose
        .iter()
        .map(|x| {
            let m = [
                x.translation[0],
                x.translation[1],
                x.translation[2],
                x.rotation[0],
                x.rotation[1],
                x.rotation[2],
                x.rotation[3],
            ];
            let r = mapper::rel_pose_error(m, pose_map[&x.from], pose_map[&x.to]).0;
            (Vector6::from_row_slice(&r).transpose()
                * Matrix6::from_row_slice(&x.information)
                * Vector6::from_row_slice(&r))[0]
        })
        .collect();
    let rp_costs: Vec<f64> = factors
        .roll_pitch
        .iter()
        .map(|x| {
            let r = mapper::roll_pitch_error(
                pose_map[&x.frame_id],
                [
                    [
                        x.measured_rotation[0],
                        x.measured_rotation[1],
                        x.measured_rotation[2],
                    ],
                    [
                        x.measured_rotation[3],
                        x.measured_rotation[4],
                        x.measured_rotation[5],
                    ],
                    [
                        x.measured_rotation[6],
                        x.measured_rotation[7],
                        x.measured_rotation[8],
                    ],
                ],
            )
            .0;
            (Vector2::from_row_slice(&r).transpose()
                * Matrix2::from_row_slice(&x.information)
                * Vector2::from_row_slice(&r))[0]
        })
        .collect();
    let calib_path =
        env::var("BASALT_CALIB").unwrap_or_else(|_| "target/euroc_ds_calib.json".into());
    let calib = BasaltCalibration::from_path(calib_path).unwrap();
    let (_, s) = mapper::global_ba_with_landmarks(
        &data,
        &factors,
        &landmarks,
        &calib,
        GlobalBaConfig::default(),
    );
    let t = s.trace.first();
    let it: Vec<_>=s.trace.iter().map(|x| { let q=x.trials.first(); serde_json::json!({"iteration":x.iteration,"vision":x.vision_cost,"relative":x.relative_cost,"roll_pitch":x.roll_pitch_cost,"hdiag0":x.h_diagonal.first().copied().unwrap_or(0.0),"max_pose_increment":q.map_or(0.0,|z|z.max_pose_increment),"f_diff":q.map_or(0.0,|z|z.f_diff),"accepted":q.is_some_and(|z|z.accepted),"landmark_3_increment":x.landmark_increments.iter().find(|(id,_)|*id==3).map_or(&[0.0;3], |(_,v)|v)}) }).collect();
    println!(
        "{}",
        serde_json::json!({"initial_cost":s.initial_cost,"final_cost":s.final_cost,"iterations":s.iterations,"final_lambda":s.final_lambda,"final_state_hash":s.final_state_hash,"trace_hash":s.trace_hash,"pose_count":s.pose_count,"track_count":s.track_count,"initial_vision":t.map_or(0.0,|x|x.vision_cost),"initial_relative":t.map_or(0.0,|x|x.relative_cost),"initial_roll_pitch":t.map_or(0.0,|x|x.roll_pitch_cost),"relative_factor_costs":rel_costs,"roll_pitch_factor_costs":rp_costs,"trace":it})
    );
}
