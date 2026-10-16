//! Browser front end for the photo pipeline (`gsplat_photos`): pick a folder
//! of photos, start a run, follow its progress and log, look at the
//! reconstructed cameras and points in 3D, and download the outputs.
//!
//! ```text
//! cargo build --release -p visloc-gsplat-train --features gpu,euroc --example gsplat_photos
//! cargo run --release --example visloc_web -- [--port 8080] [--host 127.0.0.1] [--bin-dir DIR]
//! ```
//!
//! then open `http://127.0.0.1:8080`. Each run is a `gsplat_photos`
//! process (found next to this binary unless `--bin-dir` says otherwise),
//! one at a time; its whole output also goes to `<out>/visloc_web.log`.
//! Results already on disk (a `gsplat_photos` output folder) can be opened
//! for viewing without running anything.
//!
//! The server only uses the standard library. It listens on 127.0.0.1 by
//! default: whoever can reach it can start runs, list folders and download
//! the files of a run's output folder, so only bind it to another address
//! on a network you trust.

use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use serde_json::{json, Value};

const INDEX_HTML: &str = include_str!("index.html");
/// Log lines kept per run for the page (the full log is on disk).
const MAX_LINES: usize = 4000;
/// Points sent to the viewer at most (an even subsample beyond).
const MAX_VIEW_POINTS: usize = 200_000;
/// Largest request body accepted.
const MAX_BODY: usize = 64 * 1024;
/// Prefixes of the mapper's per-step diagnostics, kept out of the page log.
const NOISY: [&str; 10] = [
    "BA_PHASES",
    "BA_INTRINSICS",
    "BA_SOLVE",
    "BA_MEMORY",
    "BA_TRACE",
    "BA ",
    "BUILD_PARTS",
    "TIMING",
    "REGISTER",
    "INIT_PAIR",
];

#[derive(Clone, Copy, PartialEq)]
enum Status {
    Running,
    Done,
    Failed,
    Cancelled,
}

impl Status {
    fn name(self) -> &'static str {
        match self {
            Status::Running => "running",
            Status::Done => "done",
            Status::Failed => "failed",
            Status::Cancelled => "cancelled",
        }
    }
}

/// What the page shows of a run's progress, parsed from its output.
#[derive(Default)]
struct Progress {
    stage: String,
    images: Option<usize>,
    registered: usize,
    step: usize,
    /// 0..=1, monotone.
    fraction: f64,
}

struct Job {
    id: u64,
    title: String,
    out: PathBuf,
    command: Vec<String>,
    steps: usize,
    mesh: bool,
    started: f64,
    elapsed: Option<f64>,
    clock: Instant,
    status: Status,
    exit: Option<String>,
    /// The last [`MAX_LINES`] page log lines; `first_line` is the index of
    /// the oldest one kept.
    lines: VecDeque<String>,
    first_line: usize,
    progress: Progress,
    child: Option<Arc<Mutex<Child>>>,
}

impl Job {
    fn summary(&self) -> Value {
        json!({
            "id": self.id,
            "title": self.title,
            "out": self.out,
            "command": self.command,
            "status": self.status.name(),
            "exit": self.exit,
            "started": self.started,
            "elapsed": self.elapsed.unwrap_or_else(|| self.clock.elapsed().as_secs_f64()),
            "stage": self.progress.stage,
            "images": self.progress.images,
            "registered": self.progress.registered,
            "step": self.progress.step,
            "steps": self.steps,
            "fraction": self.progress.fraction,
            "has_model": self.out.join("sparse/0/images.txt").is_file(),
        })
    }

    /// Takes one output line: updates the progress and keeps it for the
    /// page unless it is mapper noise.
    fn push_line(&mut self, line: &str) {
        self.parse_progress(line);
        if NOISY.iter().any(|p| line.starts_with(p)) {
            return;
        }
        self.lines.push_back(line.to_string());
        if self.lines.len() > MAX_LINES {
            self.lines.pop_front();
            self.first_line += 1;
        }
    }

    fn parse_progress(&mut self, line: &str) {
        let p = &mut self.progress;
        let train = self.steps > 0;
        // Stage weights: load, SIFT, matching, SfM, training, mesh.
        let (sfm_end, train_end) = match (train, self.mesh) {
            (false, _) => (1.0, 1.0),
            (true, true) => (0.45, 0.95),
            (true, false) => (0.45, 1.0),
        };
        fn set(p: &mut Progress, stage: &str, fraction: f64) {
            p.stage = stage.to_string();
            p.fraction = p.fraction.max(fraction.min(1.0));
        }
        let first = line.split_whitespace().next().unwrap_or("");
        if line.contains(" images at ") && first.parse::<usize>().is_ok() {
            p.images = first.parse().ok();
            set(p, "Extracting features", 0.03);
        } else if line.starts_with("sift:") {
            set(p, "Matching photos", 0.12);
        } else if line.contains(" verified of ") {
            set(p, "Reconstructing (SfM)", 0.3 * sfm_end);
        } else if let Some(rest) = line.strip_prefix("REGISTER ") {
            let reg = rest
                .split_whitespace()
                .find_map(|kv| kv.strip_prefix("num_reg_frames="))
                .and_then(|v| v.parse::<usize>().ok());
            if let (Some(reg), Some(n)) = (reg, p.images) {
                p.registered = p.registered.max(reg);
                let f = 0.3 + 0.65 * (reg as f64 / n.max(1) as f64).min(1.0);
                set(p, "Reconstructing (SfM)", f * sfm_end);
            }
        } else if line.starts_with("sfm:") {
            if let Some(reg) = line
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.split('/').next())
                .and_then(|v| v.parse::<usize>().ok())
            {
                p.registered = reg;
            }
            set(
                p,
                if train { "Training 3DGS" } else { "Finishing" },
                sfm_end,
            );
        } else if first == "step" && train {
            if let Some(step) = line
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse::<usize>().ok())
            {
                p.step = step;
                let f = sfm_end + (train_end - sfm_end) * step as f64 / self.steps as f64;
                set(p, "Training 3DGS", f);
            }
        } else if line.starts_with("trained ") {
            set(
                p,
                if self.mesh {
                    "Extracting mesh"
                } else {
                    "Writing"
                },
                train_end,
            );
        } else if line.starts_with("total ") {
            set(p, "Done", 1.0);
        }
    }
}

struct State {
    jobs: Vec<Job>,
    next_id: u64,
    bin_dir: PathBuf,
}

type Shared = Arc<Mutex<State>>;

fn main() {
    let mut port = 8080u16;
    let mut host = "127.0.0.1".to_string();
    let mut bin_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."));
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || {
            args.next().unwrap_or_else(|| {
                eprintln!("{a} needs a value");
                std::process::exit(2)
            })
        };
        match a.as_str() {
            "--port" => port = value().parse().expect("--port takes a number"),
            "--host" => host = value(),
            "--bin-dir" => bin_dir = PathBuf::from(value()),
            "-h" | "--help" => {
                println!("usage: visloc_web [--port 8080] [--host 127.0.0.1] [--bin-dir DIR]");
                return;
            }
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2)
            }
        }
    }
    let listener = TcpListener::bind((host.as_str(), port)).unwrap_or_else(|e| {
        eprintln!("cannot listen on {host}:{port}: {e}");
        std::process::exit(1)
    });
    let photos = bin_dir.join(exe("gsplat_photos"));
    println!("visloc web: http://{host}:{port}");
    if !photos.is_file() {
        println!(
            "note: {} not found; build it with\n  cargo build --release -p visloc-gsplat-train \
             --features gpu,euroc --example gsplat_photos\nor pass --bin-dir",
            photos.display()
        );
    }
    if host != "127.0.0.1" && host != "localhost" && host != "::1" {
        println!("warning: anyone who can reach {host}:{port} can start runs and read run outputs");
    }
    let state: Shared = Arc::new(Mutex::new(State {
        jobs: Vec::new(),
        next_id: 1,
        bin_dir,
    }));
    for stream in listener.incoming().flatten() {
        let state = Arc::clone(&state);
        std::thread::spawn(move || {
            if let Err(e) = handle(stream, &state) {
                eprintln!("request error: {e}");
            }
        });
    }
}

fn exe(name: &str) -> String {
    format!("{name}{}", std::env::consts::EXE_SUFFIX)
}

// ---- HTTP -------------------------------------------------------------

struct Request {
    method: String,
    path: String,
    query: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Request {
    fn param(&self, key: &str) -> Option<&str> {
        self.query
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

struct Response {
    status: u16,
    content_type: &'static str,
    body: Vec<u8>,
    extra_headers: Vec<String>,
}

impl Response {
    fn json(value: Value) -> Self {
        Self::bytes(200, "application/json", value.to_string().into_bytes())
    }

    fn error(status: u16, message: &str) -> Self {
        Self::bytes(
            status,
            "application/json",
            json!({ "error": message }).to_string().into_bytes(),
        )
    }

    fn bytes(status: u16, content_type: &'static str, body: Vec<u8>) -> Self {
        Self {
            status,
            content_type,
            body,
            extra_headers: Vec::new(),
        }
    }
}

fn handle(stream: TcpStream, state: &Shared) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(());
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("/").to_string();
    let mut content_length = 0usize;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 || header == "\r\n" || header == "\n" {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                content_length = value.trim().parse().unwrap_or(0);
            }
        }
    }
    let response = if content_length > MAX_BODY {
        Response::error(413, "request body too large")
    } else {
        let mut body = vec![0u8; content_length];
        reader.read_exact(&mut body)?;
        let (path, query) = match target.split_once('?') {
            Some((p, q)) => (p.to_string(), parse_query(q)),
            None => (target.clone(), Vec::new()),
        };
        route(
            &Request {
                method,
                path,
                query,
                body,
            },
            state,
        )
    };
    write_response(stream, response)
}

fn write_response(mut stream: TcpStream, response: Response) -> std::io::Result<()> {
    let reason = match response.status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        413 => "Payload Too Large",
        _ => "Error",
    };
    let mut head = format!(
        "HTTP/1.1 {} {reason}\r\nContent-Type: {}\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n",
        response.status,
        response.content_type,
        response.body.len()
    );
    for h in &response.extra_headers {
        head.push_str(h);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    stream.write_all(&response.body)?;
    stream.flush()
}

fn parse_query(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter(|kv| !kv.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            (percent_decode(k), percent_decode(v))
        })
        .collect()
}

fn percent_decode(s: &str) -> String {
    let hex = |b: u8| (b as char).to_digit(16).map(|d| d as u8);
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' => match (
                bytes.get(i + 1).and_then(|&b| hex(b)),
                bytes.get(i + 2).and_then(|&b| hex(b)),
            ) {
                (Some(hi), Some(lo)) => {
                    out.push(hi * 16 + lo);
                    i += 2;
                }
                _ => out.push(b'%'),
            },
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn route(req: &Request, state: &Shared) -> Response {
    let segments: Vec<&str> = req.path.trim_matches('/').split('/').collect();
    match (req.method.as_str(), segments.as_slice()) {
        ("GET", [""]) | ("GET", ["index.html"]) => Response::bytes(
            200,
            "text/html; charset=utf-8",
            INDEX_HTML.as_bytes().to_vec(),
        ),
        ("GET", ["api", "state"]) => api_state(state),
        ("GET", ["api", "browse"]) => api_browse(req.param("path")),
        ("POST", ["api", "jobs"]) => api_start(req, state),
        ("POST", ["api", "open"]) => api_open(req, state),
        ("GET", ["api", "jobs", id]) => with_job(state, id, |job| {
            let from = req
                .param("from")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0);
            // Lines from `from` on (those already dropped are skipped).
            let skip = from.saturating_sub(job.first_line);
            let lines: Vec<&String> = job.lines.iter().skip(skip).collect();
            let mut v = job.summary();
            v["next"] = json!(job.first_line + job.lines.len());
            v["lines"] = json!(lines);
            Response::json(v)
        }),
        ("POST", ["api", "jobs", id, "cancel"]) => api_cancel(state, id),
        ("GET", ["api", "jobs", id, "model"]) => match with_job_out(state, id) {
            Ok(out) => model_json(&out),
            Err(r) => r,
        },
        ("GET", ["api", "jobs", id, "files"]) => match with_job_out(state, id) {
            Ok(out) => Response::json(json!({ "files": list_files(&out) })),
            Err(r) => r,
        },
        ("GET", ["api", "jobs", id, "download"]) => match with_job_out(state, id) {
            Ok(out) => download(&out, req.param("path").unwrap_or("")),
            Err(r) => r,
        },
        (_, ["api", ..]) => Response::error(404, "no such endpoint"),
        _ => Response::error(404, "not found"),
    }
}

fn with_job(state: &Shared, id: &str, f: impl FnOnce(&Job) -> Response) -> Response {
    let Ok(id) = id.parse::<u64>() else {
        return Response::error(400, "bad run id");
    };
    let state = state.lock().expect("state lock");
    match state.jobs.iter().find(|j| j.id == id) {
        Some(job) => f(job),
        None => Response::error(404, "no such run"),
    }
}

fn with_job_out(state: &Shared, id: &str) -> Result<PathBuf, Response> {
    let id = id
        .parse::<u64>()
        .map_err(|_| Response::error(400, "bad run id"))?;
    let state = state.lock().expect("state lock");
    state
        .jobs
        .iter()
        .find(|j| j.id == id)
        .map(|j| j.out.clone())
        .ok_or_else(|| Response::error(404, "no such run"))
}

// ---- API --------------------------------------------------------------

fn api_state(state: &Shared) -> Response {
    let state = state.lock().expect("state lock");
    let photos = state.bin_dir.join(exe("gsplat_photos"));
    Response::json(json!({
        "busy": state.jobs.iter().any(|j| j.status == Status::Running),
        "photos_binary": photos,
        "photos_binary_found": photos.is_file(),
        "cwd": std::env::current_dir().ok(),
        "jobs": state.jobs.iter().rev().map(Job::summary).collect::<Vec<_>>(),
    }))
}

fn is_image(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("jpg" | "jpeg" | "png")
    )
}

fn api_browse(path: Option<&str>) -> Response {
    let dir = match path.filter(|p| !p.is_empty()) {
        Some(p) => PathBuf::from(p),
        None => std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")),
    };
    let dir = match dir.canonicalize() {
        Ok(d) if d.is_dir() => d,
        _ => return Response::error(404, "not a folder"),
    };
    let mut dirs = Vec::new();
    let mut images = 0usize;
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let hidden = entry.file_name().to_string_lossy().starts_with('.');
            if path.is_dir() && !hidden {
                dirs.push(entry.file_name().to_string_lossy().into_owned());
            } else if path.is_file() && is_image(&path) {
                images += 1;
            }
        }
    }
    dirs.sort();
    Response::json(json!({
        "path": dir,
        "parent": dir.parent(),
        "dirs": dirs,
        "images": images,
        "is_result": dir.join("sparse/0/images.txt").is_file(),
    }))
}

fn body_json(req: &Request) -> Result<Value, Response> {
    serde_json::from_slice(&req.body).map_err(|_| Response::error(400, "expected a JSON body"))
}

fn api_start(req: &Request, state: &Shared) -> Response {
    let body = match body_json(req) {
        Ok(b) => b,
        Err(r) => return r,
    };
    let text = |key: &str| body[key].as_str().map(str::trim).filter(|s| !s.is_empty());
    let Some(images) = text("images").map(PathBuf::from) else {
        return Response::error(400, "choose a folder of photos");
    };
    if !images.is_dir() {
        return Response::error(400, "the photo folder does not exist");
    }
    let out = match text("out") {
        Some(o) => PathBuf::from(o),
        None => images.with_file_name(format!(
            "{}_visloc",
            images
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "photos".into())
        )),
    };
    let mode = text("mode").unwrap_or("splat");
    let steps = if mode == "sfm" {
        0
    } else {
        body["steps"].as_u64().unwrap_or(30_000) as usize
    };
    let mesh = mode == "mesh";
    let mut args: Vec<String> = vec![
        "--images".into(),
        images.to_string_lossy().into_owned(),
        "--out".into(),
        out.to_string_lossy().into_owned(),
        "--steps".into(),
        steps.to_string(),
    ];
    if !mesh {
        args.push("--no-mesh".into());
    }
    if let Some(v) = body["max_size"].as_u64() {
        args.extend(["--max-size".into(), v.to_string()]);
    }
    if let Some(v) = body["focal"].as_f64().filter(|f| *f > 0.0) {
        args.extend(["--focal".into(), v.to_string()]);
    }
    if body["refine_intrinsics"].as_bool() == Some(false) {
        args.push("--no-refine-intrinsics".into());
    }
    if body["appearance"].as_bool() == Some(true) {
        args.push("--appearance".into());
    }
    if let Some(extra) = text("extra") {
        // Plain whitespace-separated `gsplat_photos` flags; never a shell.
        args.extend(extra.split_whitespace().map(String::from));
    }

    let mut st = state.lock().expect("state lock");
    if st.jobs.iter().any(|j| j.status == Status::Running) {
        return Response::error(409, "a run is already in progress");
    }
    let binary = st.bin_dir.join(exe("gsplat_photos"));
    if !binary.is_file() {
        return Response::error(
            400,
            &format!(
                "{} not found (build it, or pass --bin-dir)",
                binary.display()
            ),
        );
    }
    if let Err(e) = fs::create_dir_all(&out) {
        return Response::error(400, &format!("cannot create {}: {e}", out.display()));
    }
    let log_file = match fs::File::create(out.join("visloc_web.log")) {
        Ok(f) => Arc::new(Mutex::new(f)),
        Err(e) => return Response::error(400, &format!("cannot write to {}: {e}", out.display())),
    };
    let mut child = match Command::new(&binary)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return Response::error(400, &format!("cannot start {}: {e}", binary.display())),
    };
    let id = st.next_id;
    st.next_id += 1;
    let streams = (child.stdout.take(), child.stderr.take());
    let child = Arc::new(Mutex::new(child));
    let mut command = vec![binary.to_string_lossy().into_owned()];
    command.extend(args);
    st.jobs.push(Job {
        id,
        title: images
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| images.to_string_lossy().into_owned()),
        out,
        command,
        steps,
        mesh,
        started: unix_now(),
        elapsed: None,
        clock: Instant::now(),
        status: Status::Running,
        exit: None,
        lines: VecDeque::new(),
        first_line: 0,
        progress: Progress {
            stage: "Loading photos".into(),
            ..Progress::default()
        },
        child: Some(Arc::clone(&child)),
    });
    drop(st);

    let readers: Vec<_> = [
        streams.0.map(|s| Box::new(s) as Box<dyn Read + Send>),
        streams.1.map(|s| Box::new(s) as Box<dyn Read + Send>),
    ]
    .into_iter()
    .flatten()
    .map(|stream| {
        let (state, log_file) = (Arc::clone(state), Arc::clone(&log_file));
        std::thread::spawn(move || {
            for line in BufReader::new(stream).lines() {
                let Ok(line) = line else { break };
                if let Ok(mut f) = log_file.lock() {
                    let _ = writeln!(f, "{line}");
                }
                let mut st = state.lock().expect("state lock");
                if let Some(job) = st.jobs.iter_mut().find(|j| j.id == id) {
                    job.push_line(&line);
                }
            }
        })
    })
    .collect();
    let state_w = Arc::clone(state);
    std::thread::spawn(move || {
        let status = loop {
            if let Ok(Some(status)) = child.lock().expect("child lock").try_wait() {
                break status;
            }
            std::thread::sleep(Duration::from_millis(200));
        };
        for r in readers {
            let _ = r.join();
        }
        let mut st = state_w.lock().expect("state lock");
        if let Some(job) = st.jobs.iter_mut().find(|j| j.id == id) {
            job.elapsed = Some(job.clock.elapsed().as_secs_f64());
            job.child = None;
            if job.status == Status::Running {
                job.status = if status.success() {
                    Status::Done
                } else {
                    Status::Failed
                };
            }
            if job.status == Status::Done {
                job.progress.stage = "Done".into();
                job.progress.fraction = 1.0;
            }
            job.exit = Some(status.to_string());
        }
    });
    Response::json(json!({ "id": id }))
}

fn api_open(req: &Request, state: &Shared) -> Response {
    let body = match body_json(req) {
        Ok(b) => b,
        Err(r) => return r,
    };
    let Some(out) = body["out"].as_str().map(PathBuf::from) else {
        return Response::error(400, "choose a result folder");
    };
    if !out.join("sparse/0/images.txt").is_file() {
        return Response::error(400, "no sparse/0 model in that folder");
    }
    let mut st = state.lock().expect("state lock");
    let id = st.next_id;
    st.next_id += 1;
    let registered = count_images(&out.join("sparse/0/images.txt"));
    st.jobs.push(Job {
        id,
        title: format!(
            "{} (opened)",
            out.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        ),
        out,
        command: Vec::new(),
        steps: 0,
        mesh: false,
        started: unix_now(),
        elapsed: Some(0.0),
        clock: Instant::now(),
        status: Status::Done,
        exit: None,
        lines: VecDeque::new(),
        first_line: 0,
        progress: Progress {
            stage: "Opened".into(),
            registered,
            fraction: 1.0,
            ..Progress::default()
        },
        child: None,
    });
    Response::json(json!({ "id": id }))
}

fn api_cancel(state: &Shared, id: &str) -> Response {
    let Ok(id) = id.parse::<u64>() else {
        return Response::error(400, "bad run id");
    };
    let mut st = state.lock().expect("state lock");
    let Some(job) = st.jobs.iter_mut().find(|j| j.id == id) else {
        return Response::error(404, "no such run");
    };
    match &job.child {
        Some(child) if job.status == Status::Running => {
            let _ = child.lock().expect("child lock").kill();
            job.status = Status::Cancelled;
            job.progress.stage = "Cancelled".into();
            Response::json(json!({ "ok": true }))
        }
        _ => Response::error(409, "the run is not running"),
    }
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ---- Results ----------------------------------------------------------

/// The image lines of a COLMAP `images.txt` (`IMAGE_ID QW QX QY QZ TX TY
/// TZ CAMERA_ID NAME`): each is followed by its 2D points line, which is
/// empty for an image without any, so lines are taken in pairs.
fn image_lines(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    while let Some(line) = lines.next() {
        if line.trim().is_empty() {
            continue;
        }
        out.push(line);
        let _points2d = lines.next();
    }
    out
}

fn count_images(images_txt: &Path) -> usize {
    fs::read_to_string(images_txt)
        .map(|s| image_lines(&s).len())
        .unwrap_or(0)
}

/// Cameras (centre, viewing direction, up) and a subsample of the points
/// of `<out>/sparse/0`, as flat arrays for the viewer.
fn model_json(out: &Path) -> Response {
    let dir = out.join("sparse/0");
    let Ok(images) = fs::read_to_string(dir.join("images.txt")) else {
        return Response::error(404, "no model yet");
    };
    let mut cameras = Vec::new();
    let mut names = Vec::new();
    for line in image_lines(&images) {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 10 {
            continue;
        }
        let v: Vec<f64> = f[1..8].iter().filter_map(|x| x.parse().ok()).collect();
        if v.len() != 7 {
            continue;
        }
        let q = UnitQuaternion::from_quaternion(Quaternion::new(v[0], v[1], v[2], v[3]));
        let t = Vector3::new(v[4], v[5], v[6]);
        let world_from_cam = q.inverse();
        let centre = -(world_from_cam * t);
        let forward = world_from_cam * Vector3::z();
        let up = world_from_cam * -Vector3::y();
        cameras.extend(
            [centre, forward, up]
                .iter()
                .flat_map(|v| [v.x as f32, v.y as f32, v.z as f32]),
        );
        names.push(f[9..].join(" "));
    }
    let mut xyz = Vec::new();
    let mut rgb = Vec::new();
    if let Ok(points) = fs::read_to_string(dir.join("points3D.txt")) {
        let rows: Vec<&str> = points
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .collect();
        let stride = rows.len().div_ceil(MAX_VIEW_POINTS).max(1);
        for row in rows.iter().step_by(stride) {
            let f: Vec<&str> = row.split_whitespace().take(7).collect();
            if f.len() < 7 {
                continue;
            }
            let p: Vec<f32> = f[1..4].iter().filter_map(|x| x.parse().ok()).collect();
            let c: Vec<u8> = f[4..7].iter().filter_map(|x| x.parse().ok()).collect();
            if p.len() == 3 && c.len() == 3 {
                xyz.extend(p);
                rgb.extend(c);
            }
        }
    }
    Response::json(json!({
        "cameras": cameras,
        "names": names,
        "points": xyz,
        "colors": rgb,
    }))
}

/// Files a run produced that are worth downloading.
fn list_files(out: &Path) -> Vec<Value> {
    let mut files = Vec::new();
    for rel in [
        "scene.ply",
        "mesh.ply",
        "visloc_web.log",
        "image_names.txt",
        "sparse/0/cameras.txt",
        "sparse/0/images.txt",
        "sparse/0/points3D.txt",
    ] {
        if let Ok(meta) = fs::metadata(out.join(rel)) {
            if meta.is_file() {
                files.push(json!({ "path": rel, "bytes": meta.len() }));
            }
        }
    }
    files
}

fn download(out: &Path, rel: &str) -> Response {
    // Only files inside the run's output folder.
    let (Ok(root), Ok(path)) = (out.canonicalize(), out.join(rel).canonicalize()) else {
        return Response::error(404, "no such file");
    };
    if !path.starts_with(&root) || !path.is_file() {
        return Response::error(404, "no such file");
    }
    match fs::read(&path) {
        Ok(bytes) => {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().replace('"', ""))
                .unwrap_or_else(|| "download".into());
            let mut r = Response::bytes(200, "application/octet-stream", bytes);
            r.extra_headers.push(format!(
                "Content-Disposition: attachment; filename=\"{name}\""
            ));
            r
        }
        Err(_) => Response::error(404, "no such file"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_queries() {
        assert_eq!(
            parse_query("path=%2Fhome%2Fa+b&x=1&bad=%zz"),
            vec![
                ("path".to_string(), "/home/a b".to_string()),
                ("x".to_string(), "1".to_string()),
                ("bad".to_string(), "%zz".to_string()),
            ]
        );
        assert_eq!(percent_decode("%e3%81%82"), "あ");
        assert_eq!(percent_decode("50%"), "50%");
    }

    #[test]
    fn progress_follows_the_stages() {
        let mut job = Job {
            id: 1,
            title: String::new(),
            out: PathBuf::new(),
            command: Vec::new(),
            steps: 2000,
            mesh: false,
            started: 0.0,
            elapsed: None,
            clock: Instant::now(),
            status: Status::Running,
            exit: None,
            lines: VecDeque::new(),
            first_line: 0,
            progress: Progress::default(),
            child: None,
        };
        let mut last = 0.0;
        for line in [
            "10 images at 1600x1067 (0 resized from within 1%; others dropped: 0; decoded in 1.0 s)",
            "sift: mean 7003 keypoints (17.2 s)",
            "40 verified of 45 candidate pairs (9.0 s)",
            "REGISTER path=A frame=4 image=4 inliers=10 num_reg_frames=5 num_points3d=1",
            "BA_PHASES trials=1",
            "sfm: 10/10 registered, 900 points, 0.500 px, focal 880.0 px (20.0 s)",
            "step   1000  l1 0.0500  gaussians 1000  (10s)",
            "trained 2000 steps in 20.0s; held-out psnr 25.00 over 1 views",
            "total 50.0s",
        ] {
            job.push_line(line);
            assert!(job.progress.fraction >= last, "{line}");
            last = job.progress.fraction;
        }
        assert_eq!(job.progress.images, Some(10));
        assert_eq!(job.progress.registered, 10);
        assert_eq!(job.progress.step, 1000);
        assert_eq!(job.progress.fraction, 1.0);
        // The mapper's diagnostics stay out of the page log.
        assert!(job.lines.iter().all(|l| !l.starts_with("BA_PHASES")));
        assert!(job.lines.iter().all(|l| !l.starts_with("REGISTER")));
        assert_eq!(job.lines.len(), 7);
    }

    #[test]
    fn reads_images_with_empty_point_lines() {
        let text =
            "# header\n1 1 0 0 0 0 0 0 1 a.jpg\n\n2 1 0 0 0 1 0 0 1 b.jpg\n10 20 5 30 40 -1\n";
        let lines = image_lines(text);
        assert_eq!(lines.len(), 2);
        assert!(lines[1].ends_with("b.jpg"));
    }

    #[test]
    fn downloads_stay_inside_the_output_folder() {
        let dir = std::env::temp_dir().join(format!("visloc_web_test_{}", std::process::id()));
        fs::create_dir_all(dir.join("sparse/0")).unwrap();
        fs::write(dir.join("scene.ply"), b"ply").unwrap();
        assert_eq!(download(&dir, "scene.ply").status, 200);
        assert_eq!(download(&dir, "../../etc/passwd").status, 404);
        assert_eq!(download(&dir, "/etc/passwd").status, 404);
        assert_eq!(download(&dir, "missing.ply").status, 404);
        fs::remove_dir_all(&dir).unwrap();
    }
}
