//! PatchMatch on the GPU (wgpu): [`crate::patch_match`]'s per-pixel update
//! as compute shaders (`shaders/patch_match.wgsl`), one invocation per
//! pixel, in f32. It follows the CPU path step for step (sampling regions,
//! view selection, refinement and their random numbers), so the two differ
//! only by floating-point rounding.

use bytemuck::{Pod, Zeroable};
use visloc_gsplat_render::{GpuContext, GpuError};

use crate::patch_match::{
    finish_depth_map, DepthMap, PatchMatchOptions, ViewSelection, MAX_SOURCES,
};
use crate::workspace::Workspace;

/// Uniform parameters of one dispatch (`Params` in the shader).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    num_sources: u32,
    color: u32,
    iteration: u32,
    seed: u32,
    view_id: u32,
    geometric: u32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    depth_min: f32,
    depth_max: f32,
    tau: f32,
    tau1: f32,
    beta: f32,
    n1: u32,
    n2: u32,
    init_top_k: u32,
    geom_lambda: f32,
    geom_delta: f32,
    depth_perturbation: f32,
    normal_perturbation: f32,
    window_radius: i32,
    window_step: i32,
    sigma_spatial: f32,
    sigma_color: f32,
    /// 0: [`ViewSelection::Confidence`], 1: [`ViewSelection::Sampled`].
    view_selection: u32,
    pad: [u32; 3],
}

/// A source view as the shader reads it (`Source`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuSource {
    r0: [f32; 4],
    r1: [f32; 4],
    r2: [f32; 4],
    k: [f32; 4],
    dims: [u32; 4],
}

/// Uniform slots are this far apart (wgpu's minimum offset alignment).
const SLOT: u64 = 256;

/// The GPU PatchMatch: a device and the two compute pipelines.
pub struct GpuPatchMatch {
    ctx: GpuContext,
    init: wgpu::ComputePipeline,
    propagate: wgpu::ComputePipeline,
}

impl GpuPatchMatch {
    /// Creates a device (`GpuContext::new`) and compiles the shaders.
    pub fn new() -> Result<Self, GpuError> {
        Ok(Self::with_context(GpuContext::new()?))
    }

    /// Uses an existing device.
    pub fn with_context(ctx: GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mvs-patch-match"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/patch_match.wgsl").into()),
            });
        let pipeline = |entry: &str| {
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
        };
        let init = pipeline("init");
        let propagate = pipeline("propagate");
        Self {
            ctx,
            init,
            propagate,
        }
    }

    /// The adapter's name and backend, for logs.
    pub fn adapter(&self) -> String {
        let info = &self.ctx.adapter_info;
        format!("{} ({:?})", info.name, info.backend)
    }

    /// [`crate::patch_match::estimate`] on the GPU.
    pub fn estimate(
        &self,
        ws: &Workspace,
        view: usize,
        options: &PatchMatchOptions,
        prior: Option<&[DepthMap]>,
    ) -> DepthMap {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let reference = &ws.views[view];
        let (w, h) = (reference.width, reference.height);
        let source_ids: Vec<usize> = reference
            .sources
            .iter()
            .take(MAX_SOURCES)
            .copied()
            .collect();

        // Images (reference first) and each source's earlier depth map.
        let mut images: Vec<f32> = reference.gray.clone();
        let mut priors: Vec<f32> = Vec::new();
        let mut sources = Vec::with_capacity(source_ids.len());
        for &s in &source_ids {
            let source = &ws.views[s];
            let r = source.rotation * reference.rotation.transpose();
            let t = source.translation - r * reference.translation;
            let prior_offset = priors.len() as u32;
            if let Some(maps) = prior {
                priors.extend_from_slice(&maps[s].depth);
            }
            sources.push(GpuSource {
                r0: [
                    r[(0, 0)] as f32,
                    r[(0, 1)] as f32,
                    r[(0, 2)] as f32,
                    t.x as f32,
                ],
                r1: [
                    r[(1, 0)] as f32,
                    r[(1, 1)] as f32,
                    r[(1, 2)] as f32,
                    t.y as f32,
                ],
                r2: [
                    r[(2, 0)] as f32,
                    r[(2, 1)] as f32,
                    r[(2, 2)] as f32,
                    t.z as f32,
                ],
                k: [
                    source.fx as f32,
                    source.fy as f32,
                    source.cx as f32,
                    source.cy as f32,
                ],
                dims: [
                    source.width as u32,
                    source.height as u32,
                    images.len() as u32,
                    prior_offset,
                ],
            });
            images.extend_from_slice(&source.gray);
        }
        // The starting planes as (normal, depth), depth 0 for a random one.
        let init_planes: Vec<[f32; 4]> = match prior {
            Some(maps) => {
                let map = &maps[view];
                assert_eq!((map.width, map.height), (w, h), "prior depth map size");
                map.depth
                    .iter()
                    .zip(&map.normal)
                    .map(|(&d, n)| [n[0], n[1], n[2], d.max(0.0)])
                    .collect()
            }
            None => vec![[0.0; 4]; w * h],
        };

        let base = Params {
            width: w as u32,
            height: h as u32,
            num_sources: sources.len() as u32,
            color: 0,
            iteration: 0,
            seed: options.seed,
            view_id: view as u32,
            geometric: u32::from(prior.is_some()),
            fx: reference.fx as f32,
            fy: reference.fy as f32,
            cx: reference.cx as f32,
            cy: reference.cy as f32,
            depth_min: reference.depth_range.0,
            depth_max: reference.depth_range.1,
            tau: options.tau0,
            tau1: options.tau1,
            beta: options.beta,
            n1: options.n1 as u32,
            n2: options.n2 as u32,
            init_top_k: options.init_top_k as u32,
            geom_lambda: options.geom_lambda,
            geom_delta: options.geom_delta,
            depth_perturbation: options.depth_perturbation,
            normal_perturbation: options.normal_perturbation,
            window_radius: options.window_radius,
            window_step: options.window_step,
            sigma_spatial: options.sigma_spatial,
            sigma_color: options.sigma_color,
            view_selection: u32::from(options.view_selection == ViewSelection::Sampled),
            pad: [0; 3],
        };
        // Slot 0 for `init`, then one per (iteration, colour).
        let slots = 1 + 2 * options.iterations;
        let mut uniform_bytes = vec![0u8; slots * SLOT as usize];
        let mut put = |slot: usize, params: Params| {
            let bytes = bytemuck::bytes_of(&params);
            uniform_bytes[slot * SLOT as usize..slot * SLOT as usize + bytes.len()]
                .copy_from_slice(bytes);
        };
        put(0, base);
        for iteration in 0..options.iterations {
            let t = iteration as f32;
            for color in 0..2 {
                put(
                    1 + 2 * iteration + color,
                    Params {
                        color: color as u32,
                        iteration: iteration as u32,
                        tau: options.tau0 * (-t * t / options.alpha).exp(),
                        ..base
                    },
                );
            }
        }

        let storage = |label: &str, bytes: &[u8]| -> wgpu::Buffer {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (bytes.len() as u64).max(16).div_ceil(16) * 16,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            if !bytes.is_empty() {
                queue.write_buffer(&buffer, 0, bytes);
            }
            buffer
        };
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mvs-params"),
            size: uniform_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniforms, 0, &uniform_bytes);
        let sources_buf = storage("mvs-sources", bytemuck::cast_slice(&sources));
        let images_buf = storage("mvs-images", bytemuck::cast_slice(&images));
        let priors_buf = storage("mvs-priors", bytemuck::cast_slice(&priors));
        let planes_buf = storage("mvs-planes", &vec![0u8; w * h * 16]);
        let costs_buf = storage("mvs-costs", &vec![0u8; w * h * 4]);
        let best_buf = storage("mvs-best-views", &vec![0u8; w * h * 4]);
        let init_buf = storage("mvs-init-planes", bytemuck::cast_slice(&init_planes));
        drop(images);
        drop(priors);

        let params_entry = |slot: usize| wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &uniforms,
                offset: slot as u64 * SLOT,
                size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64),
            }),
        };
        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let init_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mvs-init"),
            layout: &self.init.get_bind_group_layout(0),
            entries: &[
                params_entry(0),
                entry(1, &sources_buf),
                entry(2, &images_buf),
                entry(3, &priors_buf),
                entry(4, &planes_buf),
                entry(5, &costs_buf),
                entry(6, &best_buf),
                entry(7, &init_buf),
            ],
        });
        let propagate_groups: Vec<wgpu::BindGroup> = (1..slots)
            .map(|slot| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("mvs-propagate"),
                    layout: &self.propagate.get_bind_group_layout(0),
                    entries: &[
                        params_entry(slot),
                        entry(1, &sources_buf),
                        entry(2, &images_buf),
                        entry(3, &priors_buf),
                        entry(4, &planes_buf),
                        entry(5, &costs_buf),
                        entry(6, &best_buf),
                    ],
                })
            })
            .collect();

        // One submission per dispatch keeps each well under any watchdog.
        let run =
            |pipeline: &wgpu::ComputePipeline, group: &wgpu::BindGroup, groups: (u32, u32)| {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("mvs-dispatch"),
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("mvs-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(groups.0, groups.1, 1);
                }
                queue.submit(Some(encoder.finish()));
            };
        let (gw, gh) = ((w as u32).div_ceil(16), (h as u32).div_ceil(8));
        run(&self.init, &init_group, (gw, gh));
        let half = (w as u32).div_ceil(2).div_ceil(16);
        for group in &propagate_groups {
            run(&self.propagate, group, (half, gh));
        }

        let planes: Vec<[f32; 4]> =
            bytemuck::cast_slice(&read_bytes(device, queue, &planes_buf, w * h * 16)).to_vec();
        let costs: Vec<f32> =
            bytemuck::cast_slice(&read_bytes(device, queue, &costs_buf, w * h * 4)).to_vec();
        finish_depth_map(reference, &planes, costs, options)
    }
}

/// Copies `len` bytes of `buffer` back to the host.
fn read_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    len: usize,
) -> Vec<u8> {
    if len == 0 {
        return Vec::new();
    }
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mvs-staging"),
        size: len as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mvs-readback"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, len as u64);
    queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let _ = rx.recv();
    let data = slice.get_mapped_range().expect("map range").to_vec();
    staging.unmap();
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch_match::{estimate, plane_scene};

    /// The GPU recovers the synthetic plane like the CPU path, with and
    /// without geometric consistency, under either view selection.
    #[test]
    fn gpu_patch_match_recovers_a_plane() {
        let Some(ctx) = visloc_gsplat_render::try_context() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let gpu = GpuPatchMatch::with_context(ctx);
        let ws = plane_scene();
        for view_selection in [ViewSelection::Confidence, ViewSelection::Sampled] {
            let options = PatchMatchOptions {
                iterations: 4,
                view_selection,
                ..PatchMatchOptions::default()
            };
            check(&gpu, &ws, &options);
        }
    }

    fn check(gpu: &GpuPatchMatch, ws: &Workspace, options: &PatchMatchOptions) {
        let within = |map: &DepthMap| {
            let mut good = 0;
            for y in 8..40 {
                for x in 12..52 {
                    if (map.depth[y * 64 + x] - 4.0).abs() < 0.04 {
                        good += 1;
                    }
                }
            }
            good
        };
        let inside = 32 * 40;
        let map = gpu.estimate(ws, 0, options, None);
        let cpu = estimate(ws, 0, options, None);
        assert!(
            within(&map) * 10 >= inside * 9,
            "GPU {:?}: {} of {inside}",
            options.view_selection,
            within(&map)
        );
        assert!(
            within(&cpu) * 10 >= inside * 9,
            "CPU {:?}: {} of {inside}",
            options.view_selection,
            within(&cpu)
        );
        let maps: Vec<DepthMap> = (0..3).map(|v| gpu.estimate(ws, v, options, None)).collect();
        let refined = gpu.estimate(ws, 0, options, Some(&maps));
        assert!(
            within(&refined) * 10 >= inside * 9,
            "GPU geometric {:?}: {}",
            options.view_selection,
            within(&refined)
        );
    }
}
