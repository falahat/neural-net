//! GPU backend via [CubeCL](https://github.com/tracel-ai/cubecl) +
//! wgpu. One matmul kernel, two ways to call it.
//!
//! Use [`GpuContext`] for any real workload — upload tensors once,
//! reuse output buffers via a pool, sync explicitly. The slice-based
//! [`matmul`] function is the convenience wrapper around it; every
//! call pays a full host↔device round trip.
//!
//! Dead ends we measured and dropped (don't re-add without new data):
//! 16×16 shared-memory tiled kernel (worse at every shape but
//! GPU-hot, where it won 16% — not worth maintaining; Blackwell's
//! 96 MB L2 already caches the matrices the textbook tiling was
//! trying to keep close). Register blocking (compounds with tiling;
//! moot without it). TF32 tensor cores (needs WMMA, not in cubecl 0.6).
//!
//! ## Architecture
//!
//! Setup gotchas (worked through in `bench_results/CUDA_SETUP.md`):
//! holding cubecl at 0.6 dodges a `windows-core` version conflict
//! on Windows DX12; CUDA 13.x moved DLLs to `bin\x64\`; nvcc needs
//! `NVCC_CCBIN` pointing at MSVC's `cl.exe` and the conforming-
//! preprocessor flag for CCCL headers.

use cubecl::prelude::*;
use std::sync::OnceLock;

/// Log the wgpu adapter cubecl picked. Laptops with NVIDIA Optimus
/// expose both the discrete dGPU and an Intel iGPU; if wgpu's
/// high-performance preference picks the wrong one, we want a loud
/// warning, not silent slowness. `NN_GPU_LOG=1` re-logs every call.
pub fn log_adapter_once() {
    static LOGGED: OnceLock<()> = OnceLock::new();
    let always = std::env::var("NN_GPU_LOG").ok().as_deref() == Some("1");
    if !always && LOGGED.get().is_some() {
        return;
    }

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    eprintln!("[nn::backend::gpu] wgpu adapters available:");
    for (i, a) in adapters.iter().enumerate() {
        let info = a.get_info();
        eprintln!(
            "  [{}] {} ({:?}, backend={:?}, vendor=0x{:04x})",
            i, info.name, info.device_type, info.backend, info.vendor
        );
    }

    // Pick the high-performance adapter — same selection cubecl uses.
    if let Ok(picked) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })) {
        let info = picked.get_info();
        eprintln!(
            "[nn::backend::gpu] high-performance adapter chosen: \
                   {} (device_type={:?}, vendor=0x{:04x})",
            info.name, info.device_type, info.vendor
        );
        const NVIDIA_VENDOR_ID: u32 = 0x10DE;
        if info.vendor != NVIDIA_VENDOR_ID && info.device_type == wgpu::DeviceType::IntegratedGpu {
            eprintln!("[nn::backend::gpu] ⚠ WARNING: running on INTEGRATED GPU.");
            eprintln!("[nn::backend::gpu]   On Windows + NVIDIA Optimus, force the discrete");
            eprintln!("[nn::backend::gpu]   GPU via NVIDIA Control Panel → 3D Settings →");
            eprintln!("[nn::backend::gpu]   Program Settings → add this exe → High-performance.");
        } else if info.vendor == NVIDIA_VENDOR_ID {
            eprintln!("[nn::backend::gpu] ✓ confirmed: discrete NVIDIA GPU.");
        }
    }
    LOGGED.set(()).ok();
}

/// One thread per output element. `a:[m,k]`, `b:[k,n]`, `out:[m,n]`,
/// row-major. The `if` wraps the whole body because cubecl 0.6
/// disallows early `return` from kernels.
#[cube(launch_unchecked)]
fn matmul_kernel_naive<F: Float>(
    a: &Array<F>,
    b: &Array<F>,
    out: &mut Array<F>,
    k: u32,
    n: u32,
    m: u32,
) {
    let row = ABSOLUTE_POS_X;
    let col = ABSOLUTE_POS_Y;
    if row < m && col < n {
        let mut acc = F::new(0.0);
        for kk in 0..k {
            acc += a[row * k + kk] * b[kk * n + col];
        }
        out[row * n + col] = acc;
    }
}

// ─── Resident-tensor API + output buffer pool ─────────────────────────

#[cfg(feature = "gpu")]
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

#[cfg(feature = "gpu")]
type Rt = WgpuRuntime;

#[cfg(feature = "gpu")]
type Client = cubecl::client::ComputeClient<
    <Rt as cubecl::Runtime>::Server,
    <Rt as cubecl::Runtime>::Channel,
>;

#[cfg(feature = "gpu")]
type GpuHandle = cubecl::server::Handle;

/// GPU-resident tensor. Handle is Arc'd inside cubecl, so cloning
/// is cheap; constructing one always involves a host→device upload.
#[cfg(feature = "gpu")]
pub struct GpuTensor {
    handle: GpuHandle,
    pub shape: Vec<usize>,
}

#[cfg(feature = "gpu")]
impl GpuTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn size_bytes(&self) -> usize {
        self.numel() * std::mem::size_of::<f32>()
    }
}

/// Owns the wgpu device + compute client + a pool of output
/// buffers keyed by byte size. The pool exists because
/// `client.empty()` is hundreds of µs on its own — the same cost
/// as a small kernel — so allocating fresh every call eats most of
/// the wall time at widget-scale matmuls.
#[cfg(feature = "gpu")]
pub struct GpuContext {
    device: WgpuDevice,
    client: Client,
    output_pool: std::cell::RefCell<std::collections::HashMap<usize, Vec<GpuHandle>>>,
    pool_hits: std::cell::Cell<u64>,
    pool_misses: std::cell::Cell<u64>,
}

#[cfg(feature = "gpu")]
impl GpuContext {
    pub fn new() -> Self {
        log_adapter_once();
        let device = WgpuDevice::default();
        let client = <Rt as cubecl::Runtime>::client(&device);
        Self {
            device,
            client,
            output_pool: Default::default(),
            pool_hits: Default::default(),
            pool_misses: Default::default(),
        }
    }

    /// Host→device upload. Do this once per input, outside any hot loop.
    pub fn upload(&self, data: &[f32], shape: Vec<usize>) -> GpuTensor {
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "GpuContext::upload: data length doesn't match shape"
        );
        GpuTensor {
            handle: self.client.create(f32_bytes(data)),
            shape,
        }
    }

    /// Device→host download. Blocks until all queued GPU work
    /// touching this tensor has finished.
    pub fn download(&self, t: &GpuTensor) -> Vec<f32> {
        let bytes = self.client.read_one(t.handle.clone().binding());
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn checkout_output(&self, size_bytes: usize) -> GpuHandle {
        let mut pool = self.output_pool.borrow_mut();
        if let Some(slots) = pool.get_mut(&size_bytes) {
            if let Some(h) = slots.pop() {
                self.pool_hits.set(self.pool_hits.get() + 1);
                return h;
            }
        }
        self.pool_misses.set(self.pool_misses.get() + 1);
        self.client.empty(size_bytes)
    }

    /// Return an output buffer to the pool so the next same-shape
    /// matmul skips the allocator. Dropping `t` without releasing
    /// also works — you just lose the pool hit.
    pub fn release(&self, t: GpuTensor) {
        let size = t.size_bytes();
        self.output_pool
            .borrow_mut()
            .entry(size)
            .or_default()
            .push(t.handle);
    }

    pub fn matmul(&self, a: &GpuTensor, b: &GpuTensor) -> GpuTensor {
        assert_eq!(a.shape.len(), 2);
        assert_eq!(b.shape.len(), 2);
        let (m, k) = (a.shape[0] as u32, a.shape[1] as u32);
        let (k2, n) = (b.shape[0] as u32, b.shape[1] as u32);
        assert_eq!(k, k2, "matmul: inner dims {k} vs {k2}");

        let out_bytes = (m as usize * n as usize) * std::mem::size_of::<f32>();
        let out_handle = self.checkout_output(out_bytes);

        // 16×16 workgroup, ceil(m/16) × ceil(n/16) workgroups.
        // 256 threads/workgroup = 8 warps, the minimum that fully
        // occupies one SM. A CubeDim(1,1,1) launch would run 1 thread
        // per workgroup and leave the GPU 97% idle.
        const WG: u32 = 16;
        let cube_count = CubeCount::Static((m + WG - 1) / WG, (n + WG - 1) / WG, 1);
        let cube_dim = CubeDim::new(WG, WG, 1);

        unsafe {
            matmul_kernel_naive::launch_unchecked::<f32, Rt>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<f32>(&a.handle, (m * k) as usize, 1),
                ArrayArg::from_raw_parts::<f32>(&b.handle, (k * n) as usize, 1),
                ArrayArg::from_raw_parts::<f32>(&out_handle, (m * n) as usize, 1),
                cubecl::frontend::ScalarArg::new(k),
                cubecl::frontend::ScalarArg::new(n),
                cubecl::frontend::ScalarArg::new(m),
            );
        }

        GpuTensor {
            handle: out_handle,
            shape: vec![m as usize, n as usize],
        }
    }

    /// `(hits, misses)` for the output buffer pool. Read after a
    /// hot loop to confirm reuse is actually happening.
    pub fn pool_stats(&self) -> (u64, u64) {
        (self.pool_hits.get(), self.pool_misses.get())
    }

    /// Block until every queued GPU op finishes. Required between
    /// kernel launches for honest timing — `matmul()` returns the
    /// moment the kernel is queued, NOT when it's done. Real
    /// workloads get this implicitly via the next `download()`.
    pub fn sync(&self) {
        pollster::block_on(self.client.sync());
    }

    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }
}

/// Slow path. Allocates a fresh `GpuContext`, uploads, dispatches,
/// downloads, all per call. Use [`GpuContext`] directly in any
/// loop or layer.
pub fn matmul(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> (Vec<f32>, Vec<usize>) {
    assert_eq!(a_shape.len(), 2);
    assert_eq!(b_shape.len(), 2);

    #[cfg(feature = "gpu")]
    {
        let ctx = GpuContext::new();
        let a_t = ctx.upload(a, a_shape.to_vec());
        let b_t = ctx.upload(b, b_shape.to_vec());
        let c_t = ctx.matmul(&a_t, &b_t);
        let out = ctx.download(&c_t);
        let shape = c_t.shape.clone();
        return (out, shape);
    }

    #[cfg(not(feature = "gpu"))]
    {
        let _ = (a, b, a_shape, b_shape);
        unimplemented!("backend::gpu::matmul: `gpu-cuda` runtime not wired");
    }
}

// Safety: f32 is plain-old-data; the byte view shares lifetime with the slice.
fn f32_bytes(xs: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(xs.as_ptr() as *const u8, std::mem::size_of_val(xs)) }
}
