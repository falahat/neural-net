# Setting up CUDA for the candle-cuda benchmark

If you want to run the `candle-cuda` peer on a Windows machine
with an NVIDIA GPU. The driver is **not enough** — you also need
the CUDA Toolkit (specifically `nvcc`, which `cudarc` invokes
at build time).

## Quick check before you start

```powershell
nvidia-smi      # should show your GPU + driver version
where.exe nvcc  # ← if this fails, CUDA Toolkit is NOT installed
echo $env:CUDA_PATH    # ← should point to the toolkit dir after install
```

If `nvcc` is missing, follow the rest of this doc.

## Pick a CUDA version that matches your GPU

| GPU generation | Compute capability | Minimum CUDA |
|---|---|---|
| Maxwell / Pascal (GTX 9xx, 10xx) | 5.2 / 6.x | 11.x |
| Turing (RTX 20xx) | 7.5 | 11.x |
| Ampere (RTX 30xx, A100) | 8.x | 11.x |
| Ada (RTX 40xx) | 8.9 | 11.8 |
| **Hopper (H100, RTX 50xx Laptop)** | 9.0 / 12.0 | **12.6+** |
| Blackwell datacenter (B100, B200) | 10.0 | 12.8+ |

If you have an RTX 5070 / 5080 / 5090 (laptop or desktop), pick CUDA
**12.6 or newer**. Older toolkits can't compile kernels for the
`sm_120` target Blackwell needs.

## Install — Windows

1. Go to <https://developer.nvidia.com/cuda-downloads>.
2. Pick: OS = **Windows**, Architecture = **x86_64**, Version = **11**, Installer Type = **exe (local)**.
   - "Local" downloads everything as one file (~3 GB); "network" downloads on demand. Local is more reliable.
3. Run the installer. On the "Installation Options" screen:
   - **Custom** install.
   - Required: **CUDA → Toolkit** (the only must-have).
   - Optional and safe to uncheck: GeForce Experience, NVIDIA HD Audio, Visual Studio integration if you don't use VS.
4. Accept the rest of the defaults. Install takes 10-15 min.
5. **Restart your terminal** (env vars get set during install).
6. Verify:
   ```powershell
   where.exe nvcc                # should print path to nvcc.exe
   nvcc --version                # should print CUDA 12.6 (or newer)
   echo $env:CUDA_PATH           # should point at the install dir
   ```

## Install — Linux

```bash
# Ubuntu 22.04 / 24.04 example. Adapt for your distro.
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6   # or whichever version matches your GPU

# Add to your shell rc:
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Verify
nvcc --version
```

## Run the candle-cuda bench

Once `nvcc --version` works in a fresh shell:

```powershell
# From the repo root
cargo run --profile release-bench -p nn-bench --example candle --features candle-cuda -- `
    --label candle-cuda `
    --out crates/engine/nn/bench_results/candle_cuda.json

# Then rebuild the dashboard
cargo run --release --example build_viz -p nn

# Open
start crates\ml\nn\bench_results\viz.html
```

The dashboard auto-picks up `candle_cuda.json` — you'll see a new
KPI card and the GFLOP/s curve will gain a candle-cuda line.

## What candle-cuda gets you

On a recent NVIDIA GPU (RTX 4070+) at GPU-hot (2048×1024×1024),
candle-cuda typically lands in the 8000-15000 GFLOP/s range —
about **10-20× the multi-threaded CPU** we have today, and
**500-1000× our scalar baseline**.

This is the upper bound we're benchmarking against. Our own
`--features gpu` path (cubecl/wgpu) currently sits at ~12 GFLOP/s
because the v1 kernel is the naive one-thread-per-output-element
with per-call buffer transfer. A tiled kernel with buffer pooling closes
the gap — see `docs/designs/nn_benchmark_harness.md` §5.1.

## Troubleshooting

**"nvcc fatal: Unsupported gpu architecture 'compute_120'"** — your
CUDA Toolkit is too old for the GPU. Install CUDA 12.6+.

**"linker 'link.exe' not found"** — install Visual Studio Build
Tools 2022 with the "Desktop C++" workload; CUDA on Windows
requires MSVC's linker.

**"could not find 'libcudart.so'"** (Linux) — `LD_LIBRARY_PATH`
isn't set. Re-check the install commands above.

**Compile is slow (10+ min)** — first build of cudarc compiles
a lot of CUDA driver bindings. Subsequent builds are fast.

**`STATUS_DLL_NOT_FOUND` (exit code `0xc0000135`) on Windows** —
candle's CUDA DLLs aren't on PATH at runtime. CUDA 13+ moved
them from `bin\` to `bin\x64\`. Either add both to PATH or run
the bench from a shell where `where.exe cudart64_13.dll`
resolves.

**`nvcc fatal: Cannot find compiler 'cl.exe' in PATH`** — nvcc
spawns its own subprocess to invoke MSVC and that subprocess
strips PATH. Set `NVCC_CCBIN` env var to the absolute path of
the directory containing `cl.exe`:
```powershell
$env:NVCC_CCBIN = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\<version>\bin\Hostx64\x64"
```

**`fatal error C1189: ... /Zc:preprocessor` from CCCL on CUDA 13** —
CUDA 13's CCCL headers require MSVC's conforming preprocessor.
Set:
```powershell
$env:NVCC_PREPEND_FLAGS = "-Xcompiler /Zc:preprocessor"
```

**`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` at runtime when candle
loads its elementwise kernels** — this is the gnarly one. Your
driver doesn't accept the PTX bytecode version your toolkit
emits. Driver / toolkit pairing rules:

| Driver build | Highest PTX it loads | Toolkit that emits compatible PTX |
|---|---|---|
| ≤ 570 | PTX 8.2 | ≤ CUDA 12.4 |
| 575-585 | PTX 8.3 | ≤ CUDA 12.8 |
| 580-595 | PTX 8.3 (most) → 8.4 (latest) | CUDA 12.x and most 13.0/13.1 |
| 600+ | PTX 8.4+ | CUDA 13.2+ |

Practical fixes:
- **Best:** update the driver to ≥ 600 (NVIDIA's latest
  GeForce Game Ready or Studio driver). 5-min reboot.
- **Alternative:** install CUDA Toolkit 12.8 (last toolkit
  that emits PTX 8.3, which every driver ≥ 575 accepts).
  You can keep 13.2 installed too — they coexist in
  versioned dirs; just point `CUDA_PATH` at the older one
  when building candle-cuda.

Matmul through cuBLAS is unaffected because cuBLAS ships
pre-built SASS for every supported architecture; the PTX JIT
path is only for kernels not in cuBLAS (elementwise, reductions).
