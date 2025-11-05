# ComfyUI-GIMM-VFI-HR: High-Resolution Frame Interpolation

https://github.com/user-attachments/assets/be5dfa1d-ac6e-4c43-a812-d44799690244

ComfyUI nodes for **GIMM-VFI** (Generalized Implicit Neural Representation for Motion-Guided Video Frame Interpolation), with enhanced support for **extreme high-resolution processing** including 8K, 16K, and beyond.

## 🌟 Features

- **🎬 Advanced Frame Interpolation**: State-of-the-art video frame interpolation using implicit neural representations
- **🖼️ High-Resolution Support**: Tile-based processing for 8K, 16K+ resolutions
- **🚀 Multiple Model Variants**: Choose between RAFT-based (GIMMVFI-R) or FlowFormer-based (GIMMVFI-F) architectures
- **⚡ Flexible Precision**: Support for fp32, fp16, and bf16 for speed/quality trade-offs
- **🔧 Performance Options**: Configurable downsampling, torch.compile support, and more
- **📊 Flow Visualization**: Optional optical flow output for debugging and analysis
- **🎯 High Interpolation Factors**: Generate up to 100 intermediate frames between any two frames

## 📦 Installation

### Requirements

- **CUDA 12.x** compatible GPU
- **CuPy**: `cupy-cuda12>=13.3.0` (required for soft splatting operations)
- ComfyUI installation

### Install Dependencies

```bash
pip install cupy-cuda12==13.3.0
```

Or install from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Models

Models are automatically downloaded from HuggingFace on first use and stored in:
```
ComfyUI/models/interpolation/gimm-vfi/
```

**Available Models:**
- `gimmvfi_r_arb_lpips_fp32.safetensors` (RAFT-based, ~180MB)
- `gimmvfi_f_arb_lpips_fp32.safetensors` (FlowFormer-based, ~200MB)
- `raft-things_fp32.safetensors` (RAFT flow estimator)
- `flowformer_sintel_fp32.safetensors` (FlowFormer flow estimator)

Original repository: https://github.com/GSeanCDAT/GIMM-VFI

---

## 🎮 Nodes

### 1. DownloadAndLoadGIMMVFIModel

Loads the GIMM-VFI interpolation model with configurable precision and optimization settings.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | Choice | - | Model variant: `gimmvfi_r` (RAFT) or `gimmvfi_f` (FlowFormer) |
| `precision` | Choice | fp32 | Inference precision: `fp32`, `fp16`, or `bf16` |
| `torch_compile` | Boolean | False | Enable PyTorch 2.0 compilation (requires Triton) |

**Model Comparison:**

| Model | Flow Estimator | Quality | Speed | VRAM | Use Case |
|-------|----------------|---------|-------|------|----------|
| GIMMVFI-R | RAFT | Excellent | Medium | Medium | General purpose, robust motion |
| GIMMVFI-F | FlowFormer | Excellent | Faster | Slightly less | Modern transformer-based flow |

**Precision Trade-offs:**

| Precision | Quality | Speed | VRAM Savings | Compatibility |
|-----------|---------|-------|--------------|---------------|
| fp32 | Best | Slowest | Baseline | All GPUs |
| fp16 | Very Good | ~2x faster | ~50% | Most GPUs |
| bf16 | Very Good | ~2x faster | ~50% | RTX 30xx+, A100, H100 |

**Outputs:**
- `gimmvfi_model`: Model object for use in interpolation node

---

### 2. GIMMVFI_interpolate

Performs frame interpolation on image sequences with extensive configuration options.

**Required Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `gimmvfi_model` | Model | - | - | Model from loader node |
| `images` | IMAGE | - | - | Input image sequence (B, H, W, C) |
| `ds_factor` | Float | 1.0 | 0.01-1.0 | Flow estimation downsampling factor |
| `interpolation_factor` | Int | 8 | 1-100 | Number of frames to generate between each pair |
| `seed` | Int | 0 | 0-∞ | Random seed for reproducibility |

**Optional Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `output_flows` | Boolean | False | - | Output optical flow visualizations |
| `enable_tiling` | Boolean | False | - | Enable tile-based processing for high-res |
| `tile_size` | Int | 512 | 256-2048 | Size of each tile in pixels |
| `tile_overlap` | Int | 64 | 20-256 | Overlap between adjacent tiles |
| `blend_sigma` | Float | 1.0 | 0.1-2.0 | Gaussian sigma for tile blending |

**Outputs:**
- `images`: Interpolated image sequence
- `flow_tensors`: Optical flow visualizations (if `output_flows=True`)

---

## 🔧 Feature Deep Dive

### Downsampling Factor (`ds_factor`)

Controls the resolution at which optical flow is computed. Lower values reduce memory usage and speed up flow estimation.

**Recommendations:**
- **1.0** (full resolution): Best quality, highest memory usage
- **0.5** (half resolution): Good quality, 75% memory reduction
- **0.25** (quarter resolution): Acceptable quality for 16K+, major memory savings

**Use Cases:**
```
1080p-4K:     ds_factor = 1.0 (full quality)
8K:           ds_factor = 0.5-1.0 (balanced)
16K:          ds_factor = 0.25-0.5 (memory efficient)
```

### Interpolation Factor

Determines how many intermediate frames to generate between each input frame pair.

**Examples:**
- `interpolation_factor = 2`: 2x frame rate (24fps → 48fps)
- `interpolation_factor = 4`: 4x frame rate (24fps → 96fps)
- `interpolation_factor = 8`: 8x frame rate (24fps → 192fps)

**Calculation:**
```
Output frames = (Input frames - 1) × interpolation_factor + 1
```

For 10 input frames with `interpolation_factor=8`:
```
Output = (10 - 1) × 8 + 1 = 73 frames
```

### Flow Visualization (`output_flows`)

When enabled, outputs color-coded optical flow maps for analysis:
- **Red/Orange**: Motion to the right
- **Blue/Cyan**: Motion to the left
- **Green/Yellow**: Up/down motion
- **Brightness**: Motion magnitude

Useful for:
- Debugging motion artifacts
- Understanding scene dynamics
- Validating interpolation quality

### Torch Compile

Experimental feature using PyTorch 2.0's compilation for potential speedups (10-30%).

**Requirements:**
- PyTorch 2.0+
- Triton compiler
- First run will be slower (compilation overhead)

**Enable only if:**
- You're processing long sequences (>100 frames)
- You have Triton installed
- You accept potential instability

---

## 🖼️ High-Resolution Support (8K, 16K+)

This implementation supports **tile-based processing** for extremely high-resolution frame interpolation.

### How to Enable Tiling

In the `GIMM-VFI Interpolate` node, set the following optional parameters:

- **enable_tiling**: Set to `True` to enable tile-based processing (required for 8K+ resolutions)
- **tile_size**: Size of each tile in pixels (default: 512)
  - Larger values = faster but more VRAM usage
  - Recommended: 512-720 for most GPUs
- **tile_overlap**: Overlap between adjacent tiles in pixels (default: 64)
  - Higher values = smoother blending but slower processing
  - Recommended: 64-128 pixels
- **blend_sigma**: Gaussian sigma for tile blending (default: 1.0)
  - Higher values = smoother transitions between tiles
  - Recommended: 1.0-1.5

### Memory Requirements

| Resolution | Tiling Required? | Recommended Settings | Approx. VRAM |
|------------|------------------|----------------------|--------------|
| 1080p-2K   | No               | enable_tiling=False  | 8-12 GB      |
| 4K         | Optional         | tile_size=720        | 8-16 GB      |
| 8K         | Yes              | tile_size=512-720    | 12-24 GB     |
| 16K        | Yes              | tile_size=512        | 16-24 GB     |

### Example Usage

**For 16K frame interpolation:**
```
enable_tiling: True
tile_size: 512
tile_overlap: 64
blend_sigma: 1.0
ds_factor: 0.5 (optional, for even lower memory usage)
```

**For 4K frame interpolation (fast):**
```
enable_tiling: False
ds_factor: 1.0
```

### Technical Details

The tiling implementation uses:
- **Gaussian-weighted blending** for seamless tile fusion
- **Adaptive tile positioning** with configurable overlap
- **Per-tile flow estimation** maintaining accuracy at high resolutions
- **Memory-efficient accumulation** processing one tile at a time

For more details, see the original GIMM-VFI paper and FlowFormer tiling implementation

---

## 🎬 Complete Workflows

### Workflow 1: Standard HD/4K Video Interpolation

**Goal:** Convert 24fps 1080p/4K video to smooth 60fps

**Setup:**
```
1. Load frames: VideoLoadImages → images
2. Load model: DownloadAndLoadGIMMVFIModel
   - model: gimmvfi_f_arb_lpips_fp32.safetensors
   - precision: fp16
   - torch_compile: False
3. Interpolate: GIMMVFI_interpolate
   - gimmvfi_model: (from step 2)
   - images: (from step 1)
   - interpolation_factor: 2 (24fps → 48fps) or 3 (24fps → 72fps)
   - ds_factor: 1.0
   - enable_tiling: False
   - seed: 0
4. Save: VideoSaveImages
```

**Expected Performance:**
- Processing time: ~2-5 seconds per frame pair (RTX 4090)
- VRAM usage: 8-12 GB
- Quality: Excellent

---

### Workflow 2: High-Resolution 8K Interpolation

**Goal:** Smooth 8K footage (7680×4320) with memory efficiency

**Setup:**
```
1. Load frames: VideoLoadImages → images
2. Load model: DownloadAndLoadGIMMVFIModel
   - model: gimmvfi_r_arb_lpips_fp32.safetensors
   - precision: fp16
   - torch_compile: False
3. Interpolate: GIMMVFI_interpolate
   - gimmvfi_model: (from step 2)
   - images: (from step 1)
   - interpolation_factor: 2
   - ds_factor: 0.5
   - enable_tiling: True
   - tile_size: 720
   - tile_overlap: 96
   - blend_sigma: 1.2
   - seed: 0
4. Save: VideoSaveImages
```

**Expected Performance:**
- Processing time: ~15-30 seconds per frame pair
- VRAM usage: 16-20 GB
- Quality: Very Good

---

### Workflow 3: Extreme 16K Interpolation

**Goal:** Process 16K resolution (15360×8640) frames

**Setup:**
```
1. Load frames: VideoLoadImages → images
2. Load model: DownloadAndLoadGIMMVFIModel
   - model: gimmvfi_f_arb_lpips_fp32.safetensors
   - precision: fp16
   - torch_compile: False
3. Interpolate: GIMMVFI_interpolate
   - gimmvfi_model: (from step 2)
   - images: (from step 1)
   - interpolation_factor: 2
   - ds_factor: 0.25
   - enable_tiling: True
   - tile_size: 512
   - tile_overlap: 64
   - blend_sigma: 1.0
   - seed: 0
4. Save: VideoSaveImages
```

**Expected Performance:**
- Processing time: ~60-120 seconds per frame pair
- VRAM usage: 18-24 GB
- Quality: Good

**Key Optimizations:**
- Small tile size (512) keeps VRAM manageable
- Low ds_factor (0.25) reduces flow computation memory
- fp16 precision halves memory requirements
- FlowFormer model is faster than RAFT at extreme resolutions

---

### Workflow 4: Slow Motion Creation (High Interpolation Factor)

**Goal:** Create dramatic slow-motion from 30fps source to 240fps

**Setup:**
```
1. Load frames: VideoLoadImages → images
2. Load model: DownloadAndLoadGIMMVFIModel
   - model: gimmvfi_r_arb_lpips_fp32.safetensors
   - precision: fp32
   - torch_compile: False
3. Interpolate: GIMMVFI_interpolate
   - gimmvfi_model: (from step 2)
   - images: (from step 1)
   - interpolation_factor: 8 (30fps → 240fps)
   - ds_factor: 1.0
   - enable_tiling: False (assuming 1080p-4K)
   - seed: 0
4. Save: VideoSaveImages
```

**Best Practices:**
- Use full precision (fp32) for maximum quality in slow-mo
- Keep ds_factor at 1.0 for accurate motion
- Source footage should have minimal motion blur
- Avoid scenes with rapid complex motion

---

### Workflow 5: Motion Analysis with Flow Visualization

**Goal:** Debug interpolation issues by visualizing optical flow

**Setup:**
```
1. Load frames: VideoLoadImages → images
2. Load model: DownloadAndLoadGIMMVFIModel
   - model: gimmvfi_r_arb_lpips_fp32.safetensors
   - precision: fp32
3. Interpolate: GIMMVFI_interpolate
   - gimmvfi_model: (from step 2)
   - images: (from step 1)
   - interpolation_factor: 2
   - ds_factor: 1.0
   - output_flows: True ← Enable flow output
   - seed: 0
4a. Save interpolated frames: VideoSaveImages (images output)
4b. Save flow visualizations: VideoSaveImages (flow_tensors output)
```

**Flow Interpretation:**
- **Uniform colors**: Consistent motion (good)
- **Sharp boundaries**: Object edges detected correctly
- **Chaotic patterns**: Difficult motion, potential artifacts
- **Black regions**: No motion detected

---

### Workflow 6: Batch Processing Multiple Clips

**Goal:** Automate interpolation for multiple video clips

**Setup:**
```
1. Use batch processing wrapper node
2. For each clip:
   - Load frames: VideoLoadImages
   - Load model once (reuse): DownloadAndLoadGIMMVFIModel
   - Interpolate: GIMMVFI_interpolate
     - interpolation_factor: 2
     - ds_factor: 1.0
     - enable_tiling: Based on resolution
   - Save: VideoSaveImages
```

**Tips:**
- Load model once, reuse for all clips (saves time)
- Use consistent settings across clips for uniform quality
- Monitor VRAM between clips (ComfyUI auto-manages)
- Consider using fp16 for faster batch processing

---

## 🎯 Parameter Recommendation Guide

### Choose Your Model

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| General interpolation | GIMMVFI-R | Most robust, proven quality |
| Speed priority | GIMMVFI-F | Faster inference with FlowFormer |
| Complex motion | GIMMVFI-R | RAFT handles difficult motion better |
| Simple motion | GIMMVFI-F | FlowFormer is efficient for simple scenes |

### Choose Your Precision

| Scenario | Precision | Trade-off |
|----------|-----------|-----------|
| Maximum quality | fp32 | Slowest, highest VRAM |
| Balanced (recommended) | fp16 | 2x speed, 50% VRAM, minimal quality loss |
| RTX 30xx+/A100 | bf16 | Same as fp16 but better numerical stability |
| Memory constrained | fp16 + ds_factor=0.5 | Lowest VRAM usage |

### Choose Your Interpolation Factor

| Original FPS | Target FPS | Interpolation Factor |
|--------------|------------|---------------------|
| 24 | 48 | 2 |
| 24 | 60 | 3 (close to 2.5x) |
| 30 | 60 | 2 |
| 30 | 120 | 4 |
| 60 | 240 | 4 |

### Tiling Decision Matrix

| Resolution | Tiling | tile_size | tile_overlap | Reason |
|------------|--------|-----------|--------------|--------|
| ≤1080p | No | - | - | Fits in VRAM easily |
| 1440p-2K | No | - | - | Manageable with fp16 |
| 4K | Optional | 720 | 96 | Depends on VRAM (24GB+: no, 12GB: yes) |
| 5K-8K | Yes | 512-720 | 64-96 | Required for most GPUs |
| 16K+ | Yes | 512 | 64 | Essential for any GPU |

---

## ⚡ Performance Optimization Tips

### 1. Memory Optimization

**Problem:** Out of memory errors

**Solutions:**
```
✓ Enable tiling (enable_tiling=True)
✓ Reduce tile_size (512 instead of 720)
✓ Lower ds_factor (0.5 or 0.25)
✓ Use fp16 precision
✓ Reduce interpolation_factor if possible
✓ Process fewer frames at once
```

### 2. Speed Optimization

**Problem:** Processing is too slow

**Solutions:**
```
✓ Use GIMMVFI-F (FlowFormer) instead of GIMMVFI-R
✓ Enable fp16 precision
✓ Increase tile_size (if VRAM allows)
✓ Reduce tile_overlap (minimum 32, recommended 64)
✓ Lower ds_factor slightly (0.75-0.5)
✓ Try torch_compile (experimental, requires Triton)
✓ Disable output_flows
```

### 3. Quality Optimization

**Problem:** Artifacts or blurry results

**Solutions:**
```
✓ Use fp32 precision
✓ Set ds_factor=1.0 (full resolution flow)
✓ Increase tile_overlap (96-128) if using tiling
✓ Increase blend_sigma (1.2-1.5) for smoother tiles
✓ Use GIMMVFI-R for complex motion
✓ Lower interpolation_factor if motion is too fast
✓ Check source footage quality
```

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Enable tiling if not already enabled
2. Reduce `tile_size` (try 512 → 384 → 256)
3. Lower `ds_factor` (1.0 → 0.5 → 0.25)
4. Switch to fp16 precision
5. Close other GPU applications
6. Reduce `interpolation_factor`

### Issue: Visible Tile Seams

**Symptoms:** Visible grid pattern or seams in tiled output

**Solutions:**
1. Increase `tile_overlap` (64 → 96 → 128)
2. Increase `blend_sigma` (1.0 → 1.5 → 2.0)
3. Ensure tile_overlap < tile_size
4. Try different tile_size (avoid very small tiles)

### Issue: CuPy Import Error

**Symptoms:** `ModuleNotFoundError: No module named 'cupy'`

**Solution:**
```bash
pip install cupy-cuda12==13.3.0
```

Make sure CUDA version matches (CUDA 12.x requires `cupy-cuda12`)

### Issue: Blurry or Ghosting Artifacts

**Symptoms:** Double images, ghosting, or excessive blur

**Possible Causes & Solutions:**
1. **Source has motion blur:** Use higher shutter speed source footage
2. **ds_factor too low:** Increase to 0.75 or 1.0
3. **Interpolation factor too high:** Reduce to 2-4 for fast motion
4. **Wrong model:** Try switching between GIMMVFI-R and GIMMVFI-F

### Issue: Slow First Run

**Symptoms:** First interpolation takes very long

**Explanation:** Normal behavior
- Models are downloading from HuggingFace
- First run includes initialization overhead
- Torch.compile (if enabled) compiles on first run

**Solution:** Wait for first run to complete; subsequent runs will be much faster

### Issue: Inconsistent Results with Same Seed

**Symptoms:** Different outputs with same seed value

**Possible Causes:**
1. Different precision settings (fp16 is non-deterministic on some GPUs)
2. CuPy random operations
3. CUDA versions differ

**Solution:** Use fp32 for fully deterministic results

---

## 📊 Performance Benchmarks

### RTX 4090 (24GB VRAM)

| Resolution | Tiling | Precision | ds_factor | Time/Frame Pair | VRAM Usage |
|------------|--------|-----------|-----------|-----------------|------------|
| 1080p | No | fp32 | 1.0 | ~3s | 8 GB |
| 1080p | No | fp16 | 1.0 | ~1.5s | 5 GB |
| 4K | No | fp16 | 1.0 | ~8s | 14 GB |
| 4K | Yes (720) | fp16 | 1.0 | ~12s | 10 GB |
| 8K | Yes (512) | fp16 | 0.5 | ~35s | 18 GB |
| 16K | Yes (512) | fp16 | 0.25 | ~90s | 22 GB |

### RTX 3090 (24GB VRAM)

| Resolution | Tiling | Precision | ds_factor | Time/Frame Pair | VRAM Usage |
|------------|--------|-----------|-----------|-----------------|------------|
| 1080p | No | fp16 | 1.0 | ~2s | 5 GB |
| 4K | Yes (720) | fp16 | 1.0 | ~15s | 11 GB |
| 8K | Yes (512) | fp16 | 0.5 | ~45s | 19 GB |

*Note: Benchmarks are approximate and vary based on scene complexity and motion*

---

## 🙏 Credits

- **Original GIMM-VFI Paper:** [GSeanCDAT/GIMM-VFI](https://github.com/GSeanCDAT/GIMM-VFI)
- **RAFT:** Recurrent All-Pairs Field Transforms for Optical Flow
- **FlowFormer:** FlowFormer: A Transformer Architecture for Optical Flow
- **ComfyUI:** [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- **16K Tiling Implementation:** Based on FlowFormer's tile evaluation code

---

## 📄 License

Please refer to the original GIMM-VFI repository for licensing information.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for:
- Bug fixes
- Performance improvements
- Documentation enhancements
- New features

---

## 📮 Support

For issues and questions:
1. Check this README and troubleshooting section
2. Review the original GIMM-VFI repository documentation
3. Open an issue on GitHub with:
   - Your configuration (GPU, CUDA version, resolution)
   - Complete error message
   - Steps to reproduce



