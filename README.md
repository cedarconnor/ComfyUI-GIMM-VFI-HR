# ComfyUI nodes to use GIMM-VFI frame interpolation


https://github.com/user-attachments/assets/be5dfa1d-ac6e-4c43-a812-d44799690244

Requires cupy, currently tested only with `cupy-cuda12==13.3.0`

Original repository:

https://github.com/GSeanCDAT/GIMM-VFI

## High-Resolution Support (8K, 16K+)

This implementation now supports **tile-based processing** for extremely high-resolution frame interpolation, including 16K sequences!

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



