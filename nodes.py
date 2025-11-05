import os
import torch

import folder_paths
import yaml
import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file

from omegaconf import OmegaConf
from tqdm import tqdm
import cv2

from .gimmvfi.generalizable_INR.gimmvfi_r import GIMMVFI_R
from .gimmvfi.generalizable_INR.gimmvfi_f import GIMMVFI_F

from .gimmvfi.generalizable_INR.configs import GIMMVFIConfig
from .gimmvfi.generalizable_INR.raft import RAFT
from .gimmvfi.generalizable_INR.flowformer.core.FlowFormer.LatentCostFormer.transformer import FlowFormer
from .gimmvfi.generalizable_INR.flowformer.configs.submission import get_cfg
from .gimmvfi.utils.flow_viz import flow_to_image
from .gimmvfi.utils.utils import InputPadder, RaftArgs, easydict_to_dict

from contextlib import nullcontext
import math

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

script_directory = os.path.dirname(os.path.abspath(__file__))


# Tiling helper functions for high-resolution processing
def compute_grid_indices(image_shape, tile_size, min_overlap=20):
    """
    Compute tile positions for processing large images.

    Args:
        image_shape: (H, W) tuple of the image dimensions
        tile_size: Size of each tile (can be int or tuple)
        min_overlap: Minimum overlap between adjacent tiles in pixels

    Returns:
        List of (h, w) tuples representing top-left corner of each tile
    """
    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)

    if min_overlap >= tile_size[0] or min_overlap >= tile_size[1]:
        raise ValueError(f"Overlap {min_overlap} must be less than tile size {tile_size}")

    # Generate tile positions
    hs = list(range(0, image_shape[0], tile_size[0] - min_overlap))
    ws = list(range(0, image_shape[1], tile_size[1] - min_overlap))

    # Make sure the final tiles are flush with the image boundary
    if len(hs) > 1:
        hs[-1] = max(0, image_shape[0] - tile_size[0])
    if len(ws) > 1:
        ws[-1] = max(0, image_shape[1] - tile_size[1])

    return [(h, w) for h in hs for w in ws]


def compute_weight(hws, image_shape, tile_size, sigma=1.0, device='cuda'):
    """
    Compute Gaussian weights for blending overlapping tiles.

    Args:
        hws: List of (h, w) tile positions from compute_grid_indices
        image_shape: (H, W) tuple of the image dimensions
        tile_size: Size of each tile (can be int or tuple)
        sigma: Gaussian sigma for weight computation
        device: Device to create tensors on

    Returns:
        List of weight tensors, one for each tile
    """
    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)

    patch_num = len(hws)

    # Create meshgrid for tile
    h, w = torch.meshgrid(
        torch.arange(tile_size[0], device=device),
        torch.arange(tile_size[1], device=device),
        indexing='ij'
    )
    h, w = h.float() / float(tile_size[0]), w.float() / float(tile_size[1])

    # Center at 0.5, 0.5
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w

    # Compute Gaussian weights (higher in center, lower at edges)
    weights_hw = (h**2 + w**2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    # Create weight map for full image
    weights = torch.zeros(1, patch_num, *image_shape, device=device)
    for idx, (h_pos, w_pos) in enumerate(hws):
        weights[:, idx, h_pos:h_pos + tile_size[0], w_pos:w_pos + tile_size[1]] = weights_hw

    # Extract per-tile weights
    patch_weights = []
    for idx, (h_pos, w_pos) in enumerate(hws):
        patch_weights.append(
            weights[:, idx:idx + 1, h_pos:h_pos + tile_size[0], w_pos:w_pos + tile_size[1]]
        )

    return patch_weights


class DownloadAndLoadGIMMVFIModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ([
                    "gimmvfi_r_arb_lpips_fp32.safetensors",
                    "gimmvfi_f_arb_lpips_fp32.safetensors"
                    ],),
               },
               "optional": {
                    "precision": (["fp32", "bf16", "fp16"], {"default": "fp32"}),
                    "torch_compile": ("BOOLEAN", {"default": False, "tooltip": "Compile part of the model with torch.compile, requires Triton"}),
               },
        }

    RETURN_TYPES = ("GIMMVIF_MODEL",)
    RETURN_NAMES = ("gimmvfi_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "GIMM-VFI"
    DESCRIPTION = "Downloads and loads GIMM-VFI model from folder 'ComfyUI\models\interpolation\gimm-vfi'"

    def loadmodel(self, model, precision="fp32", torch_compile=False):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[precision]

        download_path = os.path.join(folder_paths.models_dir, 'interpolation', 'gimm-vfi')
        model_path = os.path.join(download_path, model)

        if not os.path.exists(model_path):
            log.info(f"Downloading GMMI-VFI model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="Kijai/GIMM-VFI_safetensors",
                allow_patterns=[f"*{model}*"],
                local_dir=download_path,
                local_dir_use_symlinks=False,
            )

        if "gimmvfi_r" in model:
            config_path = os.path.join(script_directory, "configs", "gimmvfi", "gimmvfi_r_arb.yaml")
            flow_model = "raft-things_fp32.safetensors"
        elif "gimmvfi_f" in model:
            config_path = os.path.join(script_directory, "configs", "gimmvfi", "gimmvfi_f_arb.yaml")
            flow_model = "flowformer_sintel_fp32.safetensors"

        flow_model_path = os.path.join(folder_paths.models_dir, 'interpolation', 'gimm-vfi', flow_model)

        if not os.path.exists(flow_model_path):
            log.info(f"Downloading RAFT model to: {flow_model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="Kijai/GIMM-VFI_safetensors",
                allow_patterns=[f"*{flow_model}*"],
                local_dir=download_path,
                local_dir_use_symlinks=False,
            )
       
            
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config = easydict_to_dict(config)
        config = OmegaConf.create(config)
        arch_defaults = GIMMVFIConfig.create(config.arch)
        config = OmegaConf.merge(arch_defaults, config.arch)

        # load model
        if "gimmvfi_r" in model:
            model = GIMMVFI_R(dtype, config)
             #load RAFT
            raft_args = RaftArgs(
                small=False,
                mixed_precision=False,
                alternate_corr=False
            )
        
            raft_model = RAFT(raft_args)
            raft_sd = load_torch_file(flow_model_path)
            raft_model.load_state_dict(raft_sd, strict=True)
            raft_model.to(dtype).to(device)
            flow_estimator = raft_model
        elif "gimmvfi_f" in model:
            model = GIMMVFI_F(dtype, config)
            cfg = get_cfg()
            flowformer = FlowFormer(cfg.latentcostformer)
            flowformer_sd = load_torch_file(flow_model_path)
            flowformer.load_state_dict(flowformer_sd, strict=True)
            flow_estimator = flowformer.to(dtype).to(device)
            
       
        sd = load_torch_file(model_path)
        model.load_state_dict(sd, strict=False)
      
        model.flow_estimator = flow_estimator
        model = model.eval().to(dtype).to(device)

        if torch_compile:
            model = torch.compile(model)
            
        return (model,)

#region Interpolate
class GIMMVFI_interpolate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gimmvfi_model": ("GIMMVIF_MODEL",),
                "images": ("IMAGE", {"tooltip": "The images to interpolate between"}),
                "ds_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "interpolation_factor": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "output_flows": ("BOOLEAN", {"default": False, "tooltip": "Output the flow tensors"}),
                "enable_tiling": ("BOOLEAN", {"default": False, "tooltip": "Enable tiling for high-resolution processing (required for 8K+)"}),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64, "tooltip": "Size of each tile (larger = faster but more memory)"}),
                "tile_overlap": ("INT", {"default": 64, "min": 20, "max": 256, "step": 4, "tooltip": "Overlap between tiles for seamless blending"}),
                "blend_sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Gaussian sigma for tile blending (higher = smoother)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("images", "flow_tensors",)
    FUNCTION = "interpolate"
    CATEGORY = "PyramidFlowWrapper"

    def interpolate(self, gimmvfi_model, images, ds_factor, interpolation_factor, seed,
                   output_flows=False, enable_tiling=False, tile_size=512,
                   tile_overlap=64, blend_sigma=1.0):
        mm.soft_empty_cache()
        images = images.permute(0, 3, 1, 2)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        dtype = gimmvfi_model.dtype
        

        out_images_list = []
        flows = []
        start = 0
        end = images.shape[0] - 1
        pbar = ProgressBar(images.shape[0] - 1)

        autocast_device = mm.get_autocast_device(device)
        cast_context = torch.autocast(device_type=autocast_device, dtype=dtype) if dtype != torch.float32 else nullcontext()

        with cast_context:
            for j in tqdm(range(start, end)):
                I0 = images[j].unsqueeze(0)
                I2 = images[j+1].unsqueeze(0)

                if j == start:
                    out_images_list.append(I0.squeeze(0).permute(1, 2, 0))

                padder = InputPadder(I0.shape, 32)
                I0_padded, I2_padded = padder.pad(I0, I2)

                if not enable_tiling:
                    # Original full-frame processing
                    xs = torch.cat((I0_padded.unsqueeze(2), I2_padded.unsqueeze(2)), dim=2).to(device, non_blocking=True)

                    batch_size = xs.shape[0]
                    s_shape = xs.shape[-2:]

                    coord_inputs = [
                        (
                            gimmvfi_model.sample_coord_input(
                                batch_size,
                                s_shape,
                                [1 / interpolation_factor * i],
                                device=xs.device,
                                upsample_ratio=ds_factor,
                            ),
                            None,
                        )
                        for i in range(1, interpolation_factor)
                    ]
                    timesteps = [
                        i * 1 / interpolation_factor * torch.ones(xs.shape[0]).to(xs.device)
                        for i in range(1, interpolation_factor)
                    ]

                    all_outputs = gimmvfi_model(xs, coord_inputs, t=timesteps, ds_factor=ds_factor)
                    out_frames = [padder.unpad(im) for im in all_outputs["imgt_pred"]]
                    out_flowts = [padder.unpad(f) for f in all_outputs["flowt"]]
                else:
                    # Tile-based processing for high resolution
                    log.info(f"Processing frame pair {j+1}/{end} with tiling: tile_size={tile_size}, overlap={tile_overlap}")

                    # Get padded image shape
                    padded_shape = I0_padded.shape[-2:]  # (H, W)

                    # Compute tile positions and Gaussian weights
                    tile_positions = compute_grid_indices(padded_shape, tile_size, tile_overlap)
                    tile_weights = compute_weight(tile_positions, padded_shape, tile_size,
                                                 sigma=blend_sigma, device=device)

                    log.info(f"Image size: {padded_shape}, Number of tiles: {len(tile_positions)}")

                    # Process each interpolation timestep separately
                    out_frames = []
                    out_flowts = []

                    for timestep_idx in range(1, interpolation_factor):
                        t_value = timestep_idx / interpolation_factor

                        # Initialize accumulators for this timestep
                        accumulated_frame = torch.zeros_like(I0_padded)
                        accumulated_flow = torch.zeros(I0_padded.shape[0], 2, *padded_shape,
                                                      device=device, dtype=dtype)
                        weight_sum = torch.zeros(I0_padded.shape[0], 1, *padded_shape,
                                                device=device, dtype=dtype)

                        # Process each tile
                        for tile_idx, (h, w) in enumerate(tile_positions):
                            h_end = min(h + tile_size, padded_shape[0])
                            w_end = min(w + tile_size, padded_shape[1])

                            # Extract tile from both frames
                            I0_tile = I0_padded[:, :, h:h_end, w:w_end]
                            I2_tile = I2_padded[:, :, h:h_end, w:w_end]

                            # Create input for model
                            xs_tile = torch.cat((I0_tile.unsqueeze(2), I2_tile.unsqueeze(2)), dim=2).to(device, non_blocking=True)

                            batch_size = xs_tile.shape[0]
                            tile_shape = xs_tile.shape[-2:]

                            # Create coordinate input for this tile and timestep
                            coord_input = (
                                gimmvfi_model.sample_coord_input(
                                    batch_size,
                                    tile_shape,
                                    [t_value],
                                    device=xs_tile.device,
                                    upsample_ratio=ds_factor,
                                ),
                                None,
                            )

                            timestep = t_value * torch.ones(xs_tile.shape[0]).to(xs_tile.device)

                            # Run model on tile
                            tile_outputs = gimmvfi_model(xs_tile, [coord_input], t=[timestep], ds_factor=ds_factor)

                            # Get tile output
                            tile_frame = tile_outputs["imgt_pred"][0]
                            tile_flow = tile_outputs["flowt"][0]

                            # Get weight for this tile
                            tile_weight = tile_weights[tile_idx]

                            # Accumulate weighted results
                            actual_h = h_end - h
                            actual_w = w_end - w
                            accumulated_frame[:, :, h:h_end, w:w_end] += tile_frame[:, :, :actual_h, :actual_w] * tile_weight[:, :, :actual_h, :actual_w]
                            accumulated_flow[:, :, h:h_end, w:w_end] += tile_flow[:, :, :actual_h, :actual_w] * tile_weight[:, :, :actual_h, :actual_w]
                            weight_sum[:, :, h:h_end, w:w_end] += tile_weight[:, :, :actual_h, :actual_w]

                        # Normalize by weight sum
                        accumulated_frame = accumulated_frame / (weight_sum + 1e-8)
                        accumulated_flow = accumulated_flow / (weight_sum + 1e-8)

                        # Unpad and store
                        out_frames.append(padder.unpad(accumulated_frame))
                        out_flowts.append(padder.unpad(accumulated_flow))

                if output_flows:
                    flowt_imgs = [
                        flow_to_image(
                            flowt.squeeze().detach().cpu().permute(1, 2, 0).numpy(),
                            convert_to_bgr=True,
                        )
                        for flowt in out_flowts
                    ]
                I1_pred_img = [
                    (I1_pred[0].detach().cpu().permute(1, 2, 0))
                    for I1_pred in out_frames
                ]

                for i in range(interpolation_factor - 1):
                    out_images_list.append(I1_pred_img[i])
                    if output_flows:
                        flows.append(flowt_imgs[i])

                out_images_list.append(
                    ((padder.unpad(I2_padded)).squeeze().detach().cpu().permute(1, 2, 0))
                )
                pbar.update(1)
        
        image_tensors = torch.stack(out_images_list)
        image_tensors = image_tensors.cpu().float()

        rgb_images = [cv2.cvtColor(flow, cv2.COLOR_BGR2RGB) for flow in flows]

        if output_flows:
            flow_tensors = torch.stack([torch.from_numpy(image) for image in rgb_images])
            flow_tensors = flow_tensors / 255.0
            flow_tensors = flow_tensors.cpu().float()
        else:
            flow_tensors = torch.zeros(1, 64, 64, 3)


        return (image_tensors, flow_tensors)

NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadGIMMVFIModel": DownloadAndLoadGIMMVFIModel,
    "GIMMVFI_interpolate": GIMMVFI_interpolate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadGIMMVFIModel": "(Down)Load GIMMVFI Model",
    "GIMMVFI_interpolate": "GIMM-VFI Interpolate",
    }
