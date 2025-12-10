#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_delight_batch.py
一次加载模型，多张图批量推理。
"""

import argparse
import logging
import os
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
from lbm.inference import evaluate_olat, get_model
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
PATH = os.path.dirname(os.path.abspath(__file__))

import torch.nn.functional as F

def pad_to_multiple(tensor, multiple=16):
    """将输入补齐到 multiple 的倍数尺寸，返回补齐后的tensor及补边参数"""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    pad = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
    tensor_padded = F.pad(tensor, pad, mode="reflect")
    pad = (0, 2, 0, 0)  # (left, right, top, bottom)
    return tensor_padded, pad

def unpad(tensor, pad):
    """按原pad反向裁切"""
    _, _, h, w = tensor.shape
    left, right, top, bottom = pad
    return tensor[:, :, top:h - bottom, left:w - right]
def _open_rgb(p):
    return Image.open(p).convert("RGB")

def load_mask(mask_path: str, target_size: tuple[int,int], device="cuda"):
    im = Image.open(mask_path)
    m = im.getchannel("A") if "A" in im.getbands() else im.convert("L")
    m = m.resize(target_size, resample=Image.NEAREST)
    return to_tensor(m).unsqueeze(0).to(device=device, dtype=torch.float32)

def _imread_hdr_float(path: Path) -> np.ndarray:
    """
    读 HDR/EXR -> float32（BGR/GRAY），不做 gamma/tonemap。
    OpenCV 以 BGR 返回。
    """
    img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    if img is None:
        # 兜底：有的系统 imread 会失败（奇怪的 hdr 变体），可提示提前转换
        raise IOError(f"Failed to read HDR: {path}")
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0
    img = img.astype(np.float32)
    return img

def _hdr_to_rgb_float_chw(img_hdr: np.ndarray, out_hw: Optional[Tuple[int,int]] = None, log_mode: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """
    HDR(BGR/GRAY) -> RGB float CHW，支持下采样和对数域变换以增强稳定性。
    """
    if img_hdr.ndim == 2:
        img_hdr = np.stack([img_hdr, img_hdr, img_hdr], axis=-1)
    elif img_hdr.shape[2] >= 3:
        img_hdr = img_hdr[:, :, :3]
    # BGR -> RGB
    img_hdr = img_hdr[:, :, ::-1]

    if out_hw is not None:
        H, W = out_hw
        img_hdr = cv2.resize(img_hdr, (W, H), interpolation=cv2.INTER_AREA)

    if log_mode:
        img_hdr = np.log(img_hdr + eps)  # 不做归一，模型学相对亮度关系
    # 直接以 float32 返回（不做 0..1 压缩）
    t = torch.from_numpy(img_hdr.astype(np.float32)).permute(2, 0, 1).contiguous()
    return t

# def _load_hdr_env(hpath: str) -> torch.Tensor:
#     hdr = _imread_hdr_float(hpath)
#     t = _hdr_to_rgb_float_chw(hdr, out_hw=(1024, 750), log_mode=True)
#     return t

def _load_hdr_env(hpath: str, out_hw: Optional[Tuple[int,int]] = None) -> torch.Tensor:
    hdr = _imread_hdr_float(hpath)
    t = _hdr_to_rgb_float_chw(hdr, out_hw=out_hw, log_mode=True)
    return t


# def run_batch(model, input_dir, mask_dir, output_dir, device="cuda", num_steps=1):
#     os.makedirs(output_dir, exist_ok=True)
#     images = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
#     logging.info(f"Found {len(images)} input images in {input_dir}")

    
#     hdrs_folder = "/mnt/bn/pico-idl-avatar2/cz/OLAT/data/OLAT_EnvMaps_HDR_10"
#     hdrs_files = [os.path.join(hdrs_folder, f) for f in os.listdir(hdrs_folder)]

#     for idx, fname in enumerate(images):
#         img_path = os.path.join(input_dir, fname)
#         base = os.path.splitext(fname)[0]
#         mask_path = os.path.join(mask_dir, f"{base}.png")
#         out_path = os.path.join(output_dir, f"{base}.png")
#         # hdr_path = os.path.join(input_dir, f"{base}.hdr")
#         # hdr_path = hdrs_files[idx]
#         # print(f"Processing {fname} with HDR {hdr_path}")

#         src = _open_rgb(img_path)
#         mask = None
#         if os.path.exists(mask_path):
#             mask = load_mask(mask_path, target_size=(src.width, src.height), device=device)
#         else:
#             logging.warning(f"Missing mask for {fname}")

#         hdr_env = None
#         for hdr_idx, hdr_path in enumerate(hdrs_files):
#             if os.path.exists(hdr_path):
#                 hdr_env = _load_hdr_env(hdr_path)
#             else:
#                 logging.warning(f"Missing HDR for {fname}")

#             with torch.no_grad():
#                 out = evaluate_olat(model, src, num_sampling_steps=num_steps, condition=None, mask=mask, hdr_env=hdr_env)
#                 out_path = os.path.join(output_dir, f"{base}_{hdr_idx}.png")
#                 out.save(out_path)

#             logging.info(f"[{idx+1}/{len(images)}] Saved {out_path}")
from torchvision.utils import make_grid, save_image
def run_batch(model, input_dir, mask_dir, output_dir, device="cuda", num_steps=1):
    os.makedirs(output_dir, exist_ok=True)
    images = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    hdrs_folder = "/mnt/bn/pico-idl-avatar2/cz/OLAT/data/OLAT_EnvMaps_HDR_128"
    # hdrs_files = [os.path.join(hdrs_folder, f) for f in os.listdir(hdrs_folder)]
    hdrs_files = sorted([os.path.join(hdrs_folder, f) for f in os.listdir(hdrs_folder)])

    for idx, fname in enumerate(images):
        img_path = os.path.join(input_dir, fname)
        base = os.path.splitext(fname)[0]
        mask_path = os.path.join(mask_dir, f"{base}_alpha.png")

        src = _open_rgb(img_path)
        mask = load_mask(mask_path, target_size=(src.width, src.height), device=device)
        img_t = to_tensor(src).unsqueeze(0).to(device, dtype=model.dtype)
        mask_t = mask.to(device, dtype=model.dtype)

        for hdr_idx, hdr_path in enumerate(hdrs_files):
            hdr_env = _load_hdr_env(hdr_path, out_hw=(src.height, src.width)).unsqueeze(0).to(device, dtype=model.dtype)

            # img_t, pad = pad_to_multiple(img_t, 16)
            # mask_t, _ = pad_to_multiple(mask_t, 16)
            # hdr_env, _ = pad_to_multiple(hdr_env, 16)

            batch = {
                "source_image": img_t,
                "mask": mask_t,
                "hdr_env": hdr_env
            }
            # print("mask:", mask_t.shape)
            # print("hdr_env:", hdr_env.shape)
            # print("target size:", img_t.shape)
            # print("pad:", pad)
            

            # === 模仿训练 log_samples 的处理 ===
            z = model.vae.encode(img_t)
            output = model.sample(
                z=z,
                num_steps=num_steps,
                conditioner_inputs=batch,
                max_samples=1,
            )
            # output = unpad(output, pad)

            # 保存与训练可视化相同的格式
            grid = make_grid(output, nrow=1, normalize=True, value_range=(0, 1))
            hdr_name = os.path.splitext(os.path.basename(hdr_path))[0]  # 提取HDR文件名（无扩展名）
            out_path = os.path.join(output_dir, f"{base}_{hdr_name}_t.png")
            # out_path = os.path.join(output_dir, f"{base}_{hdr_idx:03d}.png")
            save_image(grid, out_path)

            # # out_path = os.path.join(output_dir, f"{base}_{hdr_idx}.png")
            # out_path = os.path.join(output_dir, f"{base}_{hdr_name}_old.png")
            # out = output[0].clamp(-1, 1).float().cpu()
            # out = (out + 1) / 2
            # from torchvision.transforms import ToPILImage, ToTensor
            # to_pil = ToPILImage()
            # out_pil = to_pil(out)
            # out_pil.save(out_path)


            print(f"[{idx+1}/{len(images)}] Saved {out_path}")

def main():
    ap = argparse.ArgumentParser("LBM Relighting Batch Inference")
    ap.add_argument("--input_dir", required=True, help="输入图像文件夹")
    ap.add_argument("--mask_dir", required=True, help="对应 mask 文件夹")
    ap.add_argument("--output_dir", required=True, help="输出文件夹")
    ap.add_argument("--model_name", default="relighting")
    ap.add_argument("--num_inference_steps", type=int, default=1)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16
    # ckpt_dir = os.path.join(PATH, "ckpts", args.model_name)
    ckpt_dir = os.path.join(os.path.dirname(PATH), "ckpts", args.model_name)

    logging.info(f"🧠 Loading model from {ckpt_dir} ...")
    model = get_model(ckpt_dir, torch_dtype=torch_dtype, device=args.device)
    model.eval()
    logging.info("✅ Model loaded successfully!")

    run_batch(model, args.input_dir, args.mask_dir, args.output_dir, device=args.device, num_steps=args.num_inference_steps)

if __name__ == "__main__":
    main()
