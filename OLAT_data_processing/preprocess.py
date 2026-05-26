#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read OLAT .parquet -> decode JPEG in-memory -> process -> save processed images.
Operations:
  1) Gamma correction
  2) Contrast enhancement (CLAHE or linear)
  3) Rotate 90° (per-camera: default clockwise; specific cameras counterclockwise)
  4) Resize to quarter
Only handles OLAT parquet (skips PBR).

CHANGE (2025-08-20):
- For cameras in ROTATE_LEFT_CAMS, rotate 90° counterclockwise (left). Others remain clockwise (right).
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import pyarrow.parquet as pq
import pandas as pd

# ---- Config: cameras that should rotate LEFT (90° CCW) ----
ROTATE_LEFT_CAMS = {"C03", "C04", "C05", "C06", "C07", "C10", "C14", "C15", "C21", "C22", "C23"}

# ---------- Image ops ----------
def split_rgb_a(img):
    if img.ndim == 2:
        return img, None
    if img.shape[2] == 4:
        return img[:, :, :3], img[:, :, 3]
    return img, None

def merge_rgb_a(rgb, alpha):
    if alpha is None:
        return rgb
    return np.dstack([rgb, alpha])

def apply_gamma(img, gamma: float):
    if gamma is None or abs(gamma - 1.0) < 1e-8:
        return img
    rgb, a = split_rgb_a(img)
    rgb_f = rgb.astype(np.float32)
    is_uint8 = (rgb.dtype == np.uint8)
    scale = 255.0 if is_uint8 else (rgb_f.max() if rgb_f.max() > 1.0 else 1.0)
    if scale == 0:
        scale = 1.0
    rgb_n = np.clip(rgb_f / scale, 0.0, 1.0)
    rgb_g = np.power(rgb_n, 1.0 / max(gamma, 1e-8))
    rgb_out = np.clip(rgb_g * scale, 0, 65535).astype(rgb.dtype)
    return merge_rgb_a(rgb_out, a)

def apply_contrast_clahe(img, clip_limit=2.0, tile_grid=(8, 8)):
    rgb, a = split_rgb_a(img)
    if rgb.ndim == 2 or rgb.shape[2] == 1:
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid))
        out = clahe.apply(rgb if rgb.ndim == 2 else rgb[:, :, 0])
        return merge_rgb_a(out if rgb.ndim == 2 else out[:, :, None], a)
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return merge_rgb_a(out, a)

def apply_contrast_linear(img, alpha=1.2, beta=0.0):
    rgb, a = split_rgb_a(img)
    out = cv2.convertScaleAbs(rgb, alpha=float(alpha), beta=float(beta))
    return merge_rgb_a(out, a)

def rotate_right_90(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def rotate_left_90(img):
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

def resize_quarter(img):
    return cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA)
    # return cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)

def process_one(img, args, rotate_fn):
    out = apply_gamma(img, args.gamma)
    if args.contrast == "clahe":
        out = apply_contrast_clahe(out, clip_limit=args.clip_limit, tile_grid=(args.tile, args.tile))
    else:
        out = apply_contrast_linear(out, alpha=args.alpha, beta=args.beta)
    out = rotate_fn(out)
    out = resize_quarter(out)
    return out

# ---------- Parquet helpers ----------
def is_olat_parquet(df: pd.DataFrame) -> bool:
    if "info" not in df.columns:
        return False
    try:
        info = df["info"].values[0]
    except Exception:
        return False
    return isinstance(info, str) and ("OLAT" in info or "Photos" in info)

def load_parquet_as_row(parquet_path: Path) -> pd.DataFrame:
    pf = pq.ParquetFile(str(parquet_path))
    cols = pf.schema.names.copy()
    if "__index_level_0__" in cols:
        cols.remove("__index_level_0__")
    table = pf.read(cols, use_pandas_metadata=True)
    df = table.to_pandas()
    return df

def iter_olat_images_from_df(df: pd.DataFrame):
    """
    Yields (frame_id:str, bytes:np.ndarray_of_uint8) for OLAT parquet.
    """
    cols = list(df.columns)
    if "info" in cols:
        cols.remove("info")
    skip_cols = {"img_list"}
    cols = [c for c in cols if c not in skip_cols]
    row = df.iloc[0]
    for key in cols:
        binval = row[key]
        if binval is None:
            continue
        yield key, np.frombuffer(binval, dtype=np.uint8)

def extract_cam_from_info(info: str) -> str:
    """
    info 示例: 'OLAT_xxx/Photos/C01' 或含类似路径的字符串。
    返回末级目录名，如 'C01'；若无法解析则返回空字符串。
    """
    # 统一分隔符并取最后一段
    name = Path(info.replace("\\", "/")).name
    return name if name.startswith("C") else ""

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Inline process OLAT parquet: decode -> process -> save processed images only"
    )
    ap.add_argument("-i", "--input", required=True,
                    help="Input root that contains OLAT parquet files (will recurse)")
    ap.add_argument("-o", "--output", required=True, help="Output root for processed images")
    ap.add_argument("-r", "--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite if file exists")

    # processing params
    ap.add_argument("--gamma", type=float, default=1.0, help="Gamma (>1 brighten mids; <1 darken)")
    ap.add_argument("--contrast", choices=["clahe", "linear"], default="clahe", help="Contrast method")
    ap.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE clip limit")
    ap.add_argument("--tile", type=int, default=8, help="CLAHE tile grid size (tile x tile)")
    ap.add_argument("--alpha", type=float, default=1.2, help="Linear contrast alpha (gain) when --contrast linear")
    ap.add_argument("--beta", type=float, default=0.0, help="Linear contrast beta (bias) when --contrast linear")

    # path shaping
    ap.add_argument("--keep-structure", action="store_true",
                    help="Preserve parquet's 'info' as subdir under output")
    ap.add_argument("--suffix", default="_proc", help="Output file suffix before extension (default _proc)")
    args = ap.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)
    if not in_root.exists():
        print(f"[Error] Input root not found: {in_root}", file=sys.stderr)
        sys.exit(1)
    out_root.mkdir(parents=True, exist_ok=True)

    # gather parquet files
    parquets = sorted(in_root.rglob("*.parquet") if args.recursive else in_root.glob("*.parquet"))

    if not parquets:
        print("[Info] No parquet files found.")
        return

    pbar = tqdm(parquets, desc="Parquet files")
    for pq_path in pbar:
        try:
            df = load_parquet_as_row(pq_path)
            if not is_olat_parquet(df):
                continue

            info = df["info"].values[0]  # e.g. "OLAT_xxx/Photos/C01"
            cam = extract_cam_from_info(info)  # "C01" ...
            rotate_fn = rotate_left_90 if cam in ROTATE_LEFT_CAMS else rotate_right_90

            out_dir = out_root / info if args.keep_structure else out_root / pq_path.stem
            out_dir.mkdir(parents=True, exist_ok=True)

            for frame_id, jpg_bytes in iter_olat_images_from_df(df):
                img = cv2.imdecode(jpg_bytes, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue

                out_img = process_one(img, args, rotate_fn)

                dst = out_dir / f"{frame_id}{args.suffix}.jpg"
                if (not args.overwrite) and dst.exists():
                    continue
                ok = cv2.imwrite(str(dst), out_img)
                if not ok:
                    print(f"[Warn] imwrite failed: {dst}", file=sys.stderr)

        except Exception as e:
            print(f"[Warn] Failed on {pq_path}: {e}", file=sys.stderr)

    print(f"[Done] Processed images saved under: {out_root}")

if __name__ == "__main__":
    main()
