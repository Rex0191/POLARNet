#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RelightPairsDataset
从你的 folder 结构中构造 (source -> target) 配对，并返回:
  - image:   source composite (RGB, float32[0..1], CHW)
  - target:  target composite (RGB, float32[0..1], CHW)
  - hdr_env: target 对应 HDR 环境图 (RGB, float32，线性域，可已缩小)
  - mask:    前景 alpha（同一个 Cxx 下十张共用），float32[0..1], CHW(1*H*W)

目录与命名约定（你已确认）：
- composite 根: /mnt/bn/idl-data-cache/cz/data/OLAT/synthetic_light/<sid>/<session>/<camera>/*.png
  文件名形如: <env_key>_composite.png  （env_key 与 HDR 同名）
- HDR 根:      /mnt/bn/pico-idl-avatar2/cz/OLAT/data/hdrs_all/<env_key>.hdr
- mask 根:     /mnt/bn/idl-data-cache/cz/data/OLAT/alpha_data/<session>/<camera>/<session>_<camera>_uniform_alpha.png
  （注意：mask 路径没有 <sid> 这一层）

采样策略：
- mode='all_pairs'       : 同一 Cxx 文件夹内，生成所有有序配对 L*(L-1)
- mode='random_per_epoch': 每次 __getitem__ 基于索引确定所属 Cxx 组，然后随机采 1 对（轻量/增广）
- 你也可以限制每组最大采样数量 max_pairs_per_group 用于缩小数据量

@author: you
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# ----------------------------- 工具函数 -----------------------------

def _imread_uint8(path: Path) -> np.ndarray:
    """健壮读取（支持中文/长路径），返回 uint8（BGR 或 GRAY）"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Failed to read image: {path}")
    return img

def _to_rgb_float_chw_u8(img_u8: np.ndarray) -> torch.Tensor:
    """uint8 BGR/GRAY -> float32 RGB CHW, 0..1"""
    if img_u8.ndim == 2:
        img_u8 = np.stack([img_u8, img_u8, img_u8], axis=-1)
    elif img_u8.shape[2] >= 4:
        img_u8 = img_u8[:, :, :3]  # 丢弃 alpha
    rgb = img_u8[:, :, ::-1].astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    return t

def _to_gray_float_chw_u8(img_u8: np.ndarray) -> torch.Tensor:
    """uint8 任意 -> float32 GRAY CHW(1,H,W), 0..1"""
    if img_u8.ndim == 3:
        if img_u8.shape[2] == 4:
            # 优先 alpha 通道
            gray = img_u8[:, :, 3]
        else:
            gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_u8
    gray = gray.astype(np.float32) / 255.0
    return torch.from_numpy(gray)[None, ...].contiguous()

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

def _first_image_in(dir_path: Path):
        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
        files = []
        for ext in exts:
            files.extend(dir_path.glob(ext))
        files = sorted(files)
        if not files:
            raise FileNotFoundError(f"No alpha image found in: {dir_path}")
        return files[0]
# ----------------------------- 数据集主体 -----------------------------

class RelightPairsDataset(Dataset):
    """
    遍历 composite 根目录下的 <sid>/<session>/<camera>/，在每个 camera 目录内：
      - 收集所有 *_composite.png
      - 同文件夹内两两成对（或随机配对）
      - 为 target 找到对应 HDR：hdrs_all/<env_key>.hdr
      - 为该 camera 找到公共 mask：alpha_data/<session>/<camera>/<session>_<camera>_uniform_alpha.png
    返回 dict: {"image", "target", "hdr_env", "mask", "_meta": {...}}
    """
    def __init__(
        self,
        composite_root: str,
        hdr_root: str,
        alpha_root: str,
        mode: str = "all_pairs",   # "all_pairs" | "random_per_epoch"
        hdr_out_hw: Tuple[int,int] = (64, 128),
        resize_hw: Optional[Tuple[int,int]] = None,  # 若需要统一分辨率 (H,W)
        max_pairs_per_group: Optional[int] = None,   # 仅在 all_pairs 下生效
        seed: int = 123
    ):
        self.composite_root = Path(composite_root)
        self.hdr_root = Path(hdr_root)
        self.alpha_root = Path(alpha_root)
        self.mode = mode
        assert self.mode in ("all_pairs", "random_per_epoch")
        self.hdr_out_hw = hdr_out_hw
        self.resize_hw = resize_hw
        self.max_pairs_per_group = max_pairs_per_group
        random.seed(seed)

        # 扫描所有 camera 目录，建立每组的样本清单
        # 组键：(<sid>, <session>, <camera>)；组内成员：[(env_key, composite_path), ...]
        self.groups: Dict[Tuple[str,str,str], List[Tuple[str, Path]]] = {}
        for sid_dir in sorted(self.composite_root.iterdir()):
            if not sid_dir.is_dir():
                continue
            try:
                sid_num = int(sid_dir.name)
                if 50 <= sid_num <= 96:
                    continue
            except ValueError:
                # sid_dir.name 不是纯数字就忽略这个检查
                pass
            for session_dir in sorted(sid_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                if session_dir.name.endswith("21_00") or session_dir.name.endswith("22_00"):
                    continue
                for cam_dir in sorted(session_dir.iterdir()):
                    if not cam_dir.is_dir():
                        continue
                    # 收集该 camera 下所有 *_composite.png
                    members = []
                    for p in sorted(cam_dir.glob("*_composite.png")):
                        name = p.name
                        if not name.endswith("_composite.png"):
                            continue
                        env_key = name[:-len("_composite.png")]
                        members.append((env_key, p))
                    if len(members) >= 2:
                        key = (sid_dir.name, session_dir.name, cam_dir.name)
                        self.groups[key] = members

        self.group_keys = list(self.groups.keys())
        self.group_sizes = {k: len(v) for k, v in self.groups.items()}

        # 生成索引
        if self.mode == "all_pairs":
            self.index: List[Tuple[Tuple[str,str,str], int, int]] = []
            for gk, members in self.groups.items():
                L = len(members)
                pairs = [(gk, i, j) for i in range(L) for j in range(L) if i != j]
                if self.max_pairs_per_group is not None and len(pairs) > self.max_pairs_per_group:
                    pairs = random.sample(pairs, self.max_pairs_per_group)
                self.index.extend(pairs)
        else:
            # random_per_epoch：索引只存 group，__getitem__ 时随机配
            self.index = []
            for gk, members in self.groups.items():
                # 用成员个数占位，保证 len(dataset) 与 all_pairs 同量级（或按需求缩放）
                L = len(members)
                count = L * (L - 1)
                self.index.extend([(gk, -1, -1)] * count)

    def __len__(self) -> int:
        return len(self.index)

    def _mask_path_for_group(self, session: str, camera: str) -> Path:
        # /alpha_data/<session>/<camera>/<session>_<camera>_uniform_alpha.png
        fn = f"{session}_{camera}_uniform_alpha.png"
        return self.alpha_root / session / camera / fn

    def _hdr_path_for_env(self, env_key: str) -> Path:
        # hdrs_all/<env_key>.hdr
        return self.hdr_root / f"{env_key}.hdr"

    def _load_composite(self, p: Path) -> torch.Tensor:
        img_u8 = _imread_uint8(p)
        if self.resize_hw is not None:
            H, W = self.resize_hw
            img_u8 = cv2.resize(img_u8, (W, H), interpolation=cv2.INTER_AREA)
        return _to_rgb_float_chw_u8(img_u8)

    # def _load_mask(self, session: str, camera: str, target_hw: Optional[Tuple[int,int]]) -> torch.Tensor:
    #     mpath = self._mask_path_for_group(session, camera)
    #     m_u8 = _imread_uint8(mpath)
    #     if target_hw is not None:
    #         H, W = target_hw
    #         m_u8 = cv2.resize(m_u8, (W, H), interpolation=cv2.INTER_NEAREST)
    #     return _to_gray_float_chw_u8(m_u8)
    
    def _load_mask(self, sid: str, session: str, camera: str, target_hw: Optional[Tuple[int,int]]) -> torch.Tensor:
        alpha_dir = self.alpha_root / sid / session / camera
        mpath = _first_image_in(alpha_dir)

        # 复用你原有的读图/转灰度函数
        m_u8 = _imread_uint8(mpath)
        if target_hw is not None:
            H, W = target_hw
            m_u8 = cv2.resize(m_u8, (W, H), interpolation=cv2.INTER_NEAREST)
        return _to_gray_float_chw_u8(m_u8)

    def _load_hdr_env(self, env_key: str) -> torch.Tensor:
        hpath = self._hdr_path_for_env(env_key)
        hdr = _imread_hdr_float(hpath)
        t = _hdr_to_rgb_float_chw(hdr, out_hw=self.hdr_out_hw, log_mode=True)
        return t

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gk, i, j = self.index[idx]
        sid, session, camera = gk
        members = self.groups[gk]

        if self.mode == "random_per_epoch":
            L = len(members)
            i, j = random.sample(range(L), 2)

        # (env_i, path_i) = members[i]
        (env_j, path_j) = members[j]

        # source/target composite
        # uniform_path = Path(f"/mnt/bn/idl-data-cache/cz/data/OLAT/nvidia_uniform_light/{sid}/{session}/{camera}/uniform_light_1k_composite.png")
        uniform_path = Path(f"/mnt/bn/idl-data-cache/cz/data/OLAT_final/uniform/{sid}/{session}/{camera}/uniform_light_1k_composite.png")
        
        tgt = self._load_composite(path_j)

        # 对应 mask（同组共用）
        target_hw = (tgt.shape[1], tgt.shape[2]) if self.resize_hw is None else self.resize_hw
        # mask = self._load_mask(session, camera, target_hw=target_hw)
        mask = self._load_mask(sid, session, camera, target_hw=target_hw)

        # if uniform_path.exists():
        src_uniform = self._load_composite(uniform_path)
        src = src_uniform * mask
        tgt = tgt * mask
        # residual = src - tgt
        # else:
        #     print(f"[WARN] Uniform light not found: {uniform_path}, fallback to normal src")
        #     (env_i, path_i) = members[i]
        #     src = self._load_composite(path_i)

        # target 的 HDR 作为 condition
        hdr_env = self._load_hdr_env(env_j)

        return {
            "image": src,          # source composite
            "target": tgt,         # target composite
            "hdr_env": hdr_env,    # condition (target env)
            "mask": mask,          # 前景 alpha（同组共用）
            "uniform_fg": src_uniform,
            "_meta": {
                "sid": sid, "session": session, "camera": camera,
                "src_path": str(uniform_path), "tgt_path": str(path_j),
                "src_env": "uniform", "tgt_env": env_j,
            }
        }

# ----------------------------- 简单 collate -----------------------------

def relight_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    # 假设每个样本的 keys 一致
    out = {}
    out["image"]   = torch.stack([b["image"]   for b in batch], dim=0)
    out["target"]  = torch.stack([b["target"]  for b in batch], dim=0)
    out["hdr_env"] = torch.stack([b["hdr_env"] for b in batch], dim=0)
    out["mask"]    = torch.stack([b["mask"]    for b in batch], dim=0)
    out["uniform_fg"]    = torch.stack([b["uniform_fg"]    for b in batch], dim=0)
    out["_meta"]   = [b["_meta"] for b in batch]
    return out
