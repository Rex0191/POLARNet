#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
studio_hdr_generator_full.py
一个完整、可独立运行的 HDR 生成器：支持四种形状
- "gaussian"    : 高斯光斑（角半径 size_deg，softness 影响边缘缓和）
- "rect"        : 条带（**修正**：size_deg 表示半厚度，aspect 只在横向拉伸 -> 不会再黑屏）
- "greatcircle" : 对角大圆色带（tilt_deg 控制方向，size_deg 控制半宽，softness 控制羽化）
- "sector"      : 扇形/楔形（phi_span_deg、theta_inner_deg、theta_outer_deg、softness）

输入：一个 JSON（见同目录示例），每个场景包含 base_level/base_tint/res/recipe。
输出：每个场景生成 .hdr 与 .png 预览（可通过 --no-preview 关闭）。

用法：
  python studio_hdr_generator_full.py --scene-json ./fans_diag_crisp_400.json --out ./out_hdrs --no-preview
"""
import argparse, json, math, os
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import imageio.v2 as imageio

# ---------------- utils ----------------
def deg2rad(x): return x * math.pi / 180.0
def sph_to_xyz(theta, phi):
    """支持 theta, phi 任意广播形状，输出 (...,3)。"""
    theta = np.broadcast_to(theta, np.broadcast(theta, phi).shape)
    phi   = np.broadcast_to(phi, theta.shape)
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    return np.stack([st*cp, st*sp, ct], axis=-1)

def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# equirectangular grid
def make_grid(H, W):
    # theta: 0..pi (0 at north pole), phi: 0..2pi
    j = np.linspace(0.5, H-0.5, H, dtype=np.float32) / H
    i = np.linspace(0.5, W-0.5, W, dtype=np.float32) / W
    theta = j[:,None] * math.pi
    phi   = i[None,:] * 2.0 * math.pi
    return theta, phi

def angular_distance(theta, phi, theta0, phi0):
    # great-circle angular distance between two directions
    # cos d = sin t sin t0 cos(phi-phi0) + cos t cos t0
    dt = theta - theta0
    dphi = phi - phi0
    # wrap-aware on phi: use cos difference
    cosd = np.sin(theta)*np.sin(theta0)*np.cos(dphi) + np.cos(theta)*np.cos(theta0)
    cosd = np.clip(cosd, -1.0, 1.0)
    return np.arccos(cosd)

# ---------------- rasterizers ----------------
def raster_gaussian(theta, phi, theta0, phi0, size_deg, softness):
    size = deg2rad(size_deg)
    sigma = max(1e-6, size * max(0.4, softness))  # softness 越小越锐
    d = angular_distance(theta, phi, theta0, phi0)
    w = np.exp(-0.5 * (d/sigma)**2).astype(np.float32)
    return w

def raster_rect(theta, phi, theta0, phi0, size_deg, softness, aspect=200.0):
    """
    修正后的条带：
    - size_deg = 竖直方向“半厚度”（总厚度≈2*size_deg）
    - aspect   = 横向拉伸比例（越大越接近整条带）
    """
    size = deg2rad(size_deg)
    # 将局部坐标绕 (theta0, phi0) 建一个切平面坐标系 (u:经向, v:纬向)
    # 简化：使用经度差与纬度差近似条带范围（在常用厚度下足够）
    # 纬向：|theta - theta0| <= size with soft edge
    v = np.abs(theta - theta0)
    v_mask = 1.0 - smoothstep(size, size*(1.0+softness), v)

    # 经向：拉伸到宽：半宽 = size * aspect
    # wrap-aware shortest difference for phi
    dphi = np.angle(np.exp(1j*(phi - phi0)))
    half_w = size * max(aspect, 1e-6)
    u = np.abs(dphi)
    u_mask = 1.0 - smoothstep(half_w, half_w*(1.0+softness), u)

    return (v_mask * u_mask).astype(np.float32)

def raster_greatcircle(theta, phi, theta0, phi0, size_deg, softness, tilt_deg=45.0):
    """
    更直观的实现：
    在 equirectangular 空间中近似大圆带。
    tilt_deg=45° 表示从左下到右上。
    """
    # 将 tilt_deg 转为弧度
    tilt = np.deg2rad(tilt_deg)
    # 把 phi 映射到 [-π, π]
    dphi = np.angle(np.exp(1j*(phi - phi0)))
    # 构造“对角轴”：theta 和 phi 的线性组合
    diag = np.cos(tilt) * (theta - theta0) + np.sin(tilt) * dphi
    # 按半宽构建掩码
    halfw = np.deg2rad(size_deg)
    edge = np.abs(diag)
    mask = 1.0 - smoothstep(halfw, halfw*(1.0+softness), edge)
    return mask.astype(np.float32)


def raster_sector(theta, phi, theta0, phi0, phi_span_deg, theta_inner_deg, theta_outer_deg, softness):
    # phi window around center
    dphi = np.angle(np.exp(1j*(phi - phi0)))
    half_open = deg2rad(max(0.0, phi_span_deg*0.5))
    phi_soft = smoothstep(half_open, half_open*(1.0+softness), np.abs(dphi))

    th_in  = deg2rad(theta_inner_deg)
    th_out = deg2rad(theta_outer_deg)
    th_soft_in  = smoothstep(th_in*(1.0-softness), th_in, theta)  # fade-in from inner
    th_soft_out = 1.0 - smoothstep(th_out, th_out*(1.0+softness), theta)  # fade-out to outer

    return ((1.0 - phi_soft) * th_soft_in * th_soft_out).astype(np.float32)

# --------------- color palettes ---------------
PALETTES = {
    "rgb_primaries": {
        "red":   [1.0, 0.0, 0.0],
        "green": [0.0, 1.0, 0.0],
        "blue":  [0.0, 0.0, 1.0],
        "white": [1.0, 1.0, 1.0],
    },
    "gel_cinema": {
        "amber":   [1.0, 0.6, 0.1],
        "teal":    [0.0, 0.8, 0.75],
        "magenta": [0.95, 0.0, 0.7],
        "cyan":    [0.0, 1.0, 1.0],
    },
    "studio_basic": {
        "warm": [1.0, 0.8, 0.6],
        "cool": [0.6, 0.8, 1.0],
    }
}

def color_rgb(palette, key):
    if palette not in PALETTES or key not in PALETTES[palette]:
        raise KeyError(f"Unknown color {palette}.{key}")
    return np.array(PALETTES[palette][key], dtype=np.float32)

# --------------- rendering ---------------
def render_scene(scene: dict):
    W, H = scene.get("res", [2048,1024])
    H = int(H); W = int(W)
    theta, phi = make_grid(H, W)

    base_level = float(scene.get("base_level", 0.0))
    base_tint  = np.array(scene.get("base_tint", [1.0,1.0,1.0]), dtype=np.float32)

    img = np.ones((H,W,3), dtype=np.float32) * (base_level * base_tint[None,None,:])

    for L in scene["recipe"]:
        th = deg2rad(float(L["theta_deg_phi_deg"][0]))
        ph = deg2rad(float(L["theta_deg_phi_deg"][1]))
        intensity = float(L.get("intensity", 1.0))
        softness  = float(L.get("softness", 0.5))
        shape     = L.get("shape", "gaussian")
        c         = color_rgb(L.get("palette"), L.get("color_key"))

        if shape == "gaussian":
            size_deg = float(L.get("size_deg", 10.0))
            mask = raster_gaussian(theta, phi, th, ph, size_deg, softness)
        elif shape == "rect":
            size_deg = float(L.get("size_deg", 10.0))
            aspect   = float(L.get("aspect", 200.0))
            mask = raster_rect(theta, phi, th, ph, size_deg, softness, aspect)
        elif shape == "greatcircle":
            size_deg = float(L.get("size_deg", 12.0))
            tilt_deg = float(L.get("tilt_deg", 45.0))
            mask = raster_greatcircle(theta, phi, th, ph, size_deg, softness, tilt_deg)
        elif shape == "sector":
            phi_span = float(L.get("phi_span_deg", 120.0))
            th_in    = float(L.get("theta_inner_deg", 20.0))
            th_out   = float(L.get("theta_outer_deg", 140.0))
            mask = raster_sector(theta, phi, th, ph, phi_span, th_in, th_out, softness)
        else:
            raise ValueError(f"Unknown shape: {shape}")

        img += (intensity * mask[...,None] * c[None,None,:]).astype(np.float32)

    return img  # linear HDR

def save_hdr(path, img):
    imageio.imwrite(path, img.astype(np.float32), format='HDR')  # Radiance .hdr

def tonemap(x):
    # simple Reinhard for preview
    return x / (1.0 + x)

def save_preview(path, img):
    sdr = tonemap(np.clip(img, 0, None))
    sdr = np.power(sdr, 1/2.2)  # gamma for display
    sdr8 = np.clip(sdr*255.0, 0, 255).astype(np.uint8)
    imageio.imwrite(path, sdr8)

# --------------- CLI ---------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-json", required=True, help="输入 JSON（多个场景）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--no-preview", action="store_true", help="不保存 PNG 预览")
    args = ap.parse_args()

    scenes = json.load(open(args.scene_json, "r", encoding="utf-8"))
    os.makedirs(args.out, exist_ok=True)

    for scene in scenes:
        img = render_scene(scene)
        name = scene.get("name", "scene")
        hdr_path = os.path.join(args.out, f"{name}.hdr")
        save_hdr(hdr_path, img)
        if not args.no_preview:
            png_path = os.path.join(args.out, f"{name}.png")
            save_preview(png_path, img)

    print(f"[OK] Wrote {len(scenes)} scenes -> {args.out}")

if __name__ == "__main__":
    main()
