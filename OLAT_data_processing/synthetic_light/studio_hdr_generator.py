#!/usr/bin/env python3
"""
Studio-colored HDR environment generator (equirectangular).

- Creates Radiance .hdr maps with colored softboxes/spotlights.
- Fully configurable palettes & direction sets.
- Can render multiple scenes in a loop.

Usage examples:
  python studio_hdr_generator.py --out ./out --preset basic --write-examples
  python studio_hdr_generator.py --out ./out --scene-json scenes.json

If OpenCV HDR writing fails in your environment, the script will save .npy as a fallback.
You can later convert with OpenCV: npy -> HDR.

Angles:
- theta = elevation in degrees [0..180], 0=north pole, 90=horizon, 180=south pole
- phi   = azimuth   in degrees [0..360), 0=+X (center), 90=+Z (left on the map)

Author: ChatGPT
"""
import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np

try:
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

@dataclass
class Light:
    name: str
    theta_deg: float
    phi_deg: float
    size_deg: float
    intensity: float
    color_rgb: Tuple[float, float, float]
    aspect: float = 1.0
    shape: str = "gaussian"  # "gaussian" | "rect"
    softness: float = 1.0

@dataclass
class SceneConfig:
    name: str
    resolution: Tuple[int, int] = (1024, 512)  # (W, H)
    base_level: float = 0.0
    base_tint: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    lights: List[Light] = None

PALETTES: Dict[str, Dict[str, Tuple[float,float,float]]] = {
    "studio_basic": {
        "warm": (1.0, 0.78, 0.6),
        "cool": (0.6, 0.8, 1.0),
        "white": (1.0, 1.0, 1.0),
    },
    "gel_cinema": {
        "teal": (0.55, 0.85, 1.0),
        "amber": (1.0, 0.7, 0.35),
        "magenta": (0.95, 0.4, 0.9),
        "cyan": (0.5, 1.0, 1.0),
    },
    "rgb_primaries": {
        "red": (1.0, 0.25, 0.25),
        "green": (0.3, 1.0, 0.3),
        "blue": (0.3, 0.3, 1.0),
        "white": (1.0, 1.0, 1.0),
    },
}

DIRECTION_SETS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "classic_key_fill_rim": {
        "key":  (60,  30),
        "fill": (80, 150),
        "rim":  (70, 210),
        "top":  (30,   0),
    },
    "three_point": {
        "key":  (60,  45),
        "fill": (85, 135),
        "rim":  (75, 225),
    },
    "cross_rim": {
        "rim_r": (75, 330),
        "rim_l": (75, 210),
        "hair":  (35,   0),
    },
}

def spherical_distance(theta1, phi1, theta2, phi2):
    dphi = phi2 - phi1
    dtheta = theta2 - theta1
    a = np.sin(dtheta/2.0)**2 + np.cos(theta1)*np.cos(theta2)*np.sin(dphi/2.0)**2
    return 2.0 * np.arcsin(np.sqrt(np.maximum(a, 0.0)))

def render_scene(cfg: SceneConfig) -> np.ndarray:
    W, H = cfg.resolution
    u = (np.arange(W) + 0.5) / W
    v = (np.arange(H) + 0.5) / H
    uu, vv = np.meshgrid(u, v, indexing="xy")
    theta = vv * np.pi
    phi = uu * 2.0 * np.pi

    img = np.ones((H, W, 3), dtype=np.float32)
    img *= np.array(cfg.base_tint, dtype=np.float32).reshape(1,1,3)
    img *= float(cfg.base_level)

    if not cfg.lights:
        return img

    for L in cfg.lights:
        theta_l = np.deg2rad(L.theta_deg)
        phi_l   = np.deg2rad(L.phi_deg)
        ang = spherical_distance(theta, phi, theta_l, phi_l)

        size_rad = np.deg2rad(max(L.size_deg, 1e-3))
        soft = max(L.softness, 1e-3)

        if L.shape == "gaussian":
            std = size_rad * soft
            weight = np.exp(-0.5 * (ang / std)**2)
        else:
            dtheta = np.abs(theta - theta_l)
            dphi   = np.abs(np.mod(phi - phi_l + np.pi, 2*np.pi) - np.pi)
            half_h = size_rad / max(L.aspect, 1e-6)
            half_w = size_rad * max(L.aspect, 1e-6)
            def smooth_edge(x, edge):
                t = np.clip(1.0 - x / (edge * soft), 0.0, 1.0)
                return t * t * (3.0 - 2.0 * t)
            wv = smooth_edge(dtheta, half_h)
            wu = smooth_edge(dphi,   half_w)
            weight = wv * wu

        contrib = (np.array(L.color_rgb, dtype=np.float32).reshape(1,1,3) * L.intensity) * weight[..., None]
        img += contrib.astype(np.float32)

    return img

def save_hdr(path: str, img: np.ndarray) -> Tuple[bool, str]:
    # Try imageio HDR first
    if HAS_IMAGEIO and path.lower().endswith(".hdr"):
        try:
            iio.imwrite(path, np.maximum(img, 0.0).astype(np.float32), plugin="HDR-FI")
            return True, path
        except Exception:
            pass
    # Try OpenCV HDR
    if HAS_CV2 and path.lower().endswith(".hdr"):
        bgr = img[..., ::-1].astype(np.float32)
        if cv2.imwrite(path, bgr):
            return True, path
    # Fallback NPY
    npy_path = os.path.splitext(path)[0] + ".npy"
    np.save(npy_path, img.astype(np.float32))
    return False, npy_path

def build_scene_from_recipe(name: str, base_level: float, base_tint, res, recipe: List[Dict]) -> SceneConfig:
    lights = []
    for idx, item in enumerate(recipe):
        dir_key = item.get("dir_key")
        dir_set = item.get("dir_set", "three_point")
        theta_phi = item.get("theta_deg_phi_deg")
        if theta_phi is None and dir_key and dir_set in DIRECTION_SETS and dir_key in DIRECTION_SETS[dir_set]:
            theta_phi = DIRECTION_SETS[dir_set][dir_key]
        if theta_phi is None:
            theta_phi = (70, 0)
        color_rgb = item.get("color_rgb")
        if color_rgb is None and "color_key" in item:
            palette = PALETTES.get(item.get("palette", "studio_basic"), {})
            color_rgb = palette.get(item["color_key"], (1.0, 1.0, 1.0))

        lights.append(
            Light(
                name=item.get("label", f"light_{idx}"),
                theta_deg=float(theta_phi[0]),
                phi_deg=float(theta_phi[1]),
                size_deg=float(item.get("size_deg", 18.0)),
                intensity=float(item.get("intensity", 4.0)),
                color_rgb=tuple(color_rgb),
                aspect=float(item.get("aspect", 1.5)),
                shape=item.get("shape", "gaussian"),
                softness=float(item.get("softness", 1.0)),
            )
        )
    return SceneConfig(
        name=name,
        resolution=tuple(res),
        base_level=float(base_level),
        base_tint=tuple(base_tint),
        lights=lights,
    )

EXAMPLE_SCENES = [
    {
        "name": "warm_key_cool_fill_rim_white",
        "base_level": 0.05,
        "base_tint": [1.0, 1.0, 1.0],
        "res": [1024, 512],
        "recipe": [
            {"label":"key",  "dir_set":"classic_key_fill_rim", "dir_key":"key",  "palette":"studio_basic", "color_key":"warm", "size_deg":22, "intensity":6.0, "shape":"gaussian", "softness":1.1},
            {"label":"fill", "dir_set":"classic_key_fill_rim", "dir_key":"fill", "palette":"studio_basic", "color_key":"cool", "size_deg":24, "intensity":2.5, "shape":"gaussian", "softness":1.3},
            {"label":"rim",  "dir_set":"classic_key_fill_rim", "dir_key":"rim",  "palette":"studio_basic", "color_key":"white","size_deg":18, "intensity":3.0, "shape":"gaussian", "softness":0.9},
        ]
    },
    {
        "name": "teal_orange_plus_magenta_hair",
        "base_level": 0.03,
        "base_tint": [1.0, 1.0, 1.0],
        "res": [2048, 1024],
        "recipe": [
            {"label":"key",   "dir_set":"three_point", "dir_key":"key",  "palette":"gel_cinema", "color_key":"amber",  "size_deg":20, "intensity":6.0, "shape":"gaussian"},
            {"label":"fill",  "dir_set":"three_point", "dir_key":"fill", "palette":"gel_cinema", "color_key":"teal",   "size_deg":26, "intensity":3.0, "shape":"gaussian"},
            {"label":"rim",   "dir_set":"three_point", "dir_key":"rim",  "palette":"gel_cinema", "color_key":"magenta","size_deg":16, "intensity":4.0, "shape":"gaussian"},
            {"label":"hair",  "dir_set":"classic_key_fill_rim", "dir_key":"top", "palette":"rgb_primaries","color_key":"white","size_deg":12, "intensity":2.0, "shape":"rect", "aspect":2.5, "softness":1.2},
        ]
    },
    {
        "name": "cross_rims_rgb",
        "base_level": 0.02,
        "base_tint": [1.0, 1.0, 1.0],
        "res": [1024, 512],
        "recipe": [
            {"label":"rim_r","dir_set":"cross_rim","dir_key":"rim_r","palette":"rgb_primaries","color_key":"red",   "size_deg":18, "intensity":5.0},
            {"label":"rim_l","dir_set":"cross_rim","dir_key":"rim_l","palette":"rgb_primaries","color_key":"blue",  "size_deg":18, "intensity":5.0},
            {"label":"hair", "dir_set":"cross_rim","dir_key":"hair", "palette":"rgb_primaries","color_key":"green", "size_deg":14, "intensity":2.5, "shape":"rect", "aspect":3.0},
        ]
    }
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="./studio_hdr_out", help="Output folder")
    ap.add_argument("--scene-json", type=str, default="", help="Path to a JSON file with a list of scenes (same structure as EXAMPLE_SCENES)")
    ap.add_argument("--writeexamples", action="store_true", help="Write built-in example scenes")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    scenes = []
    if args.writeexamples or not args.scene_json:
        scenes.extend(EXAMPLE_SCENES)

    if args.scene_json:
        with open(args.scene_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, list), "scene-json should contain a list of scene dicts"
            scenes.extend(data)

    written = []
    for spec in scenes:
        cfg = build_scene_from_recipe(
            name=spec["name"],
            base_level=spec.get("base_level", 0.02),
            base_tint=tuple(spec.get("base_tint", (1.0, 1.0, 1.0))),
            res=tuple(spec.get("res", (1024, 512))),
            recipe=spec.get("recipe", []),
        )
        img = render_scene(cfg)
        out_hdr = os.path.join(args.out, f"{cfg.name}.hdr")
        ok, path = save_hdr(out_hdr, img)
        written.append((cfg.name, ok, path, cfg.resolution, len(cfg.lights) if cfg.lights else 0))

    print("WROTE:")
    for item in written:
        print(item)

if __name__ == "__main__":
    main()
