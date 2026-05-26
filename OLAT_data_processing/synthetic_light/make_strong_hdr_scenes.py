#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_strong_hdr_scenes.py
生成“强对向/多方向、硬光、强彩色”的 HDR 场景 JSON，供 studio_hdr_generator.py 使用。

特征：
- 小尺寸（缺省 5~10°）、高强度（缺省 8~16）、无底光（base_level=0.0）、softness≈0.6。
- 多种 pattern：2/3/4/5 个方向；左右、上下、对角、环绕均可。
- 颜色池默认不含白色（可设置少量 white 比例）。

用法：
  python make_strong_hdr_scenes.py --out /path/to/scenes.json --count 400
  # 更多参数详见 argparse 帮助

生成的 JSON 可直接喂给：
  python studio_hdr_generator.py --out ./studio_hdr_out --scene-json /path/to/scenes.json
"""
import argparse, json, random, math

# 颜色池（key -> palette）
COLOR_POOL = {
    "amber": "gel_cinema",
    "teal": "gel_cinema",
    "magenta": "gel_cinema",
    "cyan": "gel_cinema",
    "red": "rgb_primaries",
    "green": "rgb_primaries",
    "blue": "rgb_primaries",
    "warm": "studio_basic",
    "cool": "studio_basic",
}
WHITE = ("white", "rgb_primaries")

def pick_colors(n, allow_white_ratio=0.05):
    pool = list(COLOR_POOL.keys())
    cols = []
    for _ in range(n):
        if random.random() < allow_white_ratio:
            cols.append("white")
        else:
            cols.append(random.choice(pool))
    # 尽量去重；若全重复，后续仍然允许
    return cols

def color_to_palette(color_key):
    if color_key == "white":
        return WHITE[1]
    return COLOR_POOL[color_key]

# 预设方向 pattern（角度单位：度）
def pattern_two_lr(theta=80, jitter=8):
    # 左右对打：phi≈90° & 270°
    left_phi  = 90 + random.uniform(-jitter, jitter)
    right_phi = 270 + random.uniform(-jitter, jitter)
    return [(theta, left_phi), (theta, right_phi)]

def pattern_two_ud(theta_top=35, theta_bottom=140, jitter=6):
    # 上下对打：顶部与底部
    top_phi = 0 + random.uniform(-15, 15)     # 正前上
    bot_phi = 180 + random.uniform(-15, 15)   # 正后下
    return [(theta_top, top_phi), (theta_bottom, bot_phi)]

def pattern_two_diag(theta=78, jitter=8, flip=False):
    # 对角线对打（NE-SW 或 NW-SE）
    a = 45 if not flip else 135
    b = a + 180
    return [(theta, a + random.uniform(-jitter, jitter)),
            (theta, b + random.uniform(-jitter, jitter))]

def pattern_three_tri(theta=78, jitter=6):
    phis = [0, 120, 240]
    return [(theta, p + random.uniform(-jitter, jitter)) for p in phis]

def pattern_four_cross(theta=78, jitter=6):
    phis = [0, 90, 180, 270]
    return [(theta, p + random.uniform(-jitter, jitter)) for p in phis]

def pattern_five_penta(theta=78, jitter=5):
    phis = [0, 72, 144, 216, 288]
    return [(theta, p + random.uniform(-jitter, jitter)) for p in phis]

PATTERNS = [
    ("2_lr", lambda: pattern_two_lr(theta=random.choice([70,75,80,85]))),
    ("2_ud", lambda: pattern_two_ud(theta_top=random.choice([30,35,40]), theta_bottom=random.choice([135,140,145]))),
    ("2_diagA", lambda: pattern_two_diag(theta=random.choice([75,80,85]), flip=False)),
    ("2_diagB", lambda: pattern_two_diag(theta=random.choice([75,80,85]), flip=True)),
    ("3_tri", lambda: pattern_three_tri(theta=random.choice([75,78,82]))),
    ("4_cross", lambda: pattern_four_cross(theta=random.choice([75,78,82]))),
    ("5_penta", lambda: pattern_five_penta(theta=random.choice([75,78,82]))),
]

def build_scene(name_prefix, dirs, size_range=(5,10), inten_range=(8,16),
                base_level=0.0, softness=0.6, res=(2048,1024), allow_white_ratio=0.05):
    recipe = []
    colors = pick_colors(len(dirs), allow_white_ratio=allow_white_ratio)
    for i, ((theta, phi), ck) in enumerate(zip(dirs, colors)):
        sz = random.uniform(*size_range)
        it = random.uniform(*inten_range)
        palette = color_to_palette(ck)
        recipe.append({
            "label": f"l{i+1}_{ck}",
            "theta_deg_phi_deg": [float(theta), float(phi)],
            "palette": palette,
            "color_key": ck,
            "size_deg": float(sz),
            "intensity": float(it),
            "shape": "gaussian",
            "softness": float(softness)
        })
    scene = {
        "name": name_prefix,
        "base_level": float(base_level),
        "base_tint": [1.0, 1.0, 1.0],
        "res": list(res),
        "recipe": recipe
    }
    return scene

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="输出 JSON 路径")
    ap.add_argument("--count", type=int, default=400, help="生成场景数量")
    ap.add_argument("--allow-white", type=float, default=0.05, help="白色使用概率（0~1）")
    ap.add_argument("--size-min", type=float, default=5.0)
    ap.add_argument("--size-max", type=float, default=10.0)
    ap.add_argument("--inten-min", type=float, default=8.0)
    ap.add_argument("--inten-max", type=float, default=16.0)
    ap.add_argument("--softness", type=float, default=0.6)
    ap.add_argument("--res", type=str, default="2048x1024")
    ap.add_argument("--patterns", type=str, default="2_lr,2_ud,2_diagA,2_diagB,3_tri,4_cross,5_penta",
                    help="逗号分隔，从 {2_lr,2_ud,2_diagA,2_diagB,3_tri,4_cross,5_penta} 里选")
    args = ap.parse_args()

    W,H = map(int, args.res.lower().split('x'))
    size_range = (args.size_min, args.size_max)
    inten_range = (args.inten_min, args.inten_max)
    patt_names = [s.strip() for s in args.patterns.split(",") if s.strip()]

    patt_map = {name:fn for name, fn in PATTERNS if name in patt_names}
    if not patt_map:
        raise SystemExit("选择的 patterns 为空。")

    scenes = []
    keys = list(patt_map.keys())

    for i in range(args.count):
        pname = random.choice(keys)
        dirs = patt_map[pname]()
        name_prefix = f"strong_{pname}_{i+1:04d}"
        scene = build_scene(
            name_prefix=name_prefix,
            dirs=dirs,
            size_range=size_range,
            inten_range=inten_range,
            base_level=0.0,
            softness=args.softness,
            res=(W,H),
            allow_white_ratio=args.allow_white
        )
        scenes.append(scene)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {len(scenes)} scenes -> {args.out}")

if __name__ == "__main__":
    main()
