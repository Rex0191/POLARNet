import os
import cv2
import numpy as np
import imageio
import glob


def infer_mask_dir(
    olat_img_dir: str,
    root_in: str,
    root_alpha: str
) -> str:
    rel_path = os.path.relpath(olat_img_dir, root_in)
    alpha_dir = os.path.join(root_alpha, rel_path)
    return alpha_dir

def infer_out_dir(
    olat_img_dir: str,
    root_in: str,
    root_out: str
) -> str:
    rel_path = os.path.relpath(olat_img_dir, root_in)
    out_dir = os.path.join(root_out, rel_path)
    return out_dir

def rgb_to_intensity(rgb):
    # 亮度系数（近似人眼敏感度）
    return 0.2126 * rgb[...,0] + 0.7152 * rgb[...,1] + 0.0722 * rgb[...,2]


class OLATRelightAndComposite:
    def __init__(self, olat_txt, olat_img_dir, root_in, root_alpha, root_out,
                 base_map_dir, envmap_dir, background_dir,
                 base_size=(512, 256)):
        self.Wb, self.Hb = base_size
        self.olat_txt = olat_txt
        self.olat_img_dir = olat_img_dir
        self.base_map_dir = base_map_dir
        self.envmap_dir = envmap_dir
        self.background_dir = background_dir

        # 自动推导 mask_dir 和 out_dir
        self.mask_dir = infer_mask_dir(self.olat_img_dir, root_in, root_alpha)
        self.out_dir = infer_out_dir(self.olat_img_dir, root_in, root_out)

        os.makedirs(self.out_dir, exist_ok=True)

        print(f"[INFO] 自动推导 mask_dir: {self.mask_dir}")
        print(f"[INFO] 自动推导 out_dir: {self.out_dir}")

        self.load_data()

    def load_data(self):
        self.olats = []
        self.base_maps = []
        with open(self.olat_txt, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                name = parts[0]
                idx = int(parts[1])

                img_path = os.path.join(self.olat_img_dir, name)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"[WARN] {img_path} not found, skip.")
                    continue
                img = img.astype(np.float32) / 255.0
                self.olats.append(img)

                base_path = os.path.join(self.base_map_dir, f"{idx:03d}.png")
                base = cv2.imread(base_path, cv2.IMREAD_GRAYSCALE)
                if base is None:
                    print(f"[WARN] {base_path} not found, skip.")
                    continue
                base = cv2.resize(base, (self.Wb, self.Hb)).astype(np.float32) / 255.0
                self.base_maps.append(base)

        self.olats = np.stack(self.olats, axis=0)
        self.base_maps = np.stack(self.base_maps, 0)

        # self.envmaps = sorted(glob.glob(os.path.join(self.envmap_dir, "*.hdr")) +
        #                       glob.glob(os.path.join(self.envmap_dir, "*.exr")))
        if self.envmap_dir is not None:
            self.envmaps = sorted(
                glob.glob(os.path.join(self.envmap_dir, "*.hdr")) +
                glob.glob(os.path.join(self.envmap_dir, "*.exr")),
                reverse=True
            )
            print(f"[INFO] 找到 {len(self.envmaps)} 个环境图")
        else:
            self.envmaps = []
            print("[INFO] 未指定 envmap_dir，将由外部传入 self.envmaps")
            print(f"[INFO] 找到 {len(self.envmaps)} 个环境图")

        self.masks = glob.glob(os.path.join(self.mask_dir, "*_alpha.png"))
        if not self.masks:
            print(f"[ERR] 在 {self.mask_dir} 下没有找到 *_alpha.png")

    def compute_weights(self, envmap, return_chroma=False):
        weights_diffuse = []
        weights_specular_rgb = []
        weights_specular_chroma = []
        inten = rgb_to_intensity(envmap)  # (H,W)
        for i in range(len(self.base_maps)):
            msk = self.base_maps[i]
            if msk.ndim == 3:
                msk = msk[...,0]
            msk = msk.astype(np.float32)
            area = float(msk.sum() + 1e-6)

            w_diff = float((inten * msk).sum() / area)
            w_spec_rgb = ((envmap * msk[...,None]).sum(axis=(0,1)) / area).astype(np.float32)

            L = float(0.2126*w_spec_rgb[0] + 0.7152*w_spec_rgb[1] + 0.0722*w_spec_rgb[2] + 1e-6)
            w_spec_chroma = (w_spec_rgb / L).astype(np.float32)

            weights_diffuse.append(w_diff)
            weights_specular_rgb.append(w_spec_rgb)
            weights_specular_chroma.append(w_spec_chroma)

        if return_chroma:
            return (np.array(weights_diffuse, np.float32),
                    np.stack(weights_specular_rgb, axis=0),
                    np.stack(weights_specular_chroma, axis=0))
        else:
            # Backward compatibility: return only (diffuse, spec_rgb)
            return (np.array(weights_diffuse, np.float32),
                    np.stack(weights_specular_rgb, axis=0))


    def run(self):
        for env_path in self.envmaps:
            env = imageio.imread(env_path)
            env = cv2.resize(env, (self.Wb, self.Hb)).astype(np.float32)
            den = np.quantile(rgb_to_intensity(env), getattr(self, "env_exposure_q", 0.999)) + 1e-6
            env = env / den

            weights_diffuse, weights_specular = self.compute_weights(env)

            # --- New composition ---
            result = np.zeros_like(self.olats[0])
            weights_diffuse, weights_specular_rgb, weights_specular_chroma = self.compute_weights(env, return_chroma=True)

            diffuse_strength  = getattr(self, "diffuse_strength", 0.8)
            specular_strength = getattr(self, "specular_strength", 0.8)
            specular_gain     = getattr(self, "specular_gain", 1.0)

            for i in range(len(self.olats)):
                result += diffuse_strength * weights_diffuse[i] * self.olats[i]
                chroma = weights_specular_chroma[i][None, None, :]
                result += specular_strength * (specular_gain * self.olats[i] * chroma)

            exp_q = getattr(self, "exposure_q", 0.999)
            exp = np.quantile(rgb_to_intensity(result), exp_q)
            result = result / max(exp, 1e-6)

            result = np.clip(result, 0, None)
            result = np.where(result <= 0.0031308, result, np.power(result, 1/getattr(self, "gamma_power", 1.4)))

        #     result = np.where(
        #     result <= 0.0031308,
        #     12.92 * result,
        #     result
        # )

            env_name = os.path.splitext(os.path.basename(env_path))[0]
            bg_path = os.path.join(self.background_dir, env_name + ".png")
            if not os.path.exists(bg_path):
                print(f"[WARN] 背景 {bg_path} 不存在，跳过。")
                continue
            bg = cv2.imread(bg_path).astype(np.float32) / 255.0

            if not self.masks:
                print("[ERR] 没找到 mask，跳过。")
                continue
            mask = cv2.imread(self.masks[0], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            mask = cv2.merge([mask, mask, mask])
            mask = cv2.resize(mask, (bg.shape[1], bg.shape[0]))

            if result.shape != bg.shape:
                result = cv2.resize(result, (bg.shape[1], bg.shape[0]))

            final = mask * result + (1 - mask) * bg

            out_path = os.path.join(self.out_dir, env_name + "_composite.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, (final * 255).astype(np.uint8))
            print(f"✅ {env_name} -> {out_path}")


# if __name__ == "__main__":
#     relighter = OLATRelightAndComposite(
#         olat_txt="/mnt/bn/pico-idl-avatar2/cz/OLAT/data/light_157_proc.txt",
#         olat_img_dir="/mnt/bn/pico-idl-avatar2/cz/OLAT/datasets_processed/0010/OLAT_SJTU_4D_WJ_0807_01_00/C01",
#         base_map_dir="/mnt/bn/pico-idl-avatar2/cz/OLAT/data/OLAT_EnvMaps/",
#         envmap_dir="/mnt/bn/pico-idl-avatar2/cz/OLAT/data/hdrs_all/",
#         background_dir="/mnt/bn/pico-idl-avatar2/cz/OLAT/data/hdr_background/"
#     )
#     relighter.run()
#     print("全部完成 ✅")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OLAT relight and composite with stronger colored specular")
    parser.add_argument("--olat_txt", required=True)
    parser.add_argument("--olat_img_dir", required=True)
    parser.add_argument("--base_map_dir", required=True, help="dir with *_alpha.png base masks")
    parser.add_argument("--envmap_dir", required=True, help="dir with .hdr/.exr environment maps")
    parser.add_argument("--background_dir", required=True, help="dir with background PNGs named after env (e.g., sky.hdr -> sky.png)")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    relighter = OLATRelightAndComposite(
        olat_txt=args.olat_txt,
        olat_img_dir=args.olat_img_dir,
        base_map_dir=args.base_map_dir,
        envmap_dir=args.envmap_dir,
        background_dir=args.background_dir,
        out_dir=args.out_dir
    )
    relighter.run()
    print("全部完成 ✅")
