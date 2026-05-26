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
    # BR之前反了！
    return 0.2126 * rgb[...,2] + 0.7152 * rgb[...,1] + 0.0722 * rgb[...,0]


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

    def compute_weights(self, envmap):
        weights_diffuse = []
        weights_specular = []
        for i in range(len(self.base_maps)):
            mask = self.base_maps[i, :, :, None]  # (H,W,1)
            
            # 灰度强度 -> diffuse
            intensity = rgb_to_intensity(envmap)
            w_diff = np.sum(intensity * self.base_maps[i])  # 标量
            
            # 保留RGB -> specular
            w_spec = np.sum(envmap * mask, axis=(0,1))  # (3,)

            weights_diffuse.append(w_diff)
            weights_specular.append(w_spec)
        
        return np.array(weights_diffuse, dtype=np.float32), np.stack(weights_specular, axis=0)


    def run(self):
        for env_path in self.envmaps:
            env = imageio.imread(env_path)
            env = cv2.resize(env, (self.Wb, self.Hb)).astype(np.float32)
            env = env / np.max(env)

            weights_diffuse, weights_specular = self.compute_weights(env)

            result = np.zeros_like(self.olats[0])

            for i in range(len(self.olats)):
                
                # 1. Diffuse：标量 × OLAT → 保持肤色（不会全脸染色）
                result += 0.5* weights_diffuse[i] * self.olats[i]

                # 2. Specular：RGB × OLAT → 叠加彩色高光
                specular_strength = 0.5   # 可以调节 0.2~0.5
                result += self.olats[i] * weights_specular[i][::-1][None, None, :] * specular_strength
            result /= (np.max(result) + 1e-6)
            # test 
            result = np.where(result <= 0.0031308,
                  1 * result,
                  np.power(result, 1/1.0))

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
            # mask = cv2.resize(mask, (bg.shape[1], bg.shape[0]))
            bg = cv2.resize(bg, (result.shape[1], result.shape[0]))
            mask = cv2.resize(mask, (result.shape[1], result.shape[0]))

            # if result.shape != bg.shape:
            #     result = cv2.resize(result, (bg.shape[1], bg.shape[0]))

            final = mask * result + (1 - mask) * bg
            # final = result

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
