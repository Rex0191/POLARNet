import os
import cv2
import numpy as np
import imageio
import glob


def gamma_stretch(img, gamma=2.2):
    """
    对最终结果做分位点拉伸 + gamma 矫正。
    img: HxWx3, float32, [0,1]
    """
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    img_gamma = np.power(img, 1.0/float(gamma))
    return np.clip(img_gamma, 0.0, 1.0)

def infer_out_dir(
    olat_img_dir: str,
    root_in: str = "/mnt/bn/idl-data-cache/cz/data/OLAT/ori_OLAT",
    root_out: str = "/mnt/bn/idl-data-cache/cz/data/OLAT/for_segmentation"
) -> str:
    """
    根据输入路径推导输出目录：
    - 保留 root_in 之后的相对路径
    - 拼接到 root_out 下
    
    Args:
        olat_img_dir: 输入路径（可以是图片文件或目录）
        root_in: 输入根目录 (默认: /mnt/bn/pico-idl-avatar2/cz/OLAT/datasets_processed)
        root_out: 输出根目录 (默认: /mnt/bn/idl-data-cache/cz/data/OLAT/synthetic_light)
    
    Returns:
        输出目录路径
    """
    # 相对路径
    rel_path = os.path.relpath(olat_img_dir, root_in)
    # 拼接输出路径
    out_dir = os.path.join(root_out, rel_path)
    return out_dir


class OLATDelight:
    def __init__(self, olat_txt, olat_img_dir,
                 base_map_dir, envmap_dir, background_dir,
                 base_size=(512, 256)):
        self.Wb, self.Hb = base_size
        self.olat_txt = olat_txt
        self.olat_img_dir = olat_img_dir
        self.base_map_dir = base_map_dir
        self.envmap_dir = envmap_dir
        self.background_dir = background_dir

        # 自动推导 mask_dir 和 out_dir
        
        self.out_dir = infer_out_dir(self.olat_img_dir)
        os.makedirs(self.out_dir, exist_ok=True)

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

    def compute_weights(self, envmap):
        weights = []
        for i in range(len(self.base_maps)):
            w = np.sum(envmap * self.base_maps[i, :, :, None])
            weights.append(w)
        return np.array(weights, dtype=np.float32)

    def run(self):
        for env_path in self.envmaps:
            env = imageio.imread(env_path)
            env = cv2.resize(env, (self.Wb, self.Hb)).astype(np.float32)
            env = env / np.max(env)

            weights = self.compute_weights(env)

            result = np.zeros_like(self.olats[0])
            for i in range(len(weights)):
                result += weights[i] * self.olats[i]
            result /= np.max(result)
            # test 
            result = np.where(
            result <= 0.0031308,
            12.92 * result,
            result
        )

            env_name = os.path.splitext(os.path.basename(env_path))[0]

            final = result
            final = gamma_stretch(final, gamma=2.4)

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
