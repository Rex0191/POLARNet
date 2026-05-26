import os
from uniform_with_bg_gamma import OLATDelight, infer_out_dir
import argparse

def has_composite(out_dir: str, suffix="_composite.png") -> bool:
    """
    判断目录里是否已经有合成的结果文件
    （避免误判：只认以 _composite.png 结尾的成品文件）
    """
    if not os.path.isdir(out_dir):
        return False
    try:
        for fn in os.listdir(out_dir):
            if fn.endswith(suffix):
                return True
    except FileNotFoundError:
        return False
    return False

def batch_process(root_dir,
                  root_out,
                  olat_txt,
                  base_map_dir,
                  envmap_dir,
                  background_dir,
                  cam_range=(5, 20),
                  skip_existing=True):
    """
    遍历 root_dir 下所有 ID 的 C05~C20 相机目录，批量做 OLAT 打光 + 合成
    :param skip_existing: 如果输出目录已有结果，则跳过
    """
    for id_name in os.listdir(root_dir):
        id_path = os.path.join(root_dir, id_name)
        if not os.path.isdir(id_path):
            continue

        # ID 下的 OLAT session
        for session in os.listdir(id_path):
            session_path = os.path.join(id_path, session)
            if not os.path.isdir(session_path):
                continue

            # 遍历 C05~C20
            for cam_idx in range(cam_range[0], cam_range[1] + 1):
                cam_name = f"C{cam_idx:02d}"
                cam_path = os.path.join(session_path, cam_name)
                if not os.path.isdir(cam_path):
                    continue

                # 推导输出目录
                out_dir = infer_out_dir(cam_path, root_in=root_dir, root_out=root_out)

                # 如果已有结果且启用 skip_existing，就跳过
                if skip_existing and has_composite(out_dir):
                    print(f"[SKIP] 已有结果，跳过 {cam_path}")
                    continue

                print(f"\n[INFO] Processing {cam_path} ...")

                relighter = OLATDelight(
                    olat_txt=olat_txt,
                    olat_img_dir=cam_path,
                    root_in=root_dir,
                    root_out=root_out,
                    base_map_dir=base_map_dir,
                    envmap_dir=envmap_dir,
                    background_dir=background_dir
                )
                relighter.run()

    print("全部完成 ✅")

import random

def batch_process_random_envmaps(root_dir,
                                root_out,
                                 olat_txt,
                                 base_map_dir,
                                 envmap_dir,
                                 background_dir,
                                 cam_range=(5, 20),
                                 skip_existing=True,
                                 num_envmaps=10):
    """
    遍历 root_dir 下所有 ID 的 C05~C20 相机目录，随机选取 num_envmaps 个环境光
    """
    # 获取所有可用 envmaps
    all_envmaps = [os.path.join(envmap_dir, f)
                   for f in os.listdir(envmap_dir)
                   if f.endswith(".hdr") or f.endswith(".exr")]

    if len(all_envmaps) < num_envmaps:
        raise ValueError(f"环境图数量不足: 仅 {len(all_envmaps)} 张")

    for id_name in os.listdir(root_dir):
        id_path = os.path.join(root_dir, id_name)
        if not os.path.isdir(id_path):
            continue

        # ID 下的 OLAT session
        for session in os.listdir(id_path):
            session_path = os.path.join(id_path, session)
            if not os.path.isdir(session_path):
                continue

            # 遍历 C05~C20
            for cam_idx in range(cam_range[0], cam_range[1] + 1):
                cam_name = f"C{cam_idx:02d}"
                cam_path = os.path.join(session_path, cam_name)
                if not os.path.isdir(cam_path):
                    continue

                out_dir = infer_out_dir(cam_path, root_in=root_dir, root_out=root_out)
                if skip_existing and has_composite(out_dir):
                    print(f"[SKIP] 已有结果，跳过 {cam_path}")
                    continue

                # 随机选取 num_envmaps
                envmaps = random.sample(all_envmaps, num_envmaps)

                print(f"\n[INFO] Processing {cam_path}, 随机选择 {num_envmaps} 张环境光...")

                relighter = OLATDelight(
                    olat_txt=olat_txt,
                    olat_img_dir=cam_path,
                    root_in=root_dir,
                    root_out=root_out,
                    base_map_dir=base_map_dir,
                    envmap_dir=None,           # 不用目录
                    background_dir=background_dir
                )
                # 替换 envmaps 列表
                relighter.envmaps = envmaps
                relighter.run()

    print("全部完成 ✅")

from concurrent.futures import ProcessPoolExecutor, as_completed

def process_one(cam_path, root_dir, root_out, olat_txt, base_map_dir, envmaps, background_dir):
    """单个相机目录的处理逻辑"""
    print(f"[INFO] 子进程处理 {cam_path}, 环境光 {len(envmaps)} 张")
    relighter = OLATDelight(
        olat_txt=olat_txt,
        olat_img_dir=cam_path,
        root_in=root_dir,
        root_out=root_out,
        base_map_dir=base_map_dir,
        envmap_dir=None,   # 外部指定 envmaps
        background_dir=background_dir
    )
    relighter.envmaps = envmaps  # 覆盖
    relighter.run()
    return cam_path

def batch_process_parallel(root_dir,
                           root_out,
                           olat_txt,
                           base_map_dir,
                           envmap_dir,
                           background_dir,
                           cam_range=(5, 20),
                           num_envmaps=10,
                           num_workers=4,
                           skip_existing=True):
    """并行批量处理"""
    # 所有环境图
    all_envmaps = [os.path.join(envmap_dir, f)
                   for f in os.listdir(envmap_dir)
                   if f.endswith(".hdr") or f.endswith(".exr")]
    if len(all_envmaps) < num_envmaps:
        raise ValueError(f"环境图数量不足: {len(all_envmaps)}")

    cam_paths = []
    for id_name in os.listdir(root_dir):
        id_path = os.path.join(root_dir, id_name)
        if not os.path.isdir(id_path): continue

        for session in os.listdir(id_path):
            session_path = os.path.join(id_path, session)
            if not os.path.isdir(session_path): continue

            for cam_idx in range(cam_range[0], cam_range[1] + 1):
                cam_name = f"C{cam_idx:02d}"
                cam_path = os.path.join(session_path, cam_name)
                if not os.path.isdir(cam_path): continue
                cam_paths.append(cam_path)
        # for cam_idx in range(cam_range[0], cam_range[1] + 1):
        #     cam_name = f"C{cam_idx:02d}"
        #     cam_path = os.path.join(id_path, cam_name)
        #     if not os.path.isdir(cam_path): continue
        #     cam_paths.append(cam_path)

    print(f"[INFO] 总共 {len(cam_paths)} 个相机目录待处理")

    # 多进程并行
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for cam_path in cam_paths:
            out_dir = infer_out_dir(cam_path, root_in=root_dir, root_out=root_out)
            if skip_existing and has_composite(out_dir):
                print(f"[SKIP] 已有合成结果，跳过 {cam_path}")
                continue
            envmaps = random.sample(all_envmaps, num_envmaps)
            futures.append(executor.submit(
                process_one, cam_path, root_dir, root_out, olat_txt, base_map_dir, envmaps, background_dir
            ))
        for fut in as_completed(futures):
            print(f"✅ 完成 {fut.result()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, help="第一步输出目录 (ori_OLAT_x)")
    parser.add_argument("--root_out", required=True, help="第二步输出目录 (for_segmentation_x)")
    parser.add_argument("--olat_txt", required=True)
    parser.add_argument("--base_map_dir", required=True)
    parser.add_argument("--envmap_dir", required=True)
    parser.add_argument("--background_dir", required=True)
    parser.add_argument("--num_envmaps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    batch_process_parallel(
        root_dir=args.root_dir,
        root_out=args.root_out,
        olat_txt=args.olat_txt,
        base_map_dir=args.base_map_dir,
        envmap_dir=args.envmap_dir,
        background_dir=args.background_dir,
        cam_range=(5, 20),
        num_envmaps=args.num_envmaps,
        num_workers=args.num_workers,
        skip_existing=True
    )

