import logging

import PIL
import torch
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
from copy import deepcopy
from transformers import AutoModelForImageSegmentation

from lbm.models.lbm import LBMModel
from lbm.inference.utils import extract_object, resize_and_center_crop

import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import PIL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ASPECT_RATIOS = {
    str(512 / 2048): (512, 2048),
    str(1024 / 1024): (1024, 1024),
    str(2048 / 512): (2048, 512),
    str(896 / 1152): (896, 1152),
    str(1152 / 896): (1152, 896),
    str(512 / 1920): (512, 1920),
    str(640 / 1536): (640, 1536),
    str(768 / 1280): (768, 1280),
    str(1280 / 768): (1280, 768),
    str(1536 / 640): (1536, 640),
    str(1920 / 512): (1920, 512),
}

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
).cuda()
image_size = (1024, 1024)

@torch.no_grad()
def evaluate(
    model: LBMModel,
    fg_image: PIL.Image.Image,
    bg_image: PIL.Image.Image,
    num_sampling_steps: int = 1,
):

    ori_h_bg, ori_w_bg = fg_image.size
    ar_bg = ori_h_bg / ori_w_bg
    closest_ar_bg = min(ASPECT_RATIOS, key=lambda x: abs(float(x) - ar_bg))
    dimensions_bg = ASPECT_RATIOS[closest_ar_bg]

    _, fg_mask = extract_object(birefnet, deepcopy(fg_image))

    fg_image = resize_and_center_crop(fg_image, dimensions_bg[0], dimensions_bg[1])
    fg_mask = resize_and_center_crop(fg_mask, dimensions_bg[0], dimensions_bg[1])
    bg_image = resize_and_center_crop(bg_image, dimensions_bg[0], dimensions_bg[1])

    img_pasted = Image.composite(fg_image, bg_image, fg_mask)

    img_pasted_tensor = ToTensor()(img_pasted).unsqueeze(0) * 2 - 1
    batch = {
        "source_image": img_pasted_tensor.cuda().to(torch.bfloat16),
    }

    z_source = model.vae.encode(batch[model.source_key])

    output_image = model.sample(
        z=z_source,
        num_steps=num_sampling_steps,
        conditioner_inputs=batch,
        max_samples=1,
    ).clamp(-1, 1)

    output_image = (output_image[0].float().cpu() + 1) / 2
    output_image = ToPILImage()(output_image)

    # paste the output image on the background image
    output_image = Image.composite(output_image, bg_image, fg_mask)

    output_image.resize((ori_h_bg, ori_w_bg))

    return output_image

# @torch.no_grad()
# def evaluate(
#     model: LBMModel,
#     source_image: PIL.Image.Image,
#     num_sampling_steps: int = 1,
# ):
#     """
#     Evaluate the model on an image coming from the source distribution and generate a new image from the target distribution.

#     Args:
#         model (LBMModel): The model to evaluate.
#         source_image (PIL.Image.Image): The source image to evaluate the model on.
#         num_sampling_steps (int): The number of sampling steps to use for the model.

#     Returns:
#         PIL.Image.Image: The generated image.
#     """

#     ori_h_bg, ori_w_bg = source_image.size
#     ar_bg = ori_h_bg / ori_w_bg
#     closest_ar_bg = min(ASPECT_RATIOS, key=lambda x: abs(float(x) - ar_bg))
#     source_dimensions = ASPECT_RATIOS[closest_ar_bg]

#     source_image = source_image.resize(source_dimensions)

#     img_pasted_tensor = ToTensor()(source_image).unsqueeze(0) * 2 - 1
#     batch = {
#         "source_image": img_pasted_tensor.cuda().to(torch.bfloat16),
#     }

#     z_source = model.vae.encode(batch[model.source_key])

#     output_image = model.sample(
#         z=z_source,
#         num_steps=num_sampling_steps,
#         conditioner_inputs=batch,
#         max_samples=1,
#     ).clamp(-1, 1)

#     output_image = (output_image[0].float().cpu() + 1) / 2
#     output_image = ToPILImage()(output_image)
#     output_image.resize((ori_h_bg, ori_w_bg))

#     return output_image


@torch.no_grad()
def evaluate_delighting(
    model: LBMModel,
    source_image: PIL.Image.Image,
    num_sampling_steps: int = 1,
    condition: PIL.Image.Image | None = None,
    mask: PIL.Image.Image | None = None,   # 新增 mask 输入
):
    """
    Delighting 推理：输入受光照的图，输出去光照（均匀/中性光）结果。
    不需要背景，不做前景分割。
    可选：condition 可传 light map 等条件图，若训练时用了条件分支。

    Args:
        model: 训练好的 LBM delighting 模型
        source_image: 输入图（受光照）
        num_sampling_steps: 1/2/4（与你的训练步一致；默认 1）
        condition: 可选条件图（与训练时的格式一致）

    Returns:
        PIL.Image.Image: delighted（去光照）图像
    """
    # —— 修正命名：PIL.size = (width, height)
    ori_w, ori_h = source_image.size

    # 与你现在的长宽比对齐逻辑保持一致
    ar = ori_w / ori_h
    closest_ar = min(ASPECT_RATIOS, key=lambda x: abs(float(x) - ar))
    target_w, target_h = ASPECT_RATIOS[closest_ar]

    # 尺寸对齐（保持和原 evaluate 相同的 resize 策略）
    src = source_image.resize((target_w, target_h), resample=PIL.Image.BICUBIC)

    # 组 batch：键名用你模型里声明的 source_key（通常是 "source_image"）
    img_tensor = ToTensor()(src).unsqueeze(0) * 2 - 1
    batch = {"source_image": img_tensor.cuda().to(torch.bfloat16)}

    # 可选条件（如果你的模型有条件分支）
    if condition is not None:
        cond = condition.resize((target_w, target_h), resample=PIL.Image.BICUBIC)
        cond_tensor = ToTensor()(cond).unsqueeze(0) * 2 - 1
        batch["condition"] = cond_tensor.cuda().to(torch.bfloat16)
    if mask is not None:
        m_t = mask.float().clamp(0, 1)
        if m_t.ndim == 2:                # [H,W] -> [1,1,H,W]
            m_t = m_t.unsqueeze(0).unsqueeze(0)
        elif m_t.ndim == 3 and m_t.shape[0] == 1:  # [1,H,W] -> [1,1,H,W]
            m_t = m_t.unsqueeze(0)
        # 关键：对齐到 source_image 的分辨率（H,W）= (target_h, target_w)
        if m_t.shape[-2:] != (target_h, target_w):
            import torch.nn.functional as F
            m_t = F.interpolate(m_t, size=(target_h, target_w), mode="nearest")
        batch["mask"] = m_t.cuda().to(torch.bfloat16)
    else:
        batch["mask"] = torch.ones(1, 1, target_h, target_w, device="cuda", dtype=torch.bfloat16)

    # VAE encode → LBM sample
    z_source = model.vae.encode(batch[model.source_key])
    # print("z_source shape:", z_source.shape)   # 应该是 [B,4,H/8,W/8]
    # print("mask shape:", batch["mask"].shape)  # 应该是 [B,1,H,W]
    # print("batch keys:", batch.keys())
    # for k,v in batch.items():
    #     print(k, v.shape, v.dtype, v.min().item(), v.max().item())
    output = model.sample(
        z=z_source,
        num_steps=num_sampling_steps,
        conditioner_inputs=batch,
        max_samples=1,
    ).clamp(-1, 1)

    # 回到 PIL，并且——修复：一定要接住 resize 的返回值
    out = (output[0].float().cpu() + 1) / 2
    out_pil = ToPILImage()(out)
    out_pil = out_pil.resize((ori_w, ori_h), resample=PIL.Image.BICUBIC)
    return out_pil

def evaluate_relighting(
    model: LBMModel,
    source_image: PIL.Image.Image,
    num_sampling_steps: int = 1,
    condition: PIL.Image.Image | None = None,
    mask: PIL.Image.Image | None = None,   # 新增 mask 输入
    hdr_env: PIL.Image.Image | None = None,
):
    """
    Relighting 推理：输入受光照的图，输出去光照（均匀/中性光）结果。
    不需要背景，不做前景分割。
    可选：condition 可传 light map 等条件图，若训练时用了条件分支。

    Args:
        model: 训练好的 LBM relighting 模型
        source_image: 输入图（受光照）
        num_sampling_steps: 1/2/4（与你的训练步一致；默认 1）
        condition: 可选条件图（与训练时的格式一致）

    Returns:
        PIL.Image.Image: delighted（去光照）图像
    """
    # —— 修正命名：PIL.size = (width, height)
    ori_w, ori_h = source_image.size

    # 与你现在的长宽比对齐逻辑保持一致
    ar = ori_w / ori_h
    closest_ar = min(ASPECT_RATIOS, key=lambda x: abs(float(x) - ar))
    target_w, target_h = ASPECT_RATIOS[closest_ar]

    # 尺寸对齐（保持和原 evaluate 相同的 resize 策略）
    src = source_image.resize((target_w, target_h), resample=PIL.Image.BICUBIC)

    # 组 batch：键名用你模型里声明的 source_key（通常是 "source_image"）
    img_tensor = ToTensor()(src).unsqueeze(0) * 2 - 1
    batch = {"source_image": img_tensor.cuda().to(torch.bfloat16)}

    # 可选条件（如果你的模型有条件分支）
    if condition is not None:
        cond = condition.resize((target_w, target_h), resample=PIL.Image.BICUBIC)
        cond_tensor = ToTensor()(cond).unsqueeze(0) * 2 - 1
        batch["condition"] = cond_tensor.cuda().to(torch.bfloat16)
    if mask is not None:
        m_t = mask.float().clamp(0, 1)
        if m_t.ndim == 2:                # [H,W] -> [1,1,H,W]
            m_t = m_t.unsqueeze(0).unsqueeze(0)
        elif m_t.ndim == 3 and m_t.shape[0] == 1:  # [1,H,W] -> [1,1,H,W]
            m_t = m_t.unsqueeze(0)
        # 关键：对齐到 source_image 的分辨率（H,W）= (target_h, target_w)
        if m_t.shape[-2:] != (target_h, target_w):
            import torch.nn.functional as F
            m_t = F.interpolate(m_t, size=(target_h, target_w), mode="nearest")
        batch["mask"] = m_t.cuda().to(torch.bfloat16)
    else:
        batch["mask"] = torch.ones(1, 1, target_h, target_w, device="cuda", dtype=torch.bfloat16)
        # 新增 HDR 环境图
    if hdr_env is not None:
        import torch.nn.functional as F
        # hdr_env = hdr_env.resize((target_w, target_h), resample=PIL.Image.BICUBIC)
        # hdr_env_tensor = ToTensor()(hdr_env).unsqueeze(0)
        hdr_env = hdr_env.unsqueeze(0)
        hdr_env = F.interpolate(hdr_env, size=(target_h, target_w), mode="nearest")
        
        batch["hdr_env"] = hdr_env.cuda().to(torch.bfloat16)
        # batch["hdr_env"] = hdr_env.cuda().to(torch.bfloat16)

    # VAE encode → LBM sample
    z_source = model.vae.encode(batch[model.source_key])
    # print("z_source shape:", z_source.shape)   # 应该是 [B,4,H/8,W/8]
    # print("mask shape:", batch["mask"].shape)  # 应该是 [B,1,H,W]
    # print("batch keys:", batch.keys())
    # for k,v in batch.items():
    #     print(k, v.shape, v.dtype, v.min().item(), v.max().item())
    output = model.sample(
        z=z_source,
        num_steps=num_sampling_steps,
        conditioner_inputs=batch,
        max_samples=1,
    ).clamp(-1, 1)

    # 回到 PIL，并且——修复：一定要接住 resize 的返回值
    out = (output[0].float().cpu() + 1) / 2
    out_pil = ToPILImage()(out)
    out_pil = out_pil.resize((ori_w, ori_h), resample=PIL.Image.BICUBIC)
    return out_pil



@torch.no_grad()
def evaluate_relighting_v2(
    model,
    source_image: PIL.Image.Image,
    num_sampling_steps: int = 1,
    condition: PIL.Image.Image | None = None,  # 保留旧接口
    mask: PIL.Image.Image | torch.Tensor | None = None,
    hdr_env: PIL.Image.Image | torch.Tensor | None = None,
):
    """
    与训练时 log_samples() 一致的推理实现，
    同时保持 evaluate_relighting() 的调用接口兼容。
    """
    # === Step 1. 图像预处理 ===
    to_tensor, to_pil = ToTensor(), ToPILImage()
    img_t = to_tensor(source_image).unsqueeze(0) * 2 - 1     # [1,3,H,W], -1~1
    img_t = img_t.to(model.device, dtype=model.dtype)

    # === Step 2. 构建 batch ===
    batch = {model.source_key: img_t}

    # Mask
    if mask is not None:
        if isinstance(mask, PIL.Image.Image):
            mask_t = to_tensor(mask.convert("L")).unsqueeze(0).to(model.device, dtype=model.dtype)
        else:
            mask_t = mask.to(model.device, dtype=model.dtype)
        batch["mask"] = mask_t

    # HDR 环境图
    if hdr_env is not None:
        # if isinstance(hdr_env, PIL.Image.Image):
        #     hdr_env_t = to_tensor(hdr_env.convert("RGB")).unsqueeze(0)
        # else:
        hdr_env_t = hdr_env.unsqueeze(0)
        hdr_env_t = F.interpolate(
            hdr_env_t, size=img_t.shape[-2:], mode="bilinear", align_corners=False
        )
        # gamma = 0.4  # <1 拉亮高光
        # hdr_env_t = torch.pow(hdr_env_t.clamp(min=0), gamma)
        batch["hdr_env"] = hdr_env_t.to(model.device, dtype=model.dtype)

    # 额外条件（如 condition）
    if condition is not None:
        cond_t = to_tensor(condition.convert("RGB")).unsqueeze(0) * 2 - 1
        batch["condition"] = cond_t.to(model.device, dtype=model.dtype)

    # === Step 3. Encode latent ===
    z = model.vae.encode(batch[model.source_key])

    # === Step 4. Sample ===
    with torch.autocast(device_type="cuda", dtype=model.dtype):
        output = model.sample(
            z,
            num_steps=num_sampling_steps,
            conditioner_inputs=batch,
            max_samples=1,
        )

    # === Step 5. 回到 PIL ===
    out = output[0].clamp(-1, 1).float().cpu()
    out = (out + 1) / 2
    out_pil = to_pil(out)

    return out_pil



def evaluate_olat_old(
    model: LBMModel,
    source_image: PIL.Image.Image,
    num_sampling_steps: int = 1,
    condition: PIL.Image.Image | None = None,
    mask: PIL.Image.Image | None = None,   # 新增 mask 输入
    hdr_env: PIL.Image.Image | None = None,
):
    """
    Relighting 推理：输入受光照的图，输出去光照（均匀/中性光）结果。
    不需要背景，不做前景分割。
    可选：condition 可传 light map 等条件图，若训练时用了条件分支。

    Args:
        model: 训练好的 LBM relighting 模型
        source_image: 输入图（受光照）
        num_sampling_steps: 1/2/4（与你的训练步一致；默认 1）
        condition: 可选条件图（与训练时的格式一致）

    Returns:
        PIL.Image.Image: delighted（去光照）图像
    """
    # —— 修正命名：PIL.size = (width, height)
    ori_w, ori_h = source_image.size

    # 与你现在的长宽比对齐逻辑保持一致
    ar = ori_w / ori_h
    closest_ar = min(ASPECT_RATIOS, key=lambda x: abs(float(x) - ar))
    target_w, target_h = ASPECT_RATIOS[closest_ar]

    # 尺寸对齐（保持和原 evaluate 相同的 resize 策略）
    src = source_image.resize((target_w, target_h), resample=PIL.Image.BICUBIC)

    # 组 batch：键名用你模型里声明的 source_key（通常是 "source_image"）
    img_tensor = ToTensor()(src).unsqueeze(0) * 2 - 1
    batch = {"source_image": img_tensor.cuda().to(torch.bfloat16)}

    # 可选条件（如果你的模型有条件分支）
    if condition is not None:
        cond = condition.resize((target_w, target_h), resample=PIL.Image.BICUBIC)
        cond_tensor = ToTensor()(cond).unsqueeze(0) * 2 - 1
        batch["condition"] = cond_tensor.cuda().to(torch.bfloat16)
    if mask is not None:
        m_t = mask.float().clamp(0, 1)
        if m_t.ndim == 2:                # [H,W] -> [1,1,H,W]
            m_t = m_t.unsqueeze(0).unsqueeze(0)
        elif m_t.ndim == 3 and m_t.shape[0] == 1:  # [1,H,W] -> [1,1,H,W]
            m_t = m_t.unsqueeze(0)
        # 关键：对齐到 source_image 的分辨率（H,W）= (target_h, target_w)
        if m_t.shape[-2:] != (target_h, target_w):
            import torch.nn.functional as F
            m_t = F.interpolate(m_t, size=(target_h, target_w), mode="nearest")
        batch["mask"] = m_t.cuda().to(torch.bfloat16)
    else:
        batch["mask"] = torch.ones(1, 1, target_h, target_w, device="cuda", dtype=torch.bfloat16)
        # 新增 HDR 环境图
    if hdr_env is not None:
        # hdr_env = hdr_env.resize((target_w, target_h), resample=PIL.Image.BICUBIC)
        # hdr_env_tensor = ToTensor()(hdr_env).unsqueeze(0)
        hdr_env = hdr_env.unsqueeze(0)
        hdr_env = F.interpolate(hdr_env, size=(target_h, target_w), mode="nearest")
        
        batch["hdr_env"] = hdr_env.cuda().to(torch.bfloat16)
        # batch["hdr_env"] = hdr_env.cuda().to(torch.bfloat16)

    # VAE encode → LBM sample
    z_source = model.vae.encode(batch[model.source_key])
    # print("z_source shape:", z_source.shape)   # 应该是 [B,4,H/8,W/8]
    # print("mask shape:", batch["mask"].shape)  # 应该是 [B,1,H,W]
    # print("batch keys:", batch.keys())
    # for k,v in batch.items():
    #     print(k, v.shape, v.dtype, v.min().item(), v.max().item())
    output = model.sample(
        z=z_source,
        num_steps=num_sampling_steps,
        conditioner_inputs=batch,
        max_samples=1,
    ).clamp(-1, 1)

    # src_t = batch["source_image"]  # [-1, 1] 区间
    # output = output.to(src_t.device)
    # pred = src_t - output         # 反向恢复目标图
    # pred = pred.clamp(-1, 1)       # 避免溢出

    # 回到 PIL，并且——修复：一定要接住 resize 的返回值
    out = (output[0].float().cpu() + 1) / 2
    out_pil = ToPILImage()(out)
    out_pil = out_pil.resize((ori_w, ori_h), resample=PIL.Image.BICUBIC)
    return out_pil

def evaluate_olat(
    model,
    source_image: PIL.Image.Image,
    num_sampling_steps: int = 1,
    condition: PIL.Image.Image | None = None,  # 保留旧接口
    mask: PIL.Image.Image | torch.Tensor | None = None,
    hdr_env: PIL.Image.Image | torch.Tensor | None = None,
):
    """
    与训练时 log_samples() 一致的推理实现，
    同时保持 evaluate_relighting() 的调用接口兼容。
    """
    # === Step 1. 图像预处理 ===
    to_tensor, to_pil = ToTensor(), ToPILImage()
    img_t = to_tensor(source_image).unsqueeze(0) * 2 - 1     # [1,3,H,W], -1~1
    img_t = img_t.to(model.device, dtype=model.dtype)

    # === Step 2. 构建 batch ===
    batch = {model.source_key: img_t}

    # Mask
    if mask is not None:
        if isinstance(mask, PIL.Image.Image):
            mask_t = to_tensor(mask.convert("L")).unsqueeze(0).to(model.device, dtype=model.dtype)
        else:
            mask_t = mask.to(model.device, dtype=model.dtype)
        batch["mask"] = mask_t

    # HDR 环境图
    if hdr_env is not None:
        # if isinstance(hdr_env, PIL.Image.Image):
        #     hdr_env_t = to_tensor(hdr_env.convert("RGB")).unsqueeze(0)
        # else:
        hdr_env_t = hdr_env.unsqueeze(0)
        hdr_env_t = F.interpolate(
            hdr_env_t, size=img_t.shape[-2:], mode="bilinear", align_corners=False
        )
        # gamma = 0.4  # <1 拉亮高光
        # hdr_env_t = torch.pow(hdr_env_t.clamp(min=0), gamma)
        batch["hdr_env"] = hdr_env_t.to(model.device, dtype=model.dtype)

    # 额外条件（如 condition）
    if condition is not None:
        cond_t = to_tensor(condition.convert("RGB")).unsqueeze(0) * 2 - 1
        batch["condition"] = cond_t.to(model.device, dtype=model.dtype)

    # === Step 3. Encode latent ===
    z = model.vae.encode(batch[model.source_key])

    # === Step 4. Sample ===
    with torch.autocast(device_type="cuda", dtype=model.dtype):
        output = model.sample(
            z,
            num_steps=num_sampling_steps,
            conditioner_inputs=batch,
            max_samples=1,
        )

    # === Step 5. 回到 PIL ===
    out = output[0].clamp(-1, 1).float().cpu()
    out = (out + 1) / 2
    out_pil = to_pil(out)

    return out_pil
