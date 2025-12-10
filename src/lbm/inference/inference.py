import logging

import PIL
import torch
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
from copy import deepcopy
from transformers import AutoModelForImageSegmentation

from lbm.models.lbm import LBMModel
from lbm.inference.utils import extract_object, resize_and_center_crop

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