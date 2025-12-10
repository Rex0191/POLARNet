# import logging

import PIL
import torch
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
from copy import deepcopy
from transformers import AutoModelForImageSegmentation

from lbm.models.lbm import LBMModel
from lbm.inference.utils import extract_object, resize_and_center_crop

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

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
def segment_fg(
    fg_image: PIL.Image.Image,
):

    ori_h_bg, ori_w_bg = fg_image.size
    ar_bg = ori_h_bg / ori_w_bg
    closest_ar_bg = min(ASPECT_RATIOS, key=lambda x: abs(float(x) - ar_bg))
    dimensions_bg = ASPECT_RATIOS[closest_ar_bg]

    _, fg_mask = extract_object(birefnet, deepcopy(fg_image))

    # fg_mask = resize_and_center_crop(fg_mask, dimensions_bg[0], dimensions_bg[1])

    fg_mask = fg_mask.resize((ori_h_bg, ori_w_bg))


    return fg_mask

