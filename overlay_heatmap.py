import cv2
import os
import random
import torch
import numpy as np
import sys

from PIL import Image
from matplotlib import cm


def overlay_mask(
    img: Image.Image, mask: Image.Image, colormap: str = "jet", alpha: float = 0.7
) -> Image.Image:
    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("img and mask arguments need to be PIL.Image")

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError(
            "alpha argument is expected to be of type float between 0 and 1"
        )

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray(
        (alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8)
    )

    return overlayed_img


def viz_pred_test(
    args, image, ego_pred, GT_mask, aff_list, aff_label, img_name, epoch=None
):
    mean = torch.as_tensor(
        [0.485, 0.456, 0.406], dtype=image.dtype, device=image.device
    ).view(-1, 1, 1)
    std = torch.as_tensor(
        [0.229, 0.224, 0.225], dtype=image.dtype, device=image.device
    ).view(-1, 1, 1)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img = image.squeeze(0) * std + mean
    img = img.detach().cpu().numpy() * 255
    img = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8))

    gt = Image.fromarray(GT_mask)
    gt_result = overlay_mask(img, gt, alpha=0.5)
    aff_str = aff_list[aff_label.item()]

    ego_pred = Image.fromarray(ego_pred)
    ego_pred = overlay_mask(img, ego_pred, alpha=0.5)
    return ego_pred


# ########################################single##########################################################3
file_name = "axe_003531"
obj_class = "_".join(file_name.split("_")[:-1])
aff = "hold"

if file_name == "":
    print("Please enter the file name")
    sys.exit()
if aff == "":
    print("Please enter the affordance")
    sys.exit()

target_image_path = (
    f"./data/AGD20K/Seen/testset/egocentric/{aff}/{obj_class}/{file_name}.jpg"
)
pred_path = (
    f"./to_visualize/2024-11-05 12:36:24.943797/Seen/{aff}/{file_name}/heatmap.jpg"
)
gt_path = f"./data/AGD20K/Seen/testset/GT/{aff}/{obj_class}/{file_name}.png"

target_image = Image.open(target_image_path).convert("RGB")
mask = Image.open(pred_path).convert("L")
gt_mask = Image.open(gt_path).convert("L")

# pred heatmap
# heatmap_final = np.array(pred_path)
heatmap_final = np.array(mask)
heatmap_final = (heatmap_final - heatmap_final.min()) / (
    heatmap_final.max() - heatmap_final.min()
)
heatmap_final = Image.fromarray(heatmap_final)
ego_pred = overlay_mask(target_image, heatmap_final, alpha=0.5)

# gt heatmap
gt_final = np.array(gt_mask)
gt_final = (gt_final - gt_final.min()) / (gt_final.max() - gt_final.min())
gt_final = Image.fromarray(gt_final)
gt_pred = overlay_mask(target_image, gt_final, alpha=0.5)

output_path = f"./quali/{aff}/{obj_class}"
os.makedirs(output_path, exist_ok=True)

gt_pred.save(output_path + f"/{file_name}_gt.png")
ego_pred.save(output_path + f"/{file_name}.png")

# ########################################single##########################################################3

# if __name__ == '__main__':
#     pred_path = 'affordance_output/default_exp/2024-11-05 12:36:24.943797/Seen/hold/knife_000510/heatmap.jpg'
#     aff = pred_path.split('/')[-3]
#     f_name = pred_path.split('/')[-2]
#     obj_class = 'knife'

#     target_image_path = f"./data/AGD20K/Seen/testset/egocentric/{aff}/{obj_class}/{f_name}.jpg"
#     target_image = Image.open(target_image_path).convert("RGB")

#     mask = Image.open(pred_path).convert("L")

#     # heatmap_final = np.array(pred_path)
#     heatmap_final  = np.array(mask)
#     heatmap_final = (heatmap_final-heatmap_final.min())/(heatmap_final.max()-heatmap_final.min())
#     heatmap_final = Image.fromarray(heatmap_final)
#     ego_pred = overlay_mask(target_image, heatmap_final, alpha=0.5)
#     ego_pred.save('quali.png')
