import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
import argparse
import re
import sys
import pathlib
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from tqdm import tqdm, trange

from openai import OpenAI
from torchvision import transforms
import time
import urllib.request


"""
Hyper parameters
"""


# TEXT_PROMPT = "hand. finger."
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_image(image_path: pathlib.Path):
    image = cv2.imread(str(image_path))
    assert image is not None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_width = (image.shape[1] // 32) * 32
    image_height = (image.shape[0] // 32) * 32

    image = image[:image_height, :image_width]
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--image_size", help="image size dalle generates", default="1024x1024"
    )
    parser.add_argument("--cache_path", help="path to generated image", default="cache")
    parser.add_argument(
        "--text_prompt",
        type=str,
        required=True,
        help='Objects to search (ex. "hand. finger. arm.")',
    )

    args = parser.parse_args()

    TEXT_PROMPT = args.text_prompt

    image_size = args.image_size
    cache_path = args.cache_path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam2_checkpoint = SAM2_CHECKPOINT
    model_cfg = SAM2_MODEL_CONFIG
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE,
    )

    affordance2gen_prompt = {
        "carry": "A picture of a {} and a hand carrying the {}",
        "catch": "A picture of a {} and a hand catching the {}",
        "hold": "A picture of a {} and a hand holding the {}",
        "take_photo": "A picture of a {} and a hand taking a photo with the {}",
        "type_on": "A picture of a {} and a hand typing on the {}",
        "swing": "A picture of a {} and a hand swinging the {}",
        "pour": "A picture of a {} and a hand pouring the {}",
        "pick_up": "A picture of a {} and a hand picking up the {}",
        "hit": "A picture of a {} and a hand hitting the {}",
        "look_out": "A picture of a {} and a hand looking out with the {}",
        "lift": "A picture of a {} and a hand lifting the {}",
        "write": "A picture of a {} and a hand writing the {}",
        "text_on": "A picture of a {} and a hand texting on the {}",
    }

    client = OpenAI()

    os.makedirs(cache_path, exist_ok=True)
    aff_obj_dict = json.load(open("affordance_object_dict.json"))

    for affordance_type , obj_list in tqdm(aff_obj_dict.items(), desc = 'aff'):
        os.makedirs(f'{cache_path}/{affordance_type}', exist_ok=True)
        for object_type in tqdm(obj_list, desc='obj'):
            os.makedirs(f'{cache_path}/{affordance_type}/{object_type}', exist_ok=True)
            i=0
            while i<9:
                try:
                    response = client.images.generate(
                        model="dall-e-2",
                        prompt=affordance2gen_prompt[affordance_type].format(object_type, object_type),
                        size=image_size,
                        quality="standard",
                        n=1, # generate one image for now
                    ).data[0]
                    
                    urllib.request.urlretrieve(
                        response.url,
                        f'{cache_path}/{affordance_type}/{object_type}/{i+1}.png'
                    )
                    # time.sleep(10)

                    print(affordance_type, object_type, i, 'success')
                    
                    text = TEXT_PROMPT
                    img_path = f'{cache_path}/{affordance_type}/{object_type}/{i+1}.png'
                    img_name = f"{i+1}" # w/o extension
                    image_source, image = load_image(img_path)
                    img = cv2.imread(img_path)
                    
                    sam2_predictor.set_image(image_source)

                    boxes, confidences, labels = predict(
                        model=grounding_model,
                        image=image,
                        caption=text,
                        box_threshold=BOX_THRESHOLD,
                        text_threshold=TEXT_THRESHOLD,
                    )

                    h, w, _ = image_source.shape
                    boxes = boxes * torch.Tensor([w, h, w, h])
                    input_boxes = box_convert(
                        boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy"
                    ).numpy()
                    
                    try:
                        masks, scores, logits = sam2_predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_boxes,
                            multimask_output=False,
                        )

                        """
                        Post-process the output of the model to get the masks, scores, and logits for visualization
                        """
                        # convert the shape to (n, H, W)
                        if masks.ndim == 4:
                            masks = masks.squeeze(1)

                        confidences = confidences.numpy().tolist()
                        class_names = labels

                        class_ids = np.array(list(range(len(class_names))))

                        labels = [
                            f"{class_name} {confidence:.2f}"
                            for class_name, confidence in zip(class_names, confidences)
                        ]

                        """
                        Visualize image with supervision useful API
                        """
                        detections = sv.Detections(
                            xyxy=input_boxes,  # (n, 4)
                            mask=masks.astype(bool),  # (n, h, w)
                            class_id=class_ids,
                        )

                        annotated_frame = img.copy()

                        mask_annotator = sv.MaskAnnotator()
                        annotated_frame, mask_bitmap = mask_annotator.annotate(
                            scene=annotated_frame, detections=detections
                        )

                        image_arr = np.array(img)
                        OUTPUT_DIR = f"{cache_path}/{affordance_type}/{object_type}/"
                        # cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_name}.png"), image_arr)
                        if args.text_prompt == "hand.":
                            cv2.imwrite(
                                os.path.join(OUTPUT_DIR, f"{img_name}_mask_hand.png"), mask_bitmap
                            )
                        elif args.text_prompt == "hand. finger.":
                            cv2. imwrite(
                                os.path.join(OUTPUT_DIR, f"{img_name}_mask_hand_finger.png"), mask_bitmap
                            )
                        elif args.text_prompt == "hand. finger. arm.":
                            cv2. imwrite(
                                os.path.join(OUTPUT_DIR, f"{img_name}_mask_hand_finger_arm.png"), mask_bitmap
                            )
                        else:
                            print(f"{args.text_prompt}")
                            print("Determine the name of the mask image")
                            sys.exit()

                    except Exception as e:
                        print(f"Could not find the objects in the image: {img_path}")
                        image_arr = np.array(img)
                        zero_mask = np.zeros_like(img)
                        # cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_name}.png"), image_arr)
                        if args.text_prompt == "hand.":
                            cv2.imwrite(
                                os.path.join(OUTPUT_DIR, f"{img_name}_mask_hand.png"), mask_bitmap
                            )
                        elif args.text_prompt == "hand. finger.":
                            cv2. imwrite(
                                os.path.join(OUTPUT_DIR, f"{img_name}_mask_hand_finger.png"), mask_bitmap
                            )
                        elif args.text_prompt == "hand. finger. arm.":
                            cv2. imwrite(
                                os.path.join(OUTPUT_DIR, f"{img_name}_mask_hand_finger_arm.png"), mask_bitmap
                            )
                        else:
                            print(f"{args.text_prompt}")
                            print("Determine the name of the mask image")
                            sys.exit()
                            
                    cv2.destroyAllWindows()

                    # Remove masked pixels from original image
                    orig_image = cv2.imread(f'{cache_path}/{affordance_type}/{object_type}/{i+1}.png')
                    masked_image = np.concatenate([orig_image, ~mask_bitmap], axis=-1)
                    cv2.imwrite(f"{cache_path}/{affordance_type}/{object_type}/{i+1}_mask.png", masked_image)
                    
                    
                    
                    
                    
                    

                except Exception as e:
                    print(e)
                    print(affordance_type, object_type, i, 'failed')
