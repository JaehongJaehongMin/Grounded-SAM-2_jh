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
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from tqdm import tqdm

"""
Hyper parameters
"""

# parser = argparse.ArgumentParser(description="Grounded SAM2 Local Demo")
# parser.add_argument(
#     "--trg_dir",
#     type=str,
#     default="./AGD20K_dalle_3",
#     help="Target directory containing images",
# )
# parser.add_argument(
#     "--text_prompt",
#     type=str,
#     required=True,
#     help='Objects to search (ex. "hand. finger.")',
# )
# args = parser.parse_args()


input_root = "data/memory_from_stage1Prediction_2_closed_GSAM2_outputs"

# TEXT_PROMPT = "hand. finger."
# TEXT_PROMPT = args.text_prompt
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE,
)

for aff in tqdm(os.listdir(input_root)):
    for obj in os.listdir(f"{input_root}/{aff}"):
        img_list = []
        for img in os.listdir(f"{input_root}/{aff}/{obj}"):
            if img.endswith("_nohand_hand_finger_arm.png"):
                img_list.append(img)
        for img in img_list:
            print(f"{aff}/{obj}/{img}")


            TEXT_PROMPT = f"{obj}"
            IMG_NAME = os.path.splitext(img)[0] # 0_nohand_hand_finger_arm
            # IMG_PATH = f"{args.trg_dir}/{aff}/{obj}/{img}"
            IMG_PATH = f"{input_root}/{aff}/{obj}/{img}"
            # OUTPUT_DIR = f"{input_root}/{aff}/{obj}"
            OUTPUT_DIR = f"{input_root}/{aff}/{obj}"
            # DUMP_JSON_RESULTS = True
            DUMP_JSON_RESULTS = False

            # create output directory
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            # setup the input image and text prompt for SAM 2 and Grounding DINO
            # VERY important: text queries need to be lowercased + end with a dot
            text = TEXT_PROMPT
            img_path = IMG_PATH
            img_name = IMG_NAME

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

            # process the box prompt for SAM 2
            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            input_boxes = box_convert(
                boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy"
            ).numpy()

            if torch.cuda.get_device_properties(0).major >= 8:
                # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
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
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_name}_obj_predcited.png"), mask_bitmap)
                # cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_name}.png"), image_arr)
                # if args.text_prompt == "hand.":
                #     cv2.imwrite(
                #         os.path.join(OUTPUT_DIR, f"{img_name}_mask_hand.png"), mask_bitmap
                #     )
                # elif args.text_prompt == "hand. finger.":
                #     cv2. imwrite(
                #         os.path.join(OUTPUT_DIR, f"{img_name}_mask_hand_finger.png"), mask_bitmap
                #     )
                # elif args.text_prompt == "hand. finger. arm.":
                #     cv2. imwrite(
                #         os.path.join(OUTPUT_DIR, f"{img_name}_mask_hand_finger_arm.png"), mask_bitmap
                #     )
                # elif args.text_prompt == "hand. finger. arm. person. human.":
                #     cv2. imwrite(
                #         os.path.join(OUTPUT_DIR, f"{img_name}_mask_hand_finger_arm_person_human.png"), mask_bitmap
                #     )
                # else:
                #     print(f"{args.text_prompt}")
                #     print("Determine the name of the mask image")
                #     sys.exit()
                

            except Exception as e:
                print(f"Could not find the objects in the image: {img_path}")
                # image_arr = np.array(img)
                mask_bitmap = np.zeros_like(img)
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_name}_obj_predcited.png"), mask_bitmap)
            #     cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_name}.png"), image_arr)
            #     if args.text_prompt == "hand.":
            #         cv2.imwrite(
            #             os.path.join(OUTPUT_DIR, f"{img_name}_mask_hand.png"), mask_bitmap
            #         )
            #     elif args.text_prompt == "hand. finger.":
            #         cv2. imwrite(
            #             os.path.join(OUTPUT_DIR, f"{img_name}_mask_hand_finger.png"), mask_bitmap
            #         )
            #     elif args.text_prompt == "hand. finger. arm.":
            #         cv2. imwrite(
            #             os.path.join(OUTPUT_DIR, f"{img_name}_mask_hand_finger_arm.png"), mask_bitmap
            #         )
            #     elif args.text_prompt == "hand. finger. arm. person. human.":
            #         cv2. imwrite(
            #             os.path.join(OUTPUT_DIR, f"{img_name}_mask_hand_finger_arm_person_human.png"), mask_bitmap
            #         )
            #     else:
            #         print(f"{args.text_prompt}")
            #         print("Determine the name of the mask image")
            #         sys.exit()
                    
            cv2.destroyAllWindows()
