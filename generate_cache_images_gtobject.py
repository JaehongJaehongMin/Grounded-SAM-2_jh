import os
import json
import urllib.request
from openai import OpenAI
import argparse
from tqdm import tqdm, trange

import cv2
import torch
import pathlib
import numpy as np
from torchvision import transforms

import time
import sys
# sys.path.append(os.path.join(os.environ['OSPREY_ROOT']))

# from semantic_segmentation import models
# from semantic_segmentation import load_model
# from semantic_segmentation import draw_results

affordance2gen_prompt={
    'carry' : 'A picture of a {} and a hand carrying the {}',
    'catch' : 'A picture of a {} and a hand catching the {}',
    'hold' : 'A picture of a {} and a hand holding the {}',
    'take_photo' : 'A picture of a {} and a hand taking a photo with the {}',
    'type_on' : 'A picture of a {} and a hand typing on the {}',
    'swing' : 'A picture of a {} and a hand swinging the {}',
    'pour' : 'A picture of a {} and a hand pouring the {}',
    'pick_up' : 'A picture of a {} and a hand picking up the {}',
    'hit' : 'A picture of a {} and a hand hitting the {}',
    'look_out' : 'A picture of a {} and a hand looking out with the {}',
    'lift' : 'A picture of a {} and a hand lifting the {}',
    'write' : 'A picture of a {} and a hand writing the {}',
    'text_on' : 'A picture of a {} and a hand texting on the {}'
}



def _load_image(image_path: pathlib.Path):
    image = cv2.imread(str(image_path))
    assert image is not None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_width = (image.shape[1] // 32) * 32
    image_height = (image.shape[0] // 32) * 32

    image = image[:image_height, :image_width]
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='osprey demo', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--image_size', help='image size dalle generates', default='1024x1024')
    parser.add_argument('--cache_path', help='path to generated image', default='cache')
    parser.add_argument('--skin_seg_threshold', help='threshold used in skin segmentation model', default=0.5)

    args = parser.parse_args()

    image_size = args.image_size
    cache_path = args.cache_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # pretrained_model_path = '../semantic_segmentation/pretrained/model_segmentation_skin_30.pth'
    # hand_segmentation_model = torch.load(pretrained_model_path, map_location=device)
    # hand_segmentation_model = load_model(models['FCNResNet101'], hand_segmentation_model)
    # hand_segmentation_model.to(device).eval()

    # fn_image_transform = transforms.Compose([
    #     transforms.Lambda(lambda image_path: _load_image(image_path)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # ])

    TEXT_PROMPT = "hand"
    SAM2_CHECKPOINT = "/workspace/ObjPart2DallEImageEditing/Grounded_SAM2/checkpoints/sam2.1_hiera_large.pt"
    SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    GROUNDING_DINO_CONFIG = "/workspace/ObjPart2DallEImageEditing/Grounded_SAM2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "/workspace/ObjPart2DallEImageEditing/Grounded_SAM2/gdino_checkpoints/groundingdino_swint_ogc.pth"
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


    client = OpenAI()


    os.makedirs(cache_path, exist_ok=True)
    aff_obj_dict = json.load(open('affordance_object_dict.json'))

    for affordance_type , obj_list in tqdm(aff_obj_dict.items(), desc = 'aff'):
        # if 'hold' in affordance_type:
        #     continue
        os.makedirs(f'{cache_path}/{affordance_type}', exist_ok=True)
        # for object_type in phase1_dict[affordance_type].keys():
        for object_type in tqdm(obj_list, desc='obj'):
            os.makedirs(f'{cache_path}/{affordance_type}/{object_type}', exist_ok=True)
            # for i in trange(9):
            i=0
            while i<9:
                try:
                    #####################################
                    # Hand-Object Image Generation
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
                    time.sleep(10)
                    #####################################

                    #####################################
                    # Hand segmentation
                    #####################################
                    # image = fn_image_transform(f'{cache_path}/{affordance_type}/{object_type}/{i+1}.png')
                    # with torch.no_grad():
                    #     image = image.to(device).unsqueeze(0)
                    #     results = hand_segmentation_model(image)['out']
                    #     results = torch.sigmoid(results)

                    #     results = results > args.skin_seg_threshold
                    # hand_mask = results[0][0].detach().cpu().numpy().astype(np.uint8)[...,None] * 255
                    # cv2.imwrite(f"{cache_path}/{affordance_type}/{object_type}/{i+1}_handmask.png", hand_mask)

                    # orig_image = cv2.imread(f'{cache_path}/{affordance_type}/{object_type}/{i+1}.png')
                    # masked_image = np.concatenate([orig_image, ~hand_mask], axis=-1)

                    # cv2.imwrite(f"{cache_path}/{affordance_type}/{object_type}/{i+1}_mask.png", masked_image)
                    #####################################

                    #####################################
                    # Hand segmentation (Grounded-SAM2)
                    text = TEXT_PROMPT
                    img_path = f'{cache_path}/{affordance_type}/{object_type}/{i+1}.png'
                    img_name = f"{i+1}.png"

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
                        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_name}.png"), image_arr)
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
                        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_name}.png"), image_arr)
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
                    #####################################


                    #####################################
                    # Remove hand
                    #####################################
                    response = client.images.edit(
                        model='dall-e-2',
                        image=open(f"{cache_path}/{affordance_type}/{object_type}/{i+1}.png", "rb"),
                        mask=open(f"{cache_path}/{affordance_type}/{object_type}/{i+1}_mask.png", "rb"),
                        prompt=f"A picture of the {object_type}",
                        n=1,
                        size=image_size
                    ).data[0]
                    urllib.request.urlretrieve(
                        response.url,
                        f'{cache_path}/{affordance_type}/{object_type}/{i+1}_nohand.png'
                    )
                    time.sleep(10)
                    i+=1
                except Exception as e:
                    print(e)
                    print(affordance_type, object_type, i, 'failed')
