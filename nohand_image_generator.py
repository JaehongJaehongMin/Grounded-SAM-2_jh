import os
import cv2
import sys
import numpy as np
import argparse
import requests
import urllib
import time
import re
from tqdm import tqdm
from openai import OpenAI

parser = argparse.ArgumentParser(
    description="Generate images without hands using OpenAI API."
)
parser.add_argument(
    "--input_path", type=str, required=True, help="Path to image directory (ex. ./AGD20K)"
)
parser.add_argument(
    "--image_size", type=str, default="1024x1024", help="Size of the generated image"
)
parser.add_argument(
    "--text_prompt",
    type=str,
    required=True,
    help='Mask prompt (ex. "hand. finger. arm")',
)
args = parser.parse_args()

input_path = args.input_path
mask_path = "./outputs/" + args.input_path
image_size = args.image_size
output_path = "./outputs/" + args.input_path

print("Input path:", input_path)
print("Mask path:", mask_path)
print("Output path:", output_path)

client = OpenAI()

if args.text_prompt == "hand.":
    text_prompt = "hand"
    print("Prompt:", text_prompt)
elif args.text_prompt == "hand. finger. arm.":
    text_prompt = "hand_finger_arm"
    print("Prompt:", text_prompt)
else:
    print("Invalid text prompt. Please check")
    sys.exit()

for aff in tqdm(os.listdir(f"{input_path}"), desc="Affordance"):
    for obj in tqdm(os.listdir(f"{input_path}/{aff}"), desc="Object"):
        img_list = [
            f
            for f in os.listdir(f"{input_path}/{aff}/{obj}")
            if re.match(r"^\d+\.png$", f)
        ]
        for img in img_list:
            print(f"Processing {input_path}, {aff}, {obj}, {img}...")
            os.makedirs(f"{output_path}/{aff}/{obj}", exist_ok=True)
            IMG_NAME = img.split(".")[0]

            # Remove hand from the image
            hand_mask = cv2.imread(f"{mask_path}/{aff}/{obj}/{IMG_NAME}_mask_{text_prompt}.png", cv2.IMREAD_GRAYSCALE)
            hand_mask = np.expand_dims(hand_mask, axis=-1)
            orig_image = cv2.imread(f'{input_path}/{aff}/{obj}/{IMG_NAME}.png')
            masked_image = np.concatenate([orig_image, ~hand_mask], axis=-1)
            cv2.imwrite(f"{output_path}/{aff}/{obj}/{IMG_NAME}_removed_{text_prompt}.png", masked_image)

            response = client.images.edit(
                model="dall-e-2",
                image=open(f"{input_path}/{aff}/{obj}/{IMG_NAME}.png", "rb"),
                mask=open(f"{output_path}/{aff}/{obj}/{IMG_NAME}_removed_{text_prompt}.png", "rb"),
                prompt=f"A picture of the {obj}",
                n=1,
                size=image_size,
            ).data[0]

            urllib.request.urlretrieve(
                response.url, f"{output_path}/{aff}/{obj}/{IMG_NAME}_nohand_{text_prompt}.png"
            )
