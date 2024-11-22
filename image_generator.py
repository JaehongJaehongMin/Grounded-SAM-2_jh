import numpy as np
import os
import sys
import argparse
import re
import urllib

from openai import OpenAI
from tqdm import tqdm


if __name__ == "__main__":

    def parse_arguments():
        parser = argparse.ArgumentParser(description="Generate images using OpenAI API")
        parser.add_argument(
            "--input_path", type=str, required=True, help="Path to the input directory"
        )
        parser.add_argument(
            "--output_path",
            type=str,
            # default="./outputs/AGD20K_dalle_3",
            required=True,
            help="Path to the output directory",
        )
        return parser.parse_args()

    args = parse_arguments()

    input_path = args.input_path
    output_path = args.output_path

    client = OpenAI()

    # system_prompt = (
    #     "Create an image that looks like a real photograph."
    #     "Feature natural human hands under soft, natural lighting for a lifelike appearance."
    #     "The hands should look genuine, with a relaxed and natural pose, as if photographed in real life."
    #     "Focus on realistic textures and lighting, capturing both the objects and the hands in an authentic way."
    #     "Aim for photorealism in every detail, creating an image that feels true to life and free of artificial qualities."
    # )
    system_prompt = (
        "Create an image that looks like a real photograph."
        "Focus on realistic textures and lighting."
        "Aim for photorealism in every detail, creating an image that feels true to life and free of artificial qualities."
    )

    for aff in tqdm(os.listdir(f"{input_path}"), desc="Affordance"):
        for obj in os.listdir(f"{input_path}/{aff}"):
            img_list = [
                f
                for f in os.listdir(f"{input_path}/{aff}/{obj}")
                if re.match(r"^\d+\.png$", f)
            ]
            for img in img_list:
                try:
                    IMG_NAME = img.split(".")[0]

                    print("Processing", input_path, aff, obj, img, "...")

                    response = client.images.generate(
                        model="dall-e-3",
                        # prompt=f"a picture of a hand {aff}ing a {obj}" + system_prompt,
                        prompt=f"Someone is {aff}ing with a {obj}" + system_prompt,
                        size="1024x1024",
                        quality="hd",
                        n=1,
                    ).data[0]

                    os.makedirs(f"{output_path}/{aff}/{obj}", exist_ok=True)

                    urllib.request.urlretrieve(
                        response.url, f"{output_path}/{aff}/{obj}/{IMG_NAME}.png"
                    )
                except Exception as e:
                    print("Error:", e, "on", aff, obj, img)
                    continue
