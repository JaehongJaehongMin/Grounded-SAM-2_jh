export CUDA_HOME=/usr/local/cuda-12.1/
export CUDA_VISIBLE_DEVICES=7


# python nohand_image_generator.py \
#     --input_path AGD20K_dalle_3 \
#     --mask_path outputs/AGD20K_dalle_3 \
#     --text_prompt "hand. finger. arm." \
#     --output_path outputs/AGD20K_dalle_3

python nohand_image_generator.py \
    --input_path AGD20K_2 \
    --text_prompt "hand."
