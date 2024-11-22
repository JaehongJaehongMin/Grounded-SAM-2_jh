export CUDA_HOME=/usr/local/cuda-12.1/
export CUDA_VISIBLE_DEVICES=7

# TARGET_DIR="assets_jh"
# for file in "$TARGET_DIR"/*; do
#     img_name=$(basename "$file")
#     python grounded_sam2_local_demo.py \
#     --trg_dir="assests_jh" \
#     --text_prompt="hand. finger." \
#     --img_name="$img_name"
# done


python grounded_sam2_local_demo_jh.py \
    --trg_dir="AGD20K_2" \
    --text_prompt="hand. finger. arm."
