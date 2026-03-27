# ckpt_name="smz8599/GUI-AIMA-3B"
ckpt_name="smz8599/GUI-AIMA-3B"
save_path=""

CUDA_VISIBLE_DEVICES=0 python eval/mmbench_gui_l2.py \
    --model_name_or_path "$ckpt_name" \
    --save_path "$save_path" \
    --resize_to_pixels 5860400