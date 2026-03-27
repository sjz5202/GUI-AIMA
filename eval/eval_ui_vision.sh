ckpt_name="smz8599/GUI-AIMA-3B"
save_path=""

CUDA_VISIBLE_DEVICES=1 python eval/ui_vision.py \
    --model_name_or_path "$ckpt_name" \
    --save_path "$save_path" \
    --resize_to_pixels 5860400
    
