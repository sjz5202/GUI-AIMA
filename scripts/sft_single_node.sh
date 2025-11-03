#!/bin/bash
model_type="qwen25vl"
llm_model="Qwen/Qwen2.5-VL-3B-Instruct"

# kl weighting_method="query_kl_part"
weighting_method="query_1"
name="gui_aima_${weighting_method}_3b"
output_dir="/mnt/localssd/gui_aima_final_ckpt/${name}"
max_pixels=4014080
# === Training Command ===
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29500 train.py \
  --deepspeed ./scripts/zero3.json \
  --data_path data/data_config.yaml \
  --image_folder "" \
  --model_type ${model_type} \
  --model_name_or_path ${llm_model} \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir ${output_dir} \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1 \
  --save_total_limit 1 \
  --learning_rate 5e-6 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --tf32 True \
  --model_max_length 24576 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --max_pixels ${max_pixels} \
  --unfreeze_all_parameters True \
  --unfreeze_pointer_head False \
  --unfreeze_lm_head False \
  --unfreeze_base_model False \
  --unfreeze_last_n_layers -1 \
  --unfreeze_new_tokens False \
  --unfreeze_visual False \
  --pointer_loss_weight 1.0 \
  --lm_loss_weight 1.0 \
  --empty_cache_every_n_steps 10 \
  --number_of_points 1 \
  --weighting ${weighting_method}