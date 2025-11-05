# kl weighting_method="query_kl_part"
weighting_method="query_1"
JOB_NAME=gui_aima_${weighting_method}_3b
max_pixels_setting=5860400
data_config_dir="data/data_config.yaml"
model_type="qwen25vl"
llm_model="Qwen/Qwen2.5-VL-3B-Instruct"
output_dir="/mnt/localssd/gui-aima-ckpt/${JOB_NAME}"
# -1: disable empty cache
empty_cache_interval=-1

# === Training Command ===
torchrun --nnodes=${NUM_NODES} \
  --node_rank=${NODE_RANK} \
  --nproc_per_node=8 \
  --master_addr=${MASTER_ADDR} train.py \
  --deepspeed ./scripts/zero3.json \
  --data_path ${data_config_dir} \
  --image_folder "" \
  --model_type ${model_type} \
  --model_name_or_path ${llm_model} \
  --save_total_limit 1 \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir ${output_dir} \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --learning_rate 5e-6 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --tf32 True \
  --model_max_length 24576 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --max_pixels ${max_pixels_setting} \
  --unfreeze_all_parameters True \
  --unfreeze_pointer_head False \
  --unfreeze_lm_head False \
  --unfreeze_base_model False \
  --unfreeze_last_n_layers -1 \
  --unfreeze_new_tokens False \
  --unfreeze_visual False \
  --pointer_loss_weight 1.0 \
  --lm_loss_weight 1.0 \
  --empty_cache_every_n_steps ${empty_cache_interval} \
  --weighting ${weighting_method}