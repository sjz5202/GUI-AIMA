# smz8599/GUI-AIMA-3B smz8599/GUI-AIMA-3B-kl
ckpt_name="smz8599/GUI-AIMA-3B"
out_ss_pro=""
CUDA_VISIBLE_DEVICES=0 python eval/screenSpot_pro.py --model_name_or_path "$ckpt_name" --save_path "$out_ss_pro" # --visualization_dir $visualization_dir

#two stage zoom in
out_ss_pro_two_stage=""
CUDA_VISIBLE_DEVICES=0 python eval/screenSpot_pro_two_stage.py --model_name_or_path "$ckpt_name" --save_path "$out_ss_pro_two_stage" # --visualization_dir $visualization_dir