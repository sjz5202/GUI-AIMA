# smz8599/GUI-AIMA-3B smz8599/GUI-AIMA-3B-kl
ckpt_name="smz8599/GUI-AIMA-3B-kl"
out_ss_pro="/mnt/localssd/gui_aima_res/ss_pro/GUI-AIMA-3B-kl/"
# visualization_dir =
CUDA_VISIBLE_DEVICES=0 python eval/screenSpot_pro.py --model_name_or_path "$ckpt_name" --save_path "$out_ss_pro" # --visualization_dir $visualization_dir

#two stage zoom in
out_ss_pro_two_stage="/mnt/localssd/gui_aima_res/ss_pro/GUI-AIMA-3B-kl/two_stage"
# visualization_dir =
CUDA_VISIBLE_DEVICES=1 python eval/screenSpot_pro_two_stage.py --model_name_or_path "$ckpt_name" --save_path "$out_ss_pro_two_stage" # --visualization_dir $visualization_dir