# smz8599/GUI-AIMA-3B smz8599/GUI-AIMA-3B-kl
ckpt_name="smz8599/GUI-AIMA-3B"
out_ss_v2="/mnt/localssd/gui_aima_res/ss_v2/GUI-AIMA-3B/"
# visualization_dir =
CUDA_VISIBLE_DEVICES=0 python eval/screenSpot_v2.py --model_name_or_path "$ckpt_name" --save_path "$out_ss_v2" # --visualization_dir $visualization_dir