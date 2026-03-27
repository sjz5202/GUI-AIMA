# smz8599/GUI-AIMA-3B smz8599/GUI-AIMA-3B-kl
ckpt_name="smz8599/GUI-AIMA-3B"
data_path="/OSWorld-G/benchmark"
out_osworld_g=""
# visualization_dir =
CUDA_VISIBLE_DEVICES=0 python eval/osworld-g.py --model_name_or_path "$ckpt_name" --save_path "$out_osworld_g" --data_path "$data_path" # --visualization_dir $visualization_dir

# two stage zoom in
out_osworld_g_two_stage=""
CUDA_VISIBLE_DEVICES=3 python eval/osworld-g_two_stage.py --model_name_or_path "$ckpt_name" --save_path "$out_osworld_g_two_stage" --data_path "/home/csgrad/shijiezh/doc_gui/gui_eval/OSWorld-G/benchmark" # --visualization_dir $visualization_dir