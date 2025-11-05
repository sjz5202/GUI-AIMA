import torch
import os
import json
import argparse
import math
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor

from gui_aima.constants import chat_template
from gui_aima.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_aima.inference import inference, ForceFollowTokensLogitsProcessor
from gui_aima.utils import do_boxes_overlap
from gui_aima.constants import DEFAULT_POINTER_PAD_TOKEN, DEFAULT_POINTER_END_TOKEN
from pathlib import Path
IMAGE_PATCH_SIZE =14
from gui_aima.dataset import get_multi_patch_labels    
from visualization_utils import overlay_attention, save_headwise_panels,create_overlay_image

def normalize_bbox(bbox_x1y1x2y2, img_width, img_height):
    # if bbox_x1y1x2y2 is not normalized to [0, 1], normalize it
    x1, y1, x2, y2 = bbox_x1y1x2y2
    if (0 <= x1 <= 1) and (0 <= y1 <= 1) and (0 <= x2 <= 1) and (0 <= y2 <= 1):
        return bbox_x1y1x2y2
    else:
        x1 = x1 / img_width
        y1 = y1 / img_height
        x2 = x2 / img_width
        y2 = y2 / img_height
        return x1, y1, x2, y2

def evaluate(model_name_or_path, model_type, use_placeholder, topk,max_pixels=5720064,visualization_dir=None):
    # initialize model
    data_processor = AutoProcessor.from_pretrained(model_name_or_path,max_pixels=max_pixels)
    tokenizer = data_processor.tokenizer
    for k, v in tokenizer.added_tokens_encoder.items():
        print(v, k)

    if model_type == "qwen25vl":
        print(f"Loading model with Qwen2.5-VL backbone from {model_name_or_path}")
        model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            # attn_implementation=None
            attn_implementation="flash_attention_2"
        ).eval()
        if model.config.kl_query_weighting:
            print(f"Model name: {model_name_or_path}, KL-weighting: True")
        elif model.config.query_topk is not None:
            print(f"Model name: {model_name_or_path}, KL-weighting: False, Topk: {model.config.query_topk}")
        grounding_system_message = "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, your task is to locate the screen element that corresponds to the instruction. You should output a PyAutoGUI action that performs a click on the correct position. To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>)."
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    print(f"Loaded model from {model_name_or_path}")

    dataset = load_dataset("HongxinLi/ScreenSpot_v2")["test"]
    domain_dict = {
            "windows": "desktop",
            "macos": "desktop",
            "ios": "mobile",
            "android": "mobile",
            "tool": "web",
            "shop": "web",
            "gitlab": "web",
            "forum": "web"
        }

    results = []
    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        ele = {
            "file_name": example["file_name"],
            "data_type": example["data_type"],
            "domain": domain_dict[example["data_source"]],
            "instruction": example["instruction"],
            "img_size": example["image"].size,
            "bbox_x1y1x2y2": normalize_bbox(example["bbox"], example["image"].size[0], example["image"].size[1]),
            "hit_top1": 0,
            "overlap_top1": 0,
            "hit_topk": 0,
            "overlap_topk": 0,
        }

        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": grounding_system_message,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": example["image"], # PIL.Image.Image or str to path
                        # "image_url": "https://xxxxx.png" or "https://xxxxx.jpg" or "file://xxxxx.png" or "data:image/png;base64,xxxxxxxx", will be split by "base64,"
                    },
                    {
                        "type": "text",
                        "text": example["instruction"]
                    },
                ],
            },
        ]

        pred = inference(conversation, model, tokenizer, data_processor, logits_processor=None, use_placeholder=use_placeholder, topk=3)
        topk_points = pred["topk_points"]
        gt_bbox = ele["bbox_x1y1x2y2"]
        if visualization_dir is not None:
            attn_scores = pred["attn_scores"]
            patch_w=pred["n_width"]
            patch_h=pred["n_height"]
            vis_img = example["image"]  
            overlay_img = create_overlay_image(
                vis_img, attn_scores, patch_w, patch_h, topk_points,
                topk_points[0][0], topk_points[0][1], gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3],
                example["instruction"]
            )
            vis_dir = os.path.join(visualization_dir, "vis")
            os.makedirs(vis_dir, exist_ok=True)
            stem = Path(example["file_name"]).stem
            overlay_path = os.path.join(vis_dir, f"{stem}_overlay_mean.png")
            overlay_img.save(overlay_path)
        # compute the metrics
        px, py = topk_points[0]
        x1, y1, x2, y2 = gt_bbox

        if (x1 <= px <= x2) and (y1 <= py <= y2):
            ele["hit_top1"] = 1
            ele["hit_topk"] = 1

        patch_w = pred["n_width"]
        patch_h = pred["n_height"]
        IMAGE_PATCH_SIZE_x=0.5/patch_w
        IMAGE_PATCH_SIZE_y=0.5/patch_h
        pred_bbox = [px - IMAGE_PATCH_SIZE_x, py - IMAGE_PATCH_SIZE_y, px + IMAGE_PATCH_SIZE_x, py + IMAGE_PATCH_SIZE_y]
        if do_boxes_overlap(pred_bbox, gt_bbox):
            ele["overlap_top1"] = 1
            ele["overlap_topk"] = 1

        for px, py in topk_points[1:]:
            if (x1 <= px <= x2) and (y1 <= py <= y2):
                ele["hit_topk"] = 1
            pred_bbox = [px - IMAGE_PATCH_SIZE_x, py - IMAGE_PATCH_SIZE_y, px + IMAGE_PATCH_SIZE_x, py + IMAGE_PATCH_SIZE_y]
            if do_boxes_overlap(pred_bbox, gt_bbox):
                ele["overlap_topk"] = 1

        results.append(ele)
    
    return results


def get_metric(list_of_examples, 
               domains=["mobile", "desktop", "web"],
               data_types=["text", "icon"]):
    """
    Computes metrics over a list of examples and prints/plots a table.
    
    Each element in list_of_examples is a dict containing:
        - "domain": Domain name (e.g., "web", "mobile", "desktop")
        - "data_type": Data type (e.g., "text", "icon")
        - "hit_top1", "overlap_top1", "hit_topk", "overlap_topk": binary (0 or 1)
    
    The final table has columns for each domain broken down by UI type (plus a domain-average)
    and overall columns ("All-text", "All-icon", "All-average").
    
    The rows of the table are:
        - hit_top1
        - overlap_top1
        - hit_topk
        - overlap_topk
    """
    
    # List of metric keys to compute.
    metrics = ["hit_top1", "overlap_top1", "hit_topk", "overlap_topk"]

    # Helper function to compute the mean of a given key from a list of examples.
    def compute_mean(examples, key):
        if not examples:
            return None
        return sum(example.get(key, 0) for example in examples) / len(examples)

    # Prepare results dictionary: structure {metric: {column_name: value}}.
    results = {metric: {} for metric in metrics}
    
    # Compute metrics for each group broken down by UI type.
    for domain in domains:
        # Filter examples for the current group.
        domain_examples = [ex for ex in list_of_examples if ex.get("domain") == domain]
        for data_type in data_types:
            # Filter further for the specific UI type.
            domain_data_type_examples = [ex for ex in domain_examples if ex.get("data_type") == data_type]
            col_name = f"{domain}-{data_type}"
            for metric in metrics:
                results[metric][col_name] = compute_mean(domain_data_type_examples, metric)
        
        # Compute domain-average (all UI types for this domain).
        col_name_avg = f"{domain}-avg"
        for metric in metrics:
            results[metric][col_name_avg] = compute_mean(domain_examples, metric)

    # Compute overall metrics for each UI type across all domains.
    for data_type in data_types:
        data_type_examples = [ex for ex in list_of_examples if ex.get("data_type") == data_type]
        col_name = f"All-{data_type}"
        for metric in metrics:
            results[metric][col_name] = compute_mean(data_type_examples, metric)
    
    # Compute overall average across all examples.
    overall_key = "All-avg"
    for metric in metrics:
        results[metric][overall_key] = compute_mean(list_of_examples, metric)
    
    # Define the order of columns.
    columns_order = []
    for domain in domains:
        for data_type in data_types:
            columns_order.append(f"{domain}-{data_type}")
        columns_order.append(f"{domain}-avg")
    for data_type in data_types:
        columns_order.append(f"All-{data_type}")
    columns_order.append("All-avg")
    
    # ------------- Print Table to Console -------------
    # Prepare header row.
    header = [""] + columns_order
    # Calculate column widths for console printing.
    col_widths = [max(len(col), 12) for col in header]
    
    def format_cell(cell):
        if isinstance(cell, float):
            return f"{cell*100:.2f}"
        elif cell is None:
            return "N/A"
        return str(cell)
    
    # Print header.
    header_line = " | ".join(word.ljust(width) for word, width in zip(header, col_widths))
    separator_line = "-+-".join("-" * width for width in col_widths)
    print(header_line)
    print(separator_line)
    
    for metric in metrics:
        row = [metric]
        for col in columns_order:
            val = results[metric].get(col)
            row.append(format_cell(val))
        row_line = " | ".join(word.ljust(width) for word, width in zip(row, col_widths))
        print(row_line)
    
    # ------------- Print Tab-delimited Version (for Excel Copy-Paste) -------------
    metric_info = "Tab-delimited Table for Excel:\n"
    # Header row.
    header_tab = "\t".join([""] + columns_order)
    metric_info += (header_tab + "\n")
    # Each row.
    for metric in metrics:
        row = [metric] + [format_cell(results[metric].get(col)) for col in columns_order]
        metric_info += ("\t".join(row) + "\n")
    print(metric_info)
    return metric_info


"""
# cd to project root directory
python eval/screenSpot_v2.py --save_path <path_to_save_results>
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="qwen25vl", choices=["qwen2vl", "qwen25vl"])
    parser.add_argument("--model_name_or_path", type=str, default="smz8599/GUI-AIMA-3B")
    parser.add_argument("--save_path", type=str, default="/mnt/localssd/gui_aima_res/ssp/results_v2")
    parser.add_argument('--topk', type=int, default=3, help='Topk')
    parser.add_argument('--no-placeholder', dest='use_placeholder', action='store_false', help='Disable the placeholder')
    parser.add_argument("--max_pixels", type=int, default=5760000, help="If set to <0, will not resize the image.")
    parser.add_argument("--visualization_dir", type=str, default=None, help="If set, will save the visualization images to the directory.")
    parser.set_defaults(use_placeholder=True)

    args = parser.parse_args()

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    json_path=os.path.join(save_path, "json")
    if not os.path.exists(json_path):
        os.makedirs(json_path, exist_ok=True)
    last_path=Path(args.model_name_or_path).name
    pred_path = f"{save_path}/json/{last_path}_preds.json"
    metric_path = f"{save_path}/{last_path}_metric.txt"

    if os.path.exists(metric_path):
        exit()

    if os.path.exists(pred_path):
        print(f"Loading predictions from {pred_path}")
        with open(pred_path, "r") as f:
            results = json.load(f)
    else:
        print(f"Evaluating {args.model_name_or_path}...")
        results = evaluate(args.model_name_or_path, args.model_type, args.use_placeholder, args.topk, max_pixels=args.max_pixels, visualization_dir=args.visualization_dir)
        with open(pred_path, "w") as f:
            json.dump(results, f)
        print(f"Saved {len(results)} predictions to {pred_path}")

    if not os.path.exists(metric_path):
        metric_info = get_metric(results)
        with open(metric_path, "w") as f:
            f.write(metric_info)
        print(f"Saved metric to {metric_path}")
