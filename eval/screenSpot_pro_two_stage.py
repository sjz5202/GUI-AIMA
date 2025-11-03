import torch
import os
import json
import argparse

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor
from PIL import Image
from gui_aima.constants import chat_template
from gui_aima.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_aima.inference import inference, ForceFollowTokensLogitsProcessor
from gui_aima.utils import do_boxes_overlap
from gui_aima.constants import DEFAULT_POINTER_PAD_TOKEN, DEFAULT_POINTER_END_TOKEN
from visualization_utils import overlay_attention, save_headwise_panels,create_overlay_image,crop_subimage
from pathlib import Path
import math
from PIL import ImageDraw, ImageFont # Make sure ImageFont is imported
IMAGE_PATCH_SIZE =14

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


def process_image_for_inference(
    image: str,
    norm_pxy_center: tuple[float, float],
    grounding_system_message: str,
    instruction: str,
    model,
    tokenizer,
    data_processor,
    use_placeholder: bool,
    gt_bbox_normalized: tuple[float, float, float, float], # (norm_x1, norm_y1, norm_x2, norm_y2) from the original full image
) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    """
    Processes an image by cropping a subimage, running an inference model,
    and calculating predicted and ground truth bounding box coordinates
    relative to the cropped image.

    Args:
        image_path: Path to the input image file.
        crop_subimage_func: A callable function that takes original image dimensions and
                            the pixel center point, and returns the (start_x, start_y, end_x, end_y)
                            pixel coordinates for cropping.
                            Example signature: `crop_subimage(img_width, img_height, px, py)`
        grounding_system_message: The system message to be used in the inference conversation.
        instruction: The user instruction to be used in the inference conversation.
        inference_func: The inference function to be called.
                        It should take a conversation list, model, tokenizer, data_processor,
                        and optional arguments like logits_processor, use_placeholder, and topk.
        model: The model object required by the inference_func.
        tokenizer: The tokenizer object required by the inference_func.
        data_processor: The data processor object required by the inference_func.
        use_placeholder: A boolean flag to be passed to the inference_func.
        gt_bbox_normalized: A tuple (x1, y1, x2, y2) representing the ground truth
                            bounding box coordinates, normalized to the original image dimensions.

    Returns:
        A tuple containing two tuples:
        1. (px_in_cropped_image, py_in_cropped_image): Predicted point coordinates
           relative to the cropped image.
        2. (gt_x1_in_cropped_image, gt_y1_in_cropped_image, gt_x2_in_cropped_image, gt_y2_in_cropped_image):
           Ground truth bounding box coordinates relative to the cropped image.
    """
    img_width, img_height = image.size

    # Convert normalized center point to pixel coordinates on the original image
    px_center_orig = int(norm_pxy_center[0] * img_width)
    py_center_orig = int(norm_pxy_center[1] * img_height)

    # Determine crop coordinates and perform the crop
    crop_coords = crop_subimage(img_width, img_height, px_center_orig, py_center_orig, 448)
    start_x, start_y, end_x, end_y = crop_coords
    cropped_image = image.crop((start_x, start_y, end_x, end_y))
    cropped_img_width_ori, cropped_img_height_ori = cropped_image.size
    cropped_image = cropped_image.resize((cropped_img_width_ori*2, cropped_img_height_ori*2), Image.BICUBIC)

    # Prepare the conversation for inference
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
                    "image": cropped_image,
                },
                {
                    "type": "text",
                    "text": instruction
                },
            ],
        },
    ]

    # Perform inference
    pred = inference(conversation, model, tokenizer, data_processor, logits_processor=None, use_placeholder=use_placeholder, topk=3)
    topk_points = pred["topk_points"]
    cropped_img_width, cropped_img_height = cropped_image.size
    point_tuple_list=[]
    for (norm_px_predicted_cropped, norm_py_predicted_cropped) in topk_points:
        px_in_cropped_image = int(norm_px_predicted_cropped * cropped_img_width)
        py_in_cropped_image = int(norm_py_predicted_cropped * cropped_img_height)
        px_in_original_image = px_in_cropped_image/2 +start_x
        py_in_original_image = py_in_cropped_image/2 +start_y
        point_tuple = (px_in_original_image/img_width, py_in_original_image/img_height)
        point_tuple_list.append(point_tuple)
    
    start_x_norm = start_x / img_width
    start_y_norm = start_y / img_height
    end_x_norm   = end_x   / img_width
    end_y_norm   = end_y   / img_height

    crop_w_norm = max(end_x_norm - start_x_norm, 1e-8)
    crop_h_norm = max(end_y_norm - start_y_norm, 1e-8)

    # original GT bbox (normalized on full image)
    x1o, yo1, x2o, yo2 = gt_bbox_normalized

    # project into the cropped frame (still normalized, now w.r.t. cropped image)
    x1c = (x1o - start_x_norm) / crop_w_norm
    y1c = (yo1 - start_y_norm) / crop_h_norm
    x2c = (x2o - start_x_norm) / crop_w_norm
    y2c = (yo2 - start_y_norm) / crop_h_norm

    # clamp to [0, 1] in case the GT bbox partly lies outside the crop
    x1c = max(0.0, min(1.0, x1c))
    y1c = max(0.0, min(1.0, y1c))
    x2c = max(0.0, min(1.0, x2c))
    y2c = max(0.0, min(1.0, y2c))

    box_tuple = (x1c, y1c, x2c, y2c)
    return point_tuple_list, pred, cropped_image, box_tuple

def evaluate(model_name_or_path, model_type, data_fn, image_dir, use_placeholder, topk, resize_to_pixels=None,max_pixels=5720064,visualization_dir=None,zoom_in=False):
    # initialize model
    zoom_in_count = []
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
            attn_implementation=None
        ).eval()
        if model.config.kl_query_weighting:
            print(f"Model name: {model_name_or_path}, KL-weighting: True")
        elif model.config.query_topk is not None:
            print(f"Model name: {model_name_or_path}, KL-weighting: False, Topk: {model.config.query_topk}")
        grounding_system_message = "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, your task is to locate the screen element that corresponds to the instruction. You should output a PyAutoGUI action that performs a click on the correct position. To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>)."
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    print(f"Loaded model from {model_name_or_path}")

    # load data
    with open(data_fn, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {data_fn}")
    results = []
    for i, example in tqdm(enumerate(data), total=len(data)):
        ele = {
            "file_name": example["img_filename"],
            "ui_type": example["ui_type"],
            "group": example["group"],
            "platform": example["platform"],
            "application": example["application"],
            "id": example["id"],
            "instruction": example["instruction"],
            "img_size": example["img_size"],
            "bbox_x1y1x2y2": normalize_bbox(example["bbox"], example["img_size"][0], example["img_size"][1]),
            "hit_top1": 0,
            "overlap_top1": 0,
            "hit_topk": 0,
            "overlap_topk": 0,
        }
        
        image_path = os.path.join(image_dir, example["img_filename"])
        image = Image.open(image_path)
        ori_image = image.copy()
        
        image_width, image_height = example["img_size"]
        if (resize_to_pixels is not None) and ((image_width * image_height) != resize_to_pixels):
            resize_ratio = (resize_to_pixels / (image_width * image_height)) ** 0.5
            image_width_resized, image_height_resized = int(image_width * resize_ratio), int(image_height * resize_ratio)
            image = image.resize((image_width_resized, image_height_resized))
            ele["img_size_resized"] = [image_width_resized, image_height_resized]
        else:
            ele["img_size_resized"] = None
        
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
                        "image": image, 
                    },
                    {
                        "type": "text",
                        "text": example["instruction"]
                    },
                ],
            },
        ]

        # pred = inference(conversation, model, tokenizer, data_processor, logits_processor=logits_processor_pointer, use_placeholder=use_placeholder, topk=3)
        pred = inference(conversation, model, tokenizer, data_processor, logits_processor=None, use_placeholder=use_placeholder, topk=3)
        topk_points = pred["topk_points"]
        gt_bbox = ele["bbox_x1y1x2y2"]
        x1, y1, x2, y2 = gt_bbox

        patch_w = pred["n_width"]
        patch_h = pred["n_height"]
        IMAGE_PATCH_SIZE_x=0.5/patch_w
        IMAGE_PATCH_SIZE_y=0.5/patch_h

        if visualization_dir:
            # Original normalized coordinates
            norm_px, norm_py = topk_points[0]
            norm_x1, norm_y1, norm_x2, norm_y2 = gt_bbox

            attn_scores = pred["attn_scores"]
            
            # Create overlay image using the new function
            overlay_img = create_overlay_image(
                image, attn_scores, patch_w, patch_h, topk_points,
                norm_px, norm_py, norm_x1, norm_y1, norm_x2, norm_y2,
                example["instruction"]
            )

            vis_dir = os.path.join(visualization_dir, "vis")
            os.makedirs(vis_dir, exist_ok=True)
            stem = example["id"]
            overlay_path = os.path.join(vis_dir, f"{stem}_stage1.png")
            overlay_img.save(overlay_path)
        if zoom_in:
            # print(f"Zoom-in identification: {zoom_in}")
            new_point_list, pred, cropped_image, new_gt_bbox = process_image_for_inference(ori_image, topk_points[0], grounding_system_message, example["instruction"], model, tokenizer, data_processor, use_placeholder, gt_bbox)
            px, py = new_point_list[0]
            if (x1 <= px <= x2) and (y1 <= py <= y2):
                ele["hit_top1"] = 1
                ele["hit_topk"] = 1
            else:
                ele["hit_top1"] = 0
                ele["hit_topk"] = 0

            if visualization_dir:
                # Original normalized coordinates
                norm_px, norm_py = pred["topk_points"][0]
                norm_x1, norm_y1, norm_x2, norm_y2 = new_gt_bbox

                attn_scores = pred["attn_scores"]
                
                # Create overlay image using the new function
                overlay_img_2_stage = create_overlay_image(
                    cropped_image, attn_scores, pred["n_width"], pred["n_height"], pred["topk_points"],
                    norm_px, norm_py, norm_x1, norm_y1, norm_x2, norm_y2,
                    None
                )
                vis_dir = os.path.join(visualization_dir, "vis")
                os.makedirs(vis_dir, exist_ok=True)
                stem = example["id"]
                overlay_path_2_stage = os.path.join(vis_dir, f"{stem}_stage2.png")
                overlay_img_2_stage.save(overlay_path_2_stage)
                
            pred_bbox = [px - IMAGE_PATCH_SIZE_x, py - IMAGE_PATCH_SIZE_y, px + IMAGE_PATCH_SIZE_x, py + IMAGE_PATCH_SIZE_y]
            if do_boxes_overlap(pred_bbox, gt_bbox):
                ele["overlap_top1"] = 1
                ele["overlap_topk"] = 1
            for px, py in new_point_list[1:]:
                if (x1 <= px <= x2) and (y1 <= py <= y2):
                    ele["hit_topk"] = 1
                    pred_bbox = [px - IMAGE_PATCH_SIZE_x, py - IMAGE_PATCH_SIZE_y, px + IMAGE_PATCH_SIZE_x, py + IMAGE_PATCH_SIZE_y]
                    if do_boxes_overlap(pred_bbox, gt_bbox):
                        ele["overlap_topk"] = 1
        results.append(ele)

    return results


def get_metric(list_of_examples, 
               groups=["Dev", "Creative", "CAD", "Scientific", "Office", "OS"],
               ui_types=["text", "icon"]):
    """
    Computes metrics over a list of examples and prints/plots a table.
    
    Each element in list_of_examples is a dict containing:
        - "group": Group name (e.g., "Dev", "Creative", etc.)
        - "ui_type": UI type (e.g., "text", "icon")
        - "hit_top1", "overlap_top1", "hit_topk", "overlap_topk": binary (0 or 1)
    
    The final table has columns for each group broken down by UI type (plus a group-average)
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
    for group in groups:
        # Filter examples for the current group.
        group_examples = [ex for ex in list_of_examples if ex.get("group") == group]
        for ui in ui_types:
            # Filter further for the specific UI type.
            group_ui_examples = [ex for ex in group_examples if ex.get("ui_type") == ui]
            col_name = f"{group}-{ui}"
            for metric in metrics:
                results[metric][col_name] = compute_mean(group_ui_examples, metric)
        
        # Compute group-average (all UI types for this group).
        col_name_avg = f"{group}-avg"
        for metric in metrics:
            results[metric][col_name_avg] = compute_mean(group_examples, metric)

    # Compute overall metrics for each UI type across all groups.
    for ui in ui_types:
        ui_examples = [ex for ex in list_of_examples if ex.get("ui_type") == ui]
        col_name = f"All-{ui}"
        for metric in metrics:
            results[metric][col_name] = compute_mean(ui_examples, metric)
    
    # Compute overall average across all examples.
    overall_key = "All-avg"
    for metric in metrics:
        results[metric][overall_key] = compute_mean(list_of_examples, metric)
    
    # Define the order of columns.
    columns_order = []
    for group in groups:
        for ui in ui_types:
            columns_order.append(f"{group}-{ui}")
        columns_order.append(f"{group}-avg")
    for ui in ui_types:
        columns_order.append(f"All-{ui}")
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
    metric_info += header_tab + "\n"
    # Each row.
    for metric in metrics:
        row = [metric] + [format_cell(results[metric].get(col)) for col in columns_order]
        metric_info += ("\t".join(row) + "\n")
    print(metric_info)
    return metric_info


"""
# cd to project root directory
python eval/screenSpot_pro.py --save_path <path_to_save_results> --data_path <path_to_data>
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="qwen25vl", choices=["qwen2vl", "qwen25vl"])
    parser.add_argument("--model_name_or_path", type=str, default="smz8599/GUI-AIMA-3B-kl")
    parser.add_argument("--save_path", type=str, default="/mnt/localssd/gui_aima_res/ss_pro/$ckpt_name/two_stage/")
    parser.add_argument("--data_path", type=str, default="/mnt/localssd/gui_eval_data/ScreenSpot_pro")
    parser.add_argument("--resize_to_pixels", type=int, default=2720*1530, help="If set to <0, will not resize the image.")#default=3200*1800
    parser.add_argument("--max_pixels", type=int, default=5720064, help="If set to <0, will not resize the image.")
    parser.add_argument('--no-placeholder', dest='use_placeholder', action='store_false', help='Disable the placeholder')
    parser.add_argument('--topk', type=int, default=3, help='Topk')
    parser.add_argument("--visualization_dir", type=str, default=None, help="If set, will save the visualization images to the directory.")
    parser.set_defaults(use_placeholder=True)

    args = parser.parse_args()
    zoom_in_identification = True
    resize_to_pixels = args.resize_to_pixels if args.resize_to_pixels > 0 else None
    image_dir = os.path.join(args.data_path, "images")
    data_fn = os.path.join(args.data_path, "annotations/all.json")
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
        results = evaluate(args.model_name_or_path, args.model_type, data_fn, image_dir, args.use_placeholder, args.topk, resize_to_pixels,max_pixels=args.max_pixels, visualization_dir=args.visualization_dir,zoom_in=zoom_in_identification)
        with open(pred_path, "w") as f:
            json.dump(results, f)
        print(f"Saved {len(results)} predictions to {pred_path}")

    if not os.path.exists(metric_path):
        metric_info = get_metric(results)
        with open(metric_path, "w") as f:
            f.write(metric_info)
        print(f"Saved metric to {metric_path}")
