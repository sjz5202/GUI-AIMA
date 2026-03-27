"""
UI-Vision element grounding evaluation for GUI-AIMA models.
Evaluates all 3 subcategories: basic, functional, spatial.

Data root: /data/shijie/gui_data/ui-vision/
  annotations/element_grounding/
    element_grounding_basic.json       (1772 samples)
    element_grounding_functional.json  (1772 samples)
    element_grounding_spatial.json     (1935 samples)
  images/{image_path}

Metric: action_acc = correct / total  (point-in-bbox hit)
        broken down by element_type (icon/text) and platform.

Usage:
    python eval/ui_vision_grounding.py \
        --model_name_or_path smz8599/GUI-AIMA-3B \
        --save_path /data/shijie/gui_aima_experiments/ui_vision_grounding
"""

import json
import os
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import torch
from PIL import Image
from transformers import AutoProcessor

from gui_aima.constants import chat_template, grounding_system_message
from gui_aima.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_aima.inference import inference

DATA_ROOT   = "/data/shijie/gui_data/ui-vision"
ANN_DIR     = os.path.join(DATA_ROOT, "annotations", "element_grounding")
IMAGE_DIR   = os.path.join(DATA_ROOT, "images")

SUBCATEGORIES = ["basic", "functional", "spatial"]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_sample(ex, pred_point_norm):
    """
    Returns "correct" | "wrong" | "wrong_format".
    bbox is absolute pixels x1,y1,x2,y2 — normalize before comparison.
    """
    if pred_point_norm is None:
        return "wrong_format"

    x1, y1, x2, y2 = ex["bbox"]
    # swap if needed (mirrors UI-Vision eval_uivision.py)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    w, h = ex["image_size"]
    x1, y1, x2, y2 = x1 / w, y1 / h, x2 / w, y2 / h

    px, py = pred_point_norm
    if x1 <= px <= x2 and y1 <= py <= y2:
        return "correct"
    return "wrong"


def evaluate_subcategory(subcat, model, tokenizer, data_processor, resize_to_pixels=None):
    ann_file = os.path.join(ANN_DIR, f"element_grounding_{subcat}.json")
    with open(ann_file) as f:
        data = json.load(f)
    print(f"\n[{subcat}] {len(data)} samples")

    results = []
    for ex in tqdm(data, desc=subcat):
        img_path = os.path.join(IMAGE_DIR, ex["image_path"])
        image = Image.open(img_path).convert("RGB")
        image_width, image_height = ex["image_size"]
        if resize_to_pixels is not None and (image_width * image_height) != resize_to_pixels:
            resize_ratio = (resize_to_pixels / (image_width * image_height)) ** 0.5
            image = image.resize((int(image_width * resize_ratio), int(image_height * resize_ratio)))

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": grounding_system_message}]},
            {"role": "user",   "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  ex["prompt_to_evaluate"]},
            ]},
        ]

        pred = inference(
            conversation, model, tokenizer, data_processor,
            logits_processor=None, use_placeholder=True, topk=3
        )

        pred_point = pred["topk_points"][0] if pred["topk_points"] else None
        correctness = eval_sample(ex, pred_point)

        results.append({
            "image_path":        ex["image_path"],
            "platform":          ex["platform"],
            "element_type":      ex["element_type"],
            "category":          ex["category"],
            "prompt_to_evaluate": ex["prompt_to_evaluate"],
            "pred_point":        pred_point,
            "bbox":              ex["bbox"],
            "image_size":        ex["image_size"],
            "correctness":       correctness,
        })

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def get_metric(results):
    """
    Returns dict with overall + per element_type + per platform accuracy.
    Mirrors calc_metric_for_result_list_simple from UI-Vision.
    """
    def _acc(subset):
        if not subset:
            return {"acc": 0.0, "correct": 0, "total": 0, "wrong_format": 0}
        correct      = sum(1 for r in subset if r["correctness"] == "correct")
        wrong_format = sum(1 for r in subset if r["correctness"] == "wrong_format")
        return {
            "acc":          round(correct / len(subset) * 100, 2),
            "correct":      correct,
            "total":        len(subset),
            "wrong_format": wrong_format,
        }

    by_type     = defaultdict(list)
    by_platform = defaultdict(list)
    for r in results:
        by_type[r["element_type"]].append(r)
        by_platform[r["platform"]].append(r)

    metric = {
        "overall":     _acc(results),
        "by_type":     {t: _acc(v) for t, v in sorted(by_type.items())},
        "by_platform": {p: _acc(v) for p, v in sorted(by_platform.items())},
    }
    return metric


def print_metric(subcat, metric):
    ov = metric["overall"]
    print(f"\n  [{subcat}]  acc={ov['acc']:.2f}%  ({ov['correct']}/{ov['total']})"
          f"  wrong_format={ov['wrong_format']}")

    print(f"    By type:")
    for t, v in metric["by_type"].items():
        print(f"      {t:<8}  {v['acc']:>6.2f}%  ({v['correct']}/{v['total']})")

    print(f"    By platform (top-10 by sample count):")
    by_plat = sorted(metric["by_platform"].items(), key=lambda x: -x[1]["total"])
    for p, v in by_plat[:10]:
        print(f"      {p:<30}  {v['acc']:>6.2f}%  ({v['correct']}/{v['total']})")


def print_summary(all_metrics):
    print(f"\n=== UI-Vision Element Grounding Summary ===")
    accs = []
    for subcat in SUBCATEGORIES:
        if subcat in all_metrics:
            acc = all_metrics[subcat]["overall"]["acc"]
            accs.append(acc)
            print(f"  {subcat:<12}  {acc:.2f}%")
    if accs:
        print(f"  {'average':<12}  {np.mean(accs):.2f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="smz8599/GUI-AIMA-3B")
    parser.add_argument("--save_path",
                        default="/data/shijie/gui_aima_experiments/ui_vision_grounding")
    parser.add_argument("--max_pixels", type=int, default=5760000)
    parser.add_argument("--resize_to_pixels", type=int, default=None,
                        help="Resize images so width*height equals this value before inference. "
                             "Use -1 to disable. Default: no resize.")
    parser.add_argument("--subcategories", nargs="+", default=SUBCATEGORIES,
                        choices=SUBCATEGORIES,
                        help="Which subcategories to run (default: all 3).")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    model_tag = Path(args.model_name_or_path).name

    # Load model once
    data_processor = AutoProcessor.from_pretrained(
        args.model_name_or_path, max_pixels=args.max_pixels
    )
    tokenizer = data_processor.tokenizer

    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    ).eval()
    print(f"Loaded model: {args.model_name_or_path}")

    all_metrics = {}

    for subcat in args.subcategories:
        pred_path   = os.path.join(args.save_path, f"{model_tag}_{subcat}_preds.json")
        metric_path = os.path.join(args.save_path, f"{model_tag}_{subcat}_metric.json")

        if os.path.exists(pred_path):
            print(f"Loading cached predictions: {pred_path}")
            with open(pred_path) as f:
                results = json.load(f)
        else:
            resize_to_pixels = args.resize_to_pixels if (args.resize_to_pixels is not None and args.resize_to_pixels > 0) else None
            results = evaluate_subcategory(subcat, model, tokenizer, data_processor, resize_to_pixels=resize_to_pixels)
            with open(pred_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved {len(results)} predictions → {pred_path}")

        metric = get_metric(results)
        with open(metric_path, "w") as f:
            json.dump(metric, f, indent=2)

        all_metrics[subcat] = metric
        print_metric(subcat, metric)

    print_summary(all_metrics)

    summary_path = os.path.join(args.save_path, f"{model_tag}_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model_name_or_path,
            "summary": {s: all_metrics[s]["overall"] for s in all_metrics},
            "average_acc": round(np.mean([all_metrics[s]["overall"]["acc"]
                                          for s in all_metrics]), 2),
        }, f, indent=2)
    print(f"\nSaved summary → {summary_path}")