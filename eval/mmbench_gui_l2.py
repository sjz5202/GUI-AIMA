"""
MMBench-GUI L2 (Element Grounding) evaluation for GUI-AIMA models.

Data:  /data/shijie/gui_data/MMBench-GUI/L2_annotations.json
       /data/shijie/gui_data/MMBench-GUI/MMBench-GUI-OfflineImages.zip
         └── offline_images/{platform}/{image_path}

Metrics (mirror of benchmarks/matrics.py::level2_calculate_scores):
  Per grounding_type (basic / advanced) × platform × data_type (icon / text).
  Weighted-average Basic / Advanced / Overall accuracy.

Usage:
    python eval/mmbench_gui_l2.py \
        --model_name_or_path smz8599/GUI-AIMA-3B \
        --save_path /data/shijie/gui_aima_experiments/mmbench_gui_l2
"""

import io
import json
import os
import zipfile
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

DATA_ROOT    = "/data/shijie/gui_data/MMBench-GUI"
ANN_FILE     = os.path.join(DATA_ROOT, "L2_annotations.json")
ZIP_FILE     = os.path.join(DATA_ROOT, "MMBench-GUI-OfflineImages.zip")
OFFLINE_DIR  = os.path.join(DATA_ROOT, "offline_images")   # extracted dir (if exists)


# ---------------------------------------------------------------------------
# Image loader — prefers extracted directory, falls back to zip
# ---------------------------------------------------------------------------

class ImageLoader:
    def __init__(self):
        self._zf = None
        if not os.path.isdir(OFFLINE_DIR):
            self._zf = zipfile.ZipFile(ZIP_FILE, "r")

    def load(self, platform, image_path):
        rel = f"offline_images/{platform}/{image_path}"
        if self._zf is not None:
            data = self._zf.read(rel)
            return Image.open(io.BytesIO(data)).convert("RGB")
        else:
            return Image.open(os.path.join(OFFLINE_DIR, platform, image_path)).convert("RGB")

    def close(self):
        if self._zf is not None:
            self._zf.close()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model_name_or_path, max_pixels=5760000, resize_to_pixels=None):
    data_processor = AutoProcessor.from_pretrained(model_name_or_path, max_pixels=max_pixels)
    tokenizer = data_processor.tokenizer

    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    ).eval()
    print(f"Loaded model: {model_name_or_path}")

    with open(ANN_FILE) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} L2 examples.")

    loader = ImageLoader()
    results = []

    for ex in tqdm(data):
        image = loader.load(ex["platform"], ex["image_path"])
        image_width, image_height = ex["image_size"]
        if resize_to_pixels is not None and (image_width * image_height) != resize_to_pixels:
            resize_ratio = (resize_to_pixels / (image_width * image_height)) ** 0.5
            image = image.resize((int(image_width * resize_ratio), int(image_height * resize_ratio)))

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": grounding_system_message}]},
            {"role": "user",   "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  ex["instruction"]},
            ]},
        ]

        pred = inference(
            conversation, model, tokenizer, data_processor,
            logits_processor=None, use_placeholder=True, topk=3
        )

        # bbox is already normalised [0,1] as x1,y1,x2,y2
        x1, y1, x2, y2 = ex["bbox"]
        hit = 0
        if pred["topk_points"]:
            px, py = pred["topk_points"][0]
            if x1 <= px <= x2 and y1 <= py <= y2:
                hit = 1

        results.append({
            "index":          ex["index"],
            "platform":       ex["platform"],
            "app_name":       ex["app_name"],
            "data_type":      ex["data_type"],         # icon | text
            "grounding_type": ex["grounding_type"],    # basic | advanced
            "instruction":    ex["instruction"],
            "bbox":           ex["bbox"],
            "pred_point":     pred["topk_points"][0] if pred["topk_points"] else None,
            "hit":            hit,
        })

    loader.close()
    return results


# ---------------------------------------------------------------------------
# Metrics  (mirrors level2_calculate_scores from MMBench-GUI/benchmarks/matrics.py)
# ---------------------------------------------------------------------------

def get_metric(results):
    # stats[grounding_type][platform][data_type] = list of 0/1
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in results:
        stats[r["grounding_type"]][r["platform"]][r["data_type"]].append(r["hit"])

    score_dict = {}
    for gtype, platforms in stats.items():
        score_dict[gtype] = {}
        for platform, dtype_dict in platforms.items():
            icon_hits  = dtype_dict.get("icon", [])
            text_hits  = dtype_dict.get("text", [])
            total_hits = icon_hits + text_hits

            icon_acc  = np.mean(icon_hits)  * 100 if icon_hits  else 0.0
            text_acc  = np.mean(text_hits)  * 100 if text_hits  else 0.0
            total_acc = np.mean(total_hits) * 100 if total_hits else 0.0

            score_dict[gtype][platform] = {
                "Total num":       len(total_hits),
                "Icon num":        len(icon_hits),
                "Text num":        len(text_hits),
                "Total accuracy":  round(total_acc, 2),
                "Icon accuracy":   round(icon_acc,  2),
                "Text accuracy":   round(text_acc,  2),
            }

    # Weighted summary (weighted by platform sample count)
    def weighted_avg(gtype_dict):
        nums   = [v["Total num"]      for v in gtype_dict.values()]
        scores = [v["Total accuracy"] for v in gtype_dict.values()]
        total  = sum(nums)
        return sum(s * n / total for s, n in zip(scores, nums)) if total else 0.0

    basic_avg    = weighted_avg(score_dict.get("basic",    {}))
    advanced_avg = weighted_avg(score_dict.get("advanced", {}))
    overall_avg  = np.mean([basic_avg, advanced_avg])

    summary = {
        "Average accuracy":  round(overall_avg,  2),
        "Basic accuracy":    round(basic_avg,    2),
        "Advanced accuracy": round(advanced_avg, 2),
    }
    summary.update(score_dict)
    return summary


def print_metric(metric):
    print(f"\n=== MMBench-GUI L2 Results ===")
    print(f"  Average accuracy : {metric['Average accuracy']:.2f}%")
    print(f"  Basic accuracy   : {metric['Basic accuracy']:.2f}%")
    print(f"  Advanced accuracy: {metric['Advanced accuracy']:.2f}%")

    for gtype in ["basic", "advanced"]:
        if gtype not in metric:
            continue
        print(f"\n  [{gtype}]")
        header = f"  {'Platform':<18} {'Total':>6} {'Total%':>7} {'Icon%':>7} {'Text%':>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for platform, v in sorted(metric[gtype].items()):
            print(f"  {platform:<18} {v['Total num']:>6} "
                  f"{v['Total accuracy']:>7.2f} "
                  f"{v['Icon accuracy']:>7.2f} "
                  f"{v['Text accuracy']:>7.2f}")

    # Tab-delimited for copy-paste
    print("\n  Tab-delimited:")
    cols = ["platform", "Total num", "Total accuracy", "Icon accuracy", "Text accuracy"]
    for gtype in ["basic", "advanced"]:
        if gtype not in metric:
            continue
        print(f"  [[ {gtype} ]]")
        print("  " + "\t".join(cols))
        for platform, v in sorted(metric[gtype].items()):
            row = [platform, str(v["Total num"]),
                   f"{v['Total accuracy']:.2f}",
                   f"{v['Icon accuracy']:.2f}",
                   f"{v['Text accuracy']:.2f}"]
            print("  " + "\t".join(row))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="smz8599/GUI-AIMA-3B")
    parser.add_argument("--save_path",
                        default="/data/shijie/gui_aima_experiments/mmbench_gui_l2")
    parser.add_argument("--max_pixels", type=int, default=5760000)
    parser.add_argument("--resize_to_pixels", type=int, default=None,
                        help="Resize images so width*height equals this value before inference. "
                             "Use -1 to disable. Default: no resize.")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    model_tag = Path(args.model_name_or_path).name

    pred_path   = os.path.join(args.save_path, f"{model_tag}_preds.json")
    metric_path = os.path.join(args.save_path, f"{model_tag}_metric.json")

    if os.path.exists(metric_path):
        print(f"Already done: {metric_path}")
        with open(metric_path) as f:
            print_metric(json.load(f))
        exit()

    if os.path.exists(pred_path):
        print(f"Loading predictions from {pred_path}")
        with open(pred_path) as f:
            results = json.load(f)
    else:
        resize_to_pixels = args.resize_to_pixels if (args.resize_to_pixels is not None and args.resize_to_pixels > 0) else None
        results = evaluate(args.model_name_or_path, max_pixels=args.max_pixels, resize_to_pixels=resize_to_pixels)
        with open(pred_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} predictions → {pred_path}")

    metric = get_metric(results)
    with open(metric_path, "w") as f:
        json.dump(metric, f, indent=2)
    print(f"Saved metric → {metric_path}")
    print_metric(metric)
