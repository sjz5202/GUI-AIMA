import torch

from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from transformers import AutoProcessor
from gui_aima.constants import chat_template
from gui_aima.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_aima.inference import inference


# load model
model_name_or_path = "smz8599/GUI-AIMA-3B"
data_processor = AutoProcessor.from_pretrained(model_name_or_path,max_pixels=5760000)
tokenizer = data_processor.tokenizer
model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation=None
    attn_implementation="flash_attention_2",
).eval()

# prepare example
dataset = load_dataset("HongxinLi/ScreenSpot_v2")["test"]
example = dataset[0]
print(f"Intruction: {example['instruction']}")
print(f"ground-truth action region (x1, y1, x2, y2): {[round(i, 2) for i in example['bbox']]}")

conversation = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, your task is to locate the screen element that corresponds to the instruction. You should output a PyAutoGUI action that performs a click on the correct position. To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>).",
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

# inference
pred = inference(conversation, model, tokenizer, data_processor, use_placeholder=True, topk=3)
px, py = pred["topk_points"][0]
print(f"Predicted click point: [{round(px, 4)}, {round(py, 4)}]")

# >> Model Response
# Intruction: close this window
# ground-truth action region (x1, y1, x2, y2): [0.9479, 0.1444, 0.9938, 0.2074]
# Predicted click point: [0.9667, 0.1842]