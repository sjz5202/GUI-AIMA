import torch
import json
import re
import os
from qwen_vl_utils import process_vision_info
from transformers import (
    Qwen2VLForConditionalGeneration,
    LogitsProcessor,
    LogitsProcessorList,
    AutoModelForCausalLM,
    AutoTokenizer
)
from gui_aima.constants import (
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    chat_template
)
import torch.nn.functional as F

from gui_aima.constants import (
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN_0,
    DEFAULT_POINTER_PAD_TOKEN_1,
    DEFAULT_POINTER_PAD_TOKEN_2,
    DEFAULT_POINTER_PAD_TOKEN_3,
    DEFAULT_POINTER_PAD_TOKEN_4,
    DEFAULT_POINTER_PAD_TOKEN_5,
    DEFAULT_POINTER_PAD_TOKEN_list
)

class ForceFollowTokensLogitsProcessor(LogitsProcessor):
    """
    Forces proper token sequence for pointer tokens.
    For single point: pad_token → end_token
    For multiple points: handles the sequence of start → pad_i → end for each point
    """
    def __init__(self, tokenizer, number_of_points=1):
        super().__init__()
        self.number_of_points = number_of_points
        self.tokenizer = tokenizer
        
        # Token IDs
        self.start_id = tokenizer.encode(DEFAULT_POINTER_START_TOKEN)[0]
        self.end_id = tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
        
        if number_of_points == 1:
            self.pad_ids = [tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0]]
        else:
            self.pad_ids = [tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN_list[i])[0] for i in range(number_of_points)]
        
        self.force_queue = []
        self.current_point = 0  # Track which point we're processing

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Called at each decoding step to modify scores.
        
        Args:
            input_ids: shape (batch_size, seq_len). The already-decoded tokens.
            scores:    shape (batch_size, vocab_size). Model logits for the next token.
        """
        batch_size = input_ids.shape[0]
        if batch_size > 1:
            raise NotImplementedError("Batch size must be 1 for this logits processor.")
        
        last_token_id = input_ids[0, -1].item()
        
        # Check if last token was a start token
        if last_token_id == self.start_id and self.current_point < self.number_of_points:
            # Force the corresponding pad token for this point
            self.force_queue.append(self.pad_ids[self.current_point])
            self.force_queue.append(self.end_id)
            self.current_point += 1
        
        # Check if last token was any pad token
        elif last_token_id in self.pad_ids:
            # Already handled by the queue from start token
            pass
        
        # If we have forced tokens waiting in the queue, override the distribution
        if len(self.force_queue) > 0:
            forced_token = self.force_queue.pop(0)
            new_scores = torch.full_like(scores, float('-inf'))
            new_scores[0, forced_token] = 0.0  # log prob = 0 => prob = 1
            return new_scores
        
        # Otherwise, return scores unmodified
        return scores


# Alternative simpler version if the placeholder already contains all tokens
class ForceFollowTokensLogitsProcessorSimple(LogitsProcessor):
    """
    Simpler version when use_placeholder=True and all tokens are already in the prompt.
    This just ensures no additional pointer tokens are generated.
    """
    def __init__(self, tokenizer, number_of_points=1):
        super().__init__()
        self.tokenizer = tokenizer
        
        # Collect all pointer-related token IDs to suppress
        self.pointer_token_ids = set()
        
        # Add start and end tokens
        self.pointer_token_ids.add(tokenizer.encode(DEFAULT_POINTER_START_TOKEN)[0])
        self.pointer_token_ids.add(tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0])
        
        # Add pad tokens
        if number_of_points == 1:
            self.pointer_token_ids.add(tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0])
        else:
            pad_tokens = [DEFAULT_POINTER_PAD_TOKEN_0, DEFAULT_POINTER_PAD_TOKEN_1,
                         DEFAULT_POINTER_PAD_TOKEN_2, DEFAULT_POINTER_PAD_TOKEN_3,
                         DEFAULT_POINTER_PAD_TOKEN_4, DEFAULT_POINTER_PAD_TOKEN_5]
            for i in range(number_of_points):
                self.pointer_token_ids.add(tokenizer.encode(pad_tokens[i])[0])
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        When use_placeholder=True, all pointer tokens are already in the prompt.
        We just need to ensure the model doesn't generate any additional pointer tokens.
        """
        # Suppress all pointer-related tokens
        for token_id in self.pointer_token_ids:
            scores[:, token_id] = float('-inf')
        
        return scores

def get_prediction_region_point(attn_scores, n_width, n_height, top_n=30, activation_threshold=0.3, return_all_regions=True, rect_center=False):
    """
    1. Select activated patches
    2. Divide connected patches into different regions
    3. Calculate the average activation value for each region
    4. Select the region with the highest average activation value
    5. Return the center point of that region as the final prediction point
    """

    # Get patches with activation values greater than a certain proportion of the maximum activation value as activated patches
    # Get the highest activation value and threshold
    max_score = attn_scores[0].max().item()
    threshold = max_score * activation_threshold
    # Select all patches above the threshold
    mask = attn_scores[0] > threshold
    valid_indices = torch.nonzero(mask).squeeze(-1)
    topk_values = attn_scores[0][valid_indices]
    topk_indices = valid_indices
    
    # Convert indices to 2D coordinates
    topk_coords = []
    for idx in topk_indices.tolist():
        y = idx // n_width
        x = idx % n_width
        topk_coords.append((y, x, idx))
    
    # Divide into connected regions
    regions = []
    visited = set()
    for i, (y, x, idx) in enumerate(topk_coords):
        if idx in visited:
            continue
            
        # Start a new region
        region = [(y, x, idx, topk_values[i].item())]
        visited.add(idx)
        queue = [(y, x, idx, topk_values[i].item())]
        
        # BFS to find connected points
        while queue:
            cy, cx, c_idx, c_val = queue.pop(0)
            
            # Check 4 adjacent directions
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                n_idx = ny * n_width + nx
                
                # Check if this adjacent point is in the topk list
                for j, (ty, tx, t_idx) in enumerate(topk_coords):
                    if ty == ny and tx == nx and t_idx not in visited:
                        visited.add(t_idx)
                        region.append((ny, nx, t_idx, topk_values[j].item()))
                        queue.append((ny, nx, t_idx, topk_values[j].item()))
        
        regions.append(region)
    
    # Calculate the average activation value for each region
    region_scores = []
    region_centers = []
    region_points = []
    
    for region in regions:
        # Calculate average score for the region

        # avg score
        # avg_score = sum(item[3] for item in region) / len(region)

        # max score
        avg_score = max(item[3] for item in region)

        region_scores.append(avg_score)

        # Calculate normalized center coordinates for each patch, then take the average
        normalized_centers = []
        weights = []
        y_coords = set()
        x_coords = set()

        for y, x, _, score in region:
            # Normalized coordinates of the center point for each patch
            center_y = (y + 0.5) / n_height
            center_x = (x + 0.5) / n_width
            normalized_centers.append((center_x, center_y))
            weights.append(score)

            y_coords.add(center_y)
            x_coords.add(center_x)

        region_points.append(normalized_centers)

        # Calculate the average of normalized coordinates as the region center
        if not rect_center:
            # Weighted average
            total_weight = sum(weights)
            weighted_x = sum(nc[0] * w for nc, w in zip(normalized_centers, weights)) / total_weight
            weighted_y = sum(nc[1] * w for nc, w in zip(normalized_centers, weights)) / total_weight
            avg_center_x, avg_center_y = weighted_x, weighted_y
        else:
            avg_center_x = sum(x_coords) / len(x_coords)
            avg_center_y = sum(y_coords) / len(y_coords)
        region_centers.append((avg_center_x, avg_center_y))
        
    # Select the region with the highest average activation value
    sorted_indices = sorted(range(len(region_scores)), key=lambda i: region_scores[i], reverse=True)
    sorted_scores = [region_scores[i] for i in sorted_indices]
    sorted_centers = [region_centers[i] for i in sorted_indices]
    sorted_points = [region_points[i] for i in sorted_indices]
    best_point = sorted_centers[0]

    if return_all_regions:
        # Outputs:
        # 1. best_point: the center point of the region with the highest average activation value
        # 2. sorted_centers: the center points of all regions, sorted by the average activation value in descending order
        # 3. sorted_scores: the average activation values of all regions, sorted in descending order
        # 4. sorted_points: the normalized center coordinates of all patches, sorted by the average activation value in descending order
        return best_point, sorted_centers, sorted_scores, sorted_points
    else:
        return best_point

def inference(conversation, model, tokenizer, data_processor, logits_processor=None, use_placeholder=False, topk=5):
    """
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
    """
    if isinstance(model.config.pointer_pad_token_id, list):
        number_of_points=len(model.config.pointer_pad_token_id)
    else:
        number_of_points=1
    if logits_processor is None:
        if use_placeholder:
            # When placeholder is used, all tokens are already there
            # We just need to prevent generating additional pointer tokens
            logits_processor = ForceFollowTokensLogitsProcessorSimple(
                tokenizer=tokenizer,
                number_of_points=number_of_points
            )
        else:
            # When not using placeholder, need to force the proper sequence
            logits_processor = ForceFollowTokensLogitsProcessor(
                tokenizer=tokenizer,
                number_of_points=number_of_points
            )

    if number_of_points==1:
        assiatant_starter = "" if not use_placeholder else "<|im_start|>assistant<|recipient|>os\npyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)"
    else:
        target_text = "".join(
            f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN_list[i]}{DEFAULT_POINTER_END_TOKEN}"
            for i in range(number_of_points)
        )
        assiatant_starter = "" if not use_placeholder else f"<|im_start|>assistant<|recipient|>os\npyautogui.click({target_text})"
    # assiatant_starter = "" if not use_placeholder else "<|im_start|>assistant<|recipient|>os\npyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)"
    # assiatant_starter = "" if not use_placeholder else "<|im_start|>assistant<|recipient|>os\ncoord=[(<|pointer_start|><|pointer_pad|><|pointer_end|>]"
    pred = {
        "output_text": None, # generated text
        "n_width": None, # number of patch_tokens in width dimension
        "n_height": None, # number of patch_tokens in height dimension
        "attn_scores": None, # attention scores over the image patches
        "topk_points": None, # topk points
        "topk_values": None, # topk values
        "topk_points_all": None, # all points
    }

    # prepare text
    text = data_processor.apply_chat_template(conversation,
                                            tokenize=False,
                                            add_generation_prompt=False,
                                            chat_template=chat_template
                                            )
    text += assiatant_starter

    # prepare inputs
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = data_processor(text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt"
                            )
    inputs = inputs.to(model.device)
    
    results = model.generate(**inputs,
                            max_new_tokens=2048 if not use_placeholder else 1,
                            logits_processor=LogitsProcessorList([logits_processor]),
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            output_attentions=True,
                            temperature=0.1
                            )


    # decode the generated ids
    input_ids = inputs["input_ids"][0]
    generated_ids = results.sequences[0][len(input_ids):]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    pred["output_text"] = output_text
    if use_placeholder:
        pointer_pad_mask = torch.isin(input_ids, torch.tensor(model.config.pointer_pad_token_id, device=input_ids.device))
    else:
        pointer_pad_mask = torch.isin(generated_ids[:-1], torch.tensor(model.config.pointer_pad_token_id, device=generated_ids.device))
    if len(torch.nonzero(pointer_pad_mask, as_tuple=False).squeeze(-1)) == 0:
        return pred
    visual_mask = (input_ids == model.config.image_token_id)
    visual_indices = torch.nonzero(visual_mask, as_tuple=False).squeeze(-1)
    target_indices = torch.nonzero(pointer_pad_mask, as_tuple=False).squeeze(-1)

    topk_query_indices = None
    global_pattern_per_query = None
    query_start_indice = visual_indices[-1]
    query_end_mask = (input_ids == model.config.pointer_start_token_id)
    query_end_mask = torch.nonzero(query_end_mask, as_tuple=False).squeeze(-1)[0]
    query_indices = torch.arange(query_start_indice + 1, query_end_mask, device=input_ids.device)
    if model.config.part_query_weighting:
        query_indices=query_indices[0:-12]
    else:
        query_indices=query_indices
    all_layer_hs = torch.stack(results.hidden_states[0][1:], dim=0)

    sample_layer_hs = all_layer_hs[:, 0, :, :]            # (n_layer, seq_len, d_model)

    query_hs  = sample_layer_hs[:, query_indices, :]      # (n_layer, n_query,  d_model)
    visual_hs = sample_layer_hs[:, visual_indices, :]                       # (n_layer, n_visual, d_model)

    # 3) cosine‑similarity matrix  →  (n_layer, n_query, n_visual)
    query_hs  = F.normalize(query_hs,  dim=-1)
    visual_hs = F.normalize(visual_hs, dim=-1)
    sim_matrix = torch.einsum('lqd,lvd->lqv', query_hs, visual_hs)       # l = layer, q = query, v = visual

    # 4) per‑layer visual “attention” of each query token (sum over visual tokens)
    attn_per_query = sim_matrix.sum(dim=-1) # (n_layer, n_query)
    if not model.config.kl_query_weighting:
        # 5) aggregate across layers                             # (n_query,)
        k = model.config.query_topk
        if model.config.layer_wise_query_weighting:
            topk_layer_vals, topk_layer_indices = torch.topk(attn_per_query, k, dim=-1)
            topk_query_indices=query_indices[topk_layer_indices]
        else:
            agg_attn = attn_per_query.sum(dim=0)                                   
            topk_vals, topk_local_idx = torch.topk(agg_attn, k, largest=True)
            topk_query_indices = query_indices[topk_local_idx]
    elif model.config.kl_query_weighting:
        global_pattern_per_query = attn_per_query.sum(dim=0)
        global_pattern_per_query = (global_pattern_per_query).softmax(dim=-1) 
    attn_scores, _ = model.multi_patch_pointer_head_attention(query_indices, visual_indices, target_indices,results.attentions[0],topk_query_indices,global_pattern_per_query,batch_idx=0)
    pred["attn_scores"] = attn_scores.tolist()

    _, n_height, n_width = (inputs["image_grid_thw"][0] // model.visual.spatial_merge_size).tolist()
    pred["n_width"] = n_width
    pred["n_height"] = n_height

    # get the topk points according to the attention scores
    best_point, region_points, region_scores, region_points_all = get_prediction_region_point(attn_scores, n_width, n_height, return_all_regions=True, rect_center=False)
    topk_points = region_points[:topk] if len(region_points) > topk else region_points
    topk_values = region_scores[:topk] if len(region_scores) > topk else region_scores
    topk_points_all = region_points_all[:topk] if len(region_points_all) > topk else region_points_all
    pred["topk_points"] = topk_points
    pred["topk_values"] = topk_values
    pred["topk_points_all"] = topk_points_all

    return pred