import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast, Qwen2_5_VLForConditionalGeneration
from gui_aima.constants import IGNORE_INDEX
from typing import List, Tuple, Union, Optional
from gui_aima.trainer import rank0_print
import re
from gui_aima.model_utils import calculate_attention_from_qk
class QwenVLwithVisionHeadOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    """
    Output class for Qwen2_5_VL with pointer head, extending the base output class.
    
    Args:
        lm_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Language modeling loss.
        pointer_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Vision pointer network loss.
        pointer_scores (`List[torch.FloatTensor]`, *optional*):
            Attention scores from the pointer network, one tensor per batch item.
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Combined loss (weighted sum of lm_loss and pointer_loss).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores from the language modeling head.
        past_key_values, hidden_states, attentions, rope_deltas:
            Same as parent class.
    """
    def __init__(self, lm_loss=None, pointer_loss=None, pointer_scores=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_loss = lm_loss
        self.pointer_loss = pointer_loss
        self.pointer_scores = pointer_scores

class GroundingHead_MultiPatch_Attention(nn.Module):
    def __init__(self, config=None, topk=-1):
        super().__init__()
        self.config = config
        self.topk = topk
    
    # no softmax head weight first, then weighting
    def forward(self,
                query_indices,
                visual_indices,
                target_indices,
                self_attentions,
                topk_query_indices,
                global_pattern_per_query,
                batch_idx,
                labels: Optional[torch.Tensor] = None,  # shape: [n_dec, n_enc], binary mask of patches in bbox
                do_single_patch: bool = False,
               ):
        all_head_attentions = []
        query_head_attns = []
        epsilon = 1e-8
        # lay_n , batch_n, head_n, seq_n, seq_n
        for layer_idx in range(len(self_attentions)):
            layer_attn = self_attentions[layer_idx]
            sample_layer_attn = layer_attn[batch_idx]
            if self.config.kl_query_weighting or self.config.query_topk is not None:
                if not self.config.kl_query_weighting and topk_query_indices is not None:
                    if self.config.layer_wise_query_weighting:
                        q_attn = sample_layer_attn[:, topk_query_indices[layer_idx], :][:, :, visual_indices]
                    else:
                        q_attn = sample_layer_attn[:, topk_query_indices, :][:, :, visual_indices]
                elif self.config.kl_query_weighting and global_pattern_per_query is not None:
                    # q_attn = sample_layer_attn[:, query_indices, :][:, :, visual_indices]
                    q_attn = sample_layer_attn[:, 0:-1, visual_indices]
                query_head_attns.append(q_attn)  
            # target_patch_attentions = sample_layer_attn[:, target_indices, visual_indices]
            # target_patch_attentions = sample_layer_attn[:, target_indices.unsqueeze(1), visual_indices.unsqueeze(0)]
            target_patch_attentions = sample_layer_attn[:, -1, visual_indices.unsqueeze(0)]
            # print(sample_layer_attn.shape)
            # print(sdsd)
            # target_patch_attentions = sample_layer_attn[:, target_indices.unsqueeze(1), visual_indices.unsqueeze(0)]
            if target_patch_attentions.shape[1]!=1:
                target_patch_attentions=torch.mean(target_patch_attentions, dim=1,keepdim=True)
            target_patch_attentions=target_patch_attentions.squeeze(1)
            all_head_attentions.append(target_patch_attentions)
        # lay_n * head_n, 1, patch_n
        all_head_attentions_cat = torch.cat(all_head_attentions, dim=0)
        if self.config.kl_query_weighting or self.config.query_topk is not None:
            query_head_attns_cat = torch.cat(query_head_attns, dim=0) 
            if not self.config.kl_query_weighting and topk_query_indices is not None:
                head_weights = query_head_attns_cat.sum(dim=(-1, -2)) 
                head_weights = (head_weights).softmax(dim=-1)
                # print("top-k query attn weight")
            elif self.config.kl_query_weighting and global_pattern_per_query is not None:
                head_weights = query_head_attns_cat.sum(dim=(-1)) 
                head_weights = (head_weights).softmax(dim=-1)
                head_weights = head_weights.clamp_min(1e-12)               
                global_pattern_per_query = global_pattern_per_query.clamp_min(1e-12)
                distance =- F.kl_div(
                    (head_weights).log(),                      
                    global_pattern_per_query.expand_as(head_weights),                 
                    reduction='none').sum(dim=-1) 
                head_weights=(distance).softmax(dim=-1)  
                # print("kl query attn weight")

        if self.topk != -1:
            with torch.no_grad():
            # lay_n * head_n, 1
                flat_head_weights=all_head_attentions_cat.sum(dim=-1)
            _, topk_idx = torch.topk(flat_head_weights, self.topk, largest=True)
            target_patch_attentions_topk = all_head_attentions_cat[topk_idx]
        else:
            target_patch_attentions_topk = all_head_attentions_cat

        if self.training:
            del all_head_attentions_cat, all_head_attentions
            if topk_query_indices is not None:
                del query_head_attns_cat
        else:
            del all_head_attentions_cat, all_head_attentions
            if topk_query_indices is not None:
                del query_head_attns_cat
            torch.cuda.empty_cache()

        if self.config.kl_query_weighting or self.config.query_topk is not None:
            target_patch_attentions_merged = torch.mul(head_weights[:, None], target_patch_attentions_topk)
            target_patch_attentions_merged = target_patch_attentions_merged.sum(dim=0,keepdim=True)
        else:
            target_patch_attentions_merged = target_patch_attentions_topk.mean(dim=0,keepdim=True)
        
        target_patch_attentions_merged /= (target_patch_attentions_merged.sum(dim=-1, keepdim=True) + epsilon)    
        attn_weights = target_patch_attentions_merged

        loss = None
        if (labels is not None) and (not do_single_patch):
            labels_float = labels.float()
            # Normalize each row to get target probability distribution
            target_dist = labels_float / (labels_float.sum(dim=-1, keepdim=True) + epsilon)

            # # Apply log_softmax to logits
            pred_log_probs = torch.log(attn_weights)
            # Use KL divergence as loss
            loss = F.kl_div(pred_log_probs, target_dist, reduction='batchmean')

        return attn_weights, loss

class Qwen2_5_VLForConditionalGenerationWithPointer(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_patch_pointer_head_attention=GroundingHead_MultiPatch_Attention(config=self.config)
        self.pointer_loss_weight = kwargs.get("pointer_loss_weight", 1.0)
        self.lm_loss_weight = kwargs.get("lm_loss_weight", 1.0)
        self.post_init()
    
    def reset_loss_weights(self, pointer_loss_weight, lm_loss_weight):
        self.pointer_loss_weight = pointer_loss_weight
        self.lm_loss_weight = lm_loss_weight

    def set_attention_args(self, weighting):
        self.config.layer_wise_query_weighting=None
        self.config.query_topk=None
        if 'part' in weighting:
            self.config.part_query_weighting=True
        else:
            self.config.part_query_weighting=False
        if 'kl' in weighting:
            self.config.kl_query_weighting=True
        else:
            self.config.kl_query_weighting=False
            k = int(re.search(r'(?<=query_)\d+', weighting).group())
            self.config.query_topk=k
            if 'layer_wise' in weighting:
                self.config.layer_wise_query_weighting=True
            else:
                self.config.layer_wise_query_weighting=False
        self.multi_patch_pointer_head_attention.config=self.config
   
    def forward(self,
                input_ids: torch.LongTensor = None, # (batch_size, seq_len)
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                pixel_values: Optional[torch.Tensor] = None,
                pixel_values_videos: Optional[torch.FloatTensor] = None,
                image_grid_thw: Optional[torch.LongTensor] = None,
                video_grid_thw: Optional[torch.LongTensor] = None,
                rope_deltas: Optional[torch.LongTensor] = None,
                cache_position: Optional[torch.LongTensor] = None,
                second_per_grid_ts: Optional[torch.Tensor] = None,
                # Grounding
                visual_token_indices_of_coordinates: Optional[torch.Tensor] = None, # shape: (batch_size, n_target); each element is the ground-truth index of the visual token that should be attended to for the corresponding target token
                multi_patch_labels: Optional[torch.Tensor] = None, # shape: list [(n_target, n_visual), ...]; binary mask of patches in bbox
                if_multi_patch: bool = True,
                coordinates: Optional[List[Tuple[float, float]]] = None,
                verbose: bool = False) -> Union[Tuple, QwenVLwithVisionHeadOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if verbose:
            rank0_print(f"input_ids: {input_ids.shape}, {input_ids[0][:5]}...")
            rank0_print(f"labels: {labels.shape}, {labels[0][:5]}...")
            rank0_print(f"pixel_values: {pixel_values.shape}")
            rank0_print(f"image_grid_thw: {image_grid_thw.shape}, {image_grid_thw}")
            rank0_print(f"coordinates: {coordinates}")
            rank0_print(f"visual_token_indices_of_coordinates: {visual_token_indices_of_coordinates}")
            rank0_print(f"return_dict: {return_dict}")

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids) # shape: (batch_size, seq_len, d_model)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0] # shape: (batch_size, seq_len, d_model)
        logits = self.lm_head(hidden_states)
        #[2, 16, 2385, 2385]
        # self_attentions_batch = outputs.attentions
        # num_layers = len(self_attentions_batch)
        # num_heads = self_attentions_batch[0].shape[1]

        lm_loss = None
        if labels is not None and self.lm_loss_weight > 0:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            lm_loss = loss_fct(shift_logits, shift_labels)


        # If vision supervision is requested, process the action head.
        pointer_loss = None
        pointer_scores = []
        if visual_token_indices_of_coordinates is not None:
            batch_size = input_ids.shape[0]
            pointer_losses = []
            
            # Process each sample individually because the number of visual and target tokens may vary.
            for i in range(batch_size):
                dummy_target = False
                # Get the token ids and corresponding hidden states for sample i.
                token_ids = input_ids[i]          # shape: (seq_length,)
                hs = hidden_states[i]             # shape: (seq_length, d_model)

                # Identify visual tokens indices.
                visual_mask = (token_ids == self.config.image_token_id)
                visual_indices = torch.nonzero(visual_mask, as_tuple=False).squeeze(-1) # shape: (n_visual,)

                # Identify target tokens (the ones that should attend to visual features).
                target_mask = torch.isin(token_ids, torch.tensor(self.config.pointer_pad_token_id, device=token_ids.device))
                target_indices = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)

                query_start_indice = visual_indices[-1]
                query_end_mask = (token_ids == self.config.pointer_start_token_id)
                query_end_mask = torch.nonzero(query_end_mask, as_tuple=False).squeeze(-1)[0]
                query_indices = torch.arange(query_start_indice + 1, query_end_mask, device=token_ids.device)
                if self.config.part_query_weighting==True:
                    query_indices=query_indices[0:-12]
                    # print('part query_indices')
                else:
                    query_indices=query_indices
                    # print('all query_indices')
                merged_indices = torch.cat([query_indices, target_indices], dim=0)
                calculated_attention = calculate_attention_from_qk(
                    model=self,
                    all_hidden_states=[outputs.hidden_states],
                    all_position_ids=position_ids,
                    query_indices=merged_indices,
                    all_attention_mask=attention_mask,
                )

                topk_query_indices = None
                global_pattern_per_query = None
                need_grad = self.training and bool(self.config.kl_query_weighting)
                with torch.set_grad_enabled(need_grad):
                    all_layer_hs = torch.stack(outputs.hidden_states[1:], dim=0)

                    sample_layer_hs = all_layer_hs[:, i, :, :]            # (n_layer, seq_len, d_model)

                    query_hs  = sample_layer_hs[:, query_indices, :]      # (n_layer, n_query,  d_model)
                    visual_hs = sample_layer_hs[:, visual_indices, :]                       # (n_layer, n_visual, d_model)

                    # 3) cosine‑similarity matrix  →  (n_layer, n_query, n_visual)
                    query_hs  = F.normalize(query_hs,  dim=-1)
                    visual_hs = F.normalize(visual_hs, dim=-1)
                    sim_matrix = torch.einsum('lqd,lvd->lqv', query_hs, visual_hs)       # l = layer, q = query, v = visual

                    # 4) per‑layer visual “attention” of each query token (sum over visual tokens)
                    attn_per_query = sim_matrix.sum(dim=-1)   
                    if not self.config.kl_query_weighting:
                        # 5) aggregate across layers                             # (n_query,)
                        k = self.config.query_topk
                        # k = min(k, agg_attn.size(0))     
                        if self.config.layer_wise_query_weighting:
                            # print('layer_wise', k)
                            topk_layer_vals, topk_layer_indices = torch.topk(attn_per_query, k, dim=-1)
                            # topk_query_indices=query_indices[topk_layer_indices]
                            topk_query_indices=topk_layer_indices
                        else:
                            # print('not layer_wise',k) 
                            agg_attn = attn_per_query.sum(dim=0)                                   
                            topk_vals, topk_local_idx = torch.topk(agg_attn, k, largest=True)
                            # topk_query_indices = query_indices[topk_local_idx]
                            topk_query_indices = topk_local_idx
                    elif self.config.kl_query_weighting:
                        # global pattern, merge layer                            # (n_layer, n_query)
                        global_pattern_per_query = attn_per_query.sum(dim=0)
                        global_pattern_per_query = (global_pattern_per_query).softmax(dim=-1) 
                        # topk_query_indices = None
                        # print("kl matching mode")
                    del sample_layer_hs        
                    del all_layer_hs
                # If either visual or target tokens are missing, skip this sample.
                if visual_indices.numel() == 0:
                    raise ValueError(f"No visual or target tokens found for sample {i}.")
                if target_indices.numel() == 0:
                    target_indices = torch.tensor([hs.shape[0] - 1]) # take the last token as the dummy target token
                    gt = torch.tensor([0]).to(hs.device) # take the first visual token as the dummy ground truth
                    if if_multi_patch:  # task the first 4 visual tokens as the ground truth
                        sample_labels = torch.zeros_like(visual_indices).unsqueeze(0)
                        sample_labels[0][:4] = 1
                    dummy_target = True
                else:
                    # For supervision, we assume that visual_token_indices_of_coordinates[i] is a tensor of shape (n_target,)
                    # where each element is an integer in the range [0, n_visual-1] indicating the ground-truth visual token.
                    gt = visual_token_indices_of_coordinates[i].to(hs.device) # shape: (n_target,)
                    if if_multi_patch:
                        sample_labels = multi_patch_labels[i]
                # Calculate loss for multi-patch mode
                if if_multi_patch:
                    attn_scores = None
                    loss_v = None
                    attn_scores, loss_v = self.multi_patch_pointer_head_attention(
                        query_indices,
                        visual_indices,
                        target_indices,
                        calculated_attention[0],
                        topk_query_indices,
                        global_pattern_per_query,
                        batch_idx=i,
                        labels=sample_labels
                    )
                pointer_scores.append(attn_scores.detach().cpu())

                pointer_losses.append(loss_v * 0.0 if dummy_target else loss_v)
            
            pointer_loss = torch.stack(pointer_losses).mean()

        # Combine the LM loss and vision loss using the provided loss weights.
        
        if lm_loss is None:
            total_loss = pointer_loss
        elif pointer_loss is None:
            total_loss = lm_loss
        else:
            total_loss = self.lm_loss_weight * lm_loss + self.pointer_loss_weight * pointer_loss

        if return_dict:
            return QwenVLwithVisionHeadOutputWithPast(
                lm_loss=lm_loss,
                pointer_loss=pointer_loss,
                pointer_scores=pointer_scores,
                loss=total_loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=None,
                rope_deltas=self.rope_deltas,
            )
        else:
            # When labels are provided, parent's forward returns a tuple with loss as the first element.
            if labels is not None:
                # Replace the LM loss with the combined loss.
                output = (lm_loss, pointer_loss, logits, pointer_scores,) + outputs[1:]
                print(f"returning: total_loss, logits, pointer_scores, ...")
                return (total_loss,) + output if total_loss is not None else output
            else:
                return outputs