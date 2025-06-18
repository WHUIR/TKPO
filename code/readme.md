The models were trained using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

Please replace the `modeling_qwen2.py` file in the [Transformers](https://huggingface.co/docs/transformers/index) package (version 4.43.2), located at `transformers/models/qwen2/modeling_qwen2.py`, with the provided version.


More specifically, we insert the TKPO loss code into the original the `def forward` function `class Qwen2ForCausalLM(Qwen2PreTrainedModel)`, as shown below
```python
loss = None
tkpo_loss = True
if labels is not None:
    if tkpo_loss:
        tau = 0.15
        lambda_loss = 0.5
        shift_logits = logits[...,:-1,:].contiguous()
        shift_probs_pos = torch.softmax(shift_logits, dim=-1)
        pos_token = labels[..., 1:].contiguous()
        pos_tokencp = copy.deepcopy(pos_token)
        pos_tokencp[pos_token==-100] = 100
        pos_prob = torch.gather(shift_probs_pos, 2, pos_tokencp.unsqueeze(-1)).squeeze(-1)
        pos_prob = torch.exp(pos_prob/tau)
        with torch.no_grad():
            ## {[104179, 117599]:"文学诗意", [100267, 105310]:"真实客观", [100700, 100011]:"详细全面", [110485,  99936]:“简洁关键”}
            ### language style
            input_ids_anti = input_ids.clone()
            if "104179, 117599" in str(input_ids[0, :].clone().cpu().detach().numpy().tolist()):
                input_ids_anti[:, :20][input_ids_anti[:, :20]==104179] = 100267
                input_ids_anti[:, :20][input_ids_anti[:, :20]==117599] = 105310
            else:
                input_ids_anti[:, :20][input_ids_anti[:, :20]==100267] = 104179
                input_ids_anti[:, :20][input_ids_anti[:, :20]==105310] = 117599
            """
            ### level-of-detail
            input_ids_anti = input_ids.clone()
            if "100700, 100011" in str(input_ids[0, :].clone().cpu().detach().numpy().tolist()):
                input_ids_anti[:, :20][input_ids_anti[:, :20]==100700] = 110485
                input_ids_anti[:, :20][input_ids_anti[:, :20]==100011] = 99936
            elif "100267, 105310" in str(input_ids[0, :].clone().cpu().detach().numpy().tolist()):
                input_ids_anti[:, :20][input_ids_anti[:, :20]==100267] = 110485
                input_ids_anti[:, :20][input_ids_anti[:, :20]==105310] = 99936
            else:
                input_ids_anti[:, :20][input_ids_anti[:, :20]==110485] = 100267
                input_ids_anti[:, :20][input_ids_anti[:, :20]==99936] = 105310
            """ 
            outputs_anti = self.model(
                        input_ids=input_ids_anti.to(input_ids.dtype),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        cache_position=cache_position,
                    )
            hidden_states_anit = outputs_anti[0]
            logits_anti = self.lm_head(hidden_states_anit)[...,:-1,:].contiguous()
            logits_neg = logits_anti.float()
            logits_neg.scatter_(2, pos_tokencp.unsqueeze(-1), -1e6)
            # _, neg_token = torch.topk(logits_neg, 20, dim=-1)
            # neg_token = reject_sampling_tensor(torch.softmax(logits_neg, dim=-1), k=20)
            neg_token = torch.multinomial(torch.softmax(logits_neg, dim=-1).squeeze(0), 20, replacement=False).unsqueeze(0)
            neg_probs = torch.gather(shift_probs_pos, 2, neg_token)
        
        neg_pos_prob = torch.sum(torch.exp(neg_probs/tau),dim=-1) + pos_prob
        losses_tkpo = pos_prob/neg_pos_prob
        mask_label = torch.ones_like(pos_token)
        mask_label[pos_token==-100] = 0
        loss_tkpo = -torch.log(torch.sum((losses_tkpo * mask_label),dim=-1)/torch.sum(mask_label,dim=-1))


        shift_logits_ce = shift_logits.view(-1, self.config.vocab_size)
        shift_labels_ce = pos_token.view(-1)
        loss_fct = CrossEntropyLoss()
        loss_ce = loss_fct(shift_logits_ce, shift_labels_ce)
        loss = loss_ce + lambda_loss*loss_tkpo.squeeze()
    else:
        ## CE loss
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        loss_fct = CrossEntropyLoss()
```
