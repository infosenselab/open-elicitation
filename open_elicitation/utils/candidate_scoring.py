import json
import math

import torch


def get_label_tokens(tokenizer, valid_labels, logger=None):
    
    label_tokens = {}

    for label in valid_labels:

        ids = tokenizer(label, add_special_tokens=False).input_ids

        label_tokens[label] = ids

        
        if True and logger is not None:
            logger.info(
                f"Label: `{label}`. "
                f"Ids: `{ids}` -> Converts back to: "
                f"`{tokenizer.convert_ids_to_tokens(ids)}`\n"
            )

    return label_tokens




def score_candidate(prompt_ids, candidate_ids, model, tokenizer, logger=None):
    """
    Compute log P(label | prompt). The candidate may consist of multiple BPE tokens.
    """

    candidate_ids = torch.tensor([candidate_ids], device=model.device)
    
    #DEBUG
    if False and logger is not None:
        logger.info(
            f"\n### candidate_ids.\n\n"
            f"{candidate_ids}\n"
        )


    prompt_ids = prompt_ids.to(model.device)
    full_ids = torch.cat([prompt_ids, candidate_ids], dim=1)
    
    #DEBUG
    if False and logger is not None:
        logger.info(
            f"\n### full_ids.\n\n"
            f"{full_ids}\n"
        )

    
    attention_mask = torch.ones_like(full_ids)

    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            attention_mask=attention_mask,
        )

    logits = outputs.logits


    #DEBUG
    if True and logger is not None:
        logger.info(
            f"\n### logits.\n\n"
            f"- Logits Shape: `{tuple(logits.shape)}`\n"
            f"\n{logits}\n"
        )



    prompt_len = prompt_ids.shape[1]
    candidate_len = candidate_ids.shape[1]



    # Token j is predicted by the logits at the immediately preceding position.
    start = prompt_len - 1
    end = start + candidate_len

    candidate_logits = logits[:, start:end, :]

    log_probs = torch.log_softmax(candidate_logits.float(), dim=-1)

    token_log_probs = log_probs.gather(dim=2, index=candidate_ids.unsqueeze(-1)).squeeze(-1)

    total_log_prob = token_log_probs.sum(dim=1).item()
    sequence_prob = math.exp(total_log_prob)

    
    # bookkeeping
    token_details = []

    for j in range(candidate_len):

        token_id = candidate_ids[0, j].item()
        token_text = tokenizer.convert_ids_to_tokens([token_id])[0]
        log_prob = token_log_probs[0, j].item()

        token_details.append({
            "token_id": token_id,
            "token": token_text,
            "log_prob": log_prob,
            "prob": math.exp(log_prob),
        })

    return {
        "token_count": candidate_len,
        "sequence_log_prob": total_log_prob,
        "sequence_prob": sequence_prob,
        "tokens": token_details,
    }





def score_candidates_batch(prompt_ids_list, label_tokens, valid_labels, model, tokenizer, logger=None):
    """
    Score every prompt x label continuation in one model forward pass.
    """
    
    
    full_sequences = []
    metadata = []
    
    
    for row_idx, prompt_ids in enumerate(prompt_ids_list):
        prompt_ids = prompt_ids.squeeze(0).tolist()
        
        for label in valid_labels:
            candidate_ids = label_tokens[label]
            full_ids = prompt_ids + candidate_ids

            full_sequences.append(full_ids)
        
            #DEBUG
            if False and logger is not None:
                logger.info(
                    f"\n### candidate_ids.\n\n"
                    f"{candidate_ids}\n"
                )
                
            metadata.append({
                "row_idx": row_idx,
                "label": label,
                "prompt_len": len(prompt_ids),
                "candidate_len": len(candidate_ids),
                "candidate_ids": candidate_ids,
            })
    
 
            #DEBUG
            if False and logger is not None:
                logger.info(
                    f"\n### full_ids.\n\n"
                    f"{full_ids}\n"
                )


    # Pad complete prompt+candidate sequences together.
    # tokenizer.padding_side is already set to "left".
    batch = model.device

    padded = tokenizer.pad(
        {"input_ids": full_sequences},
        padding=True,
        return_tensors="pt",
    )


    input_ids = padded["input_ids"].to(batch)
    attention_mask = padded["attention_mask"].to(batch)

    
    
    #with torch.inference_mode():
    #    outputs = model(
    #        input_ids=input_ids,
    #        attention_mask=attention_mask,
    #    )
    
    max_candidate_len = max(len(label_tokens[label]) for label in valid_labels)
    logits_to_keep = max_candidate_len + 1

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=logits_to_keep,
        )

    results = [{} for _ in prompt_ids_list]
    kept_length = outputs.logits.shape[1]
    padded_length = input_ids.shape[1]

    for sequence_idx, info in enumerate(metadata):
        row_idx = info["row_idx"]
        label = info["label"]
        candidate_len = info["candidate_len"]

        candidate_logits = outputs.logits[sequence_idx, kept_length - candidate_len - 1:kept_length - 1, :]

        candidate_ids = input_ids[sequence_idx, padded_length - candidate_len:padded_length]

        candidate_log_probs = torch.log_softmax(candidate_logits.float(), dim=-1)

        token_log_probs = candidate_log_probs.gather(dim=1, index=candidate_ids.unsqueeze(1)).squeeze(1)

        sequence_log_prob = token_log_probs.sum().item()

        results[row_idx][label] = {
            "token_count": candidate_len,
            "sequence_log_prob": sequence_log_prob,
            "sequence_prob": math.exp(sequence_log_prob),
        }


    return results
    
    
    


def build_prompt(argument, system_instruction, tokenizer):

    messages = [
            {
            "role": "system",
            "content": system_instruction,
        },
        {
            "role": "user",
            "content": argument,
        },
    ]

    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    prompt_ids = tokenized["input_ids"]

    prompt = tokenizer.decode(
        prompt_ids[0],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    return messages, prompt, prompt_ids



def process_batch(rows, model, tokenizer, valid_labels, label_tokens, model_choice, logger=None):

    prompt_ids_list = []

    for row in rows:

        messages, prompt, prompt_ids = build_prompt(
            argument=row["argument"],
            system_instruction=row["system_instruction"],
            tokenizer=tokenizer,
        )


        #prompts.append(prompt)
        prompt_ids_list.append(prompt_ids)
        
        
        # Log chat template
        if True and logger is not None:
            logger.info(
                f"## Step {row.get('step', 'N/A')} | "
                f"Trajectory {row.get('trajectory_id', 'N/A')} | "
                f"Candidate {row.get('candidate_position', 'N/A')}\n\n"
                f"- **Argument ID:** `{row['argument_id']}`  \n"
                f"- **Original topic:** `{row['original_topic']}`  \n"
                f"- **Evaluation topic:** `{row['evaluation_topic']}`  \n"
                f"- **Topic match:** `{row['topic_match']}`\n\n"
                f"### Input Chat Template\n\n"
                f"```json\n"
                f"{json.dumps(messages, indent=2, ensure_ascii=False)}\n"
                f"```\n\n"
                f"### Rendered Chat Template\n\n"
                f"```text\n"
                f"{prompt}\n"
                f"```\n"
            )

    # ------------------------
    # Tokenize batch
    # ------------------------

    
    if True and logger is not None:
        i = 0
        input_ids = prompt_ids_list[i][0]

        logger.info(
            "\n### Tokenization Check\n\n"
            f"- Input IDs shape: `{tuple(prompt_ids_list[i].shape)}`\n"
            f"\n#### Example {i}\n\n"
            f"**Input IDs:**\n"
            f"```text\n"
            f"{input_ids.tolist()}\n"
            f"```\n\n"
            f"**Tokens:**\n"
            f"```text\n"
            f"{tokenizer.convert_ids_to_tokens(input_ids.tolist())}\n"
            f"```\n\n"
            f"**Decoded:**\n"
            f"```text\n"
            f"{tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)}\n"
            f"```"
        )
    

    # One model forward pass for all prompt x label pairs.
    all_candidate_scores = score_candidates_batch(
        prompt_ids_list=prompt_ids_list,
        label_tokens=label_tokens,
        valid_labels=valid_labels,
        model=model,
        tokenizer=tokenizer,
    )

    results = []

    for row, candidate_scores in zip(rows, all_candidate_scores):
        label_probs = {
            label: candidate_scores[label]["sequence_prob"]
            for label in valid_labels
        }

        label_log_probs = {
            label: candidate_scores[label]["sequence_log_prob"]
            for label in valid_labels
        }

        predicted_label = max(label_probs, key=label_probs.get) if len(valid_labels) > 1 else None

        results.append({
            **row,
            "model": model_choice,
            "predicted_label": predicted_label,
            "label_probs": label_probs,
            "label_log_probs": label_log_probs,
        })

    return results


def format_valid_labels(valid_labels):
    return "\n".join(valid_labels)

