import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import RotaryEmbedding, apply_rotary_pos_emb
from transformers.trainer_utils import get_last_checkpoint
from itertools import chain
from typing import Optional
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from flash_attn.modules.mha import FlashSelfAttention

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="pythia-1.3b",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )

    max_positions: Optional[int] = field(
        default=8192,
        metadata={
            "help": (
                "The maximun sequence length of the model."
            )
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="pile", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

class FlashAttentionWrapper(torch.nn.Module):
    def __init__(self, attention, max_seqlen = 8192):
        super().__init__()
        self.attention = attention
        self.max_seqlen = max_seqlen
        self.flash_self_attention = FlashSelfAttention(causal = True)
        self.dropout_p = 0.0

    def forward(self,
        hidden_states,
        attention_mask,
        head_mask=None,
        layer_past=None,
        use_cache=False,
        output_attentions=False):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.attention.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.attention.num_attention_heads, 3 * self.attention.head_size)
        qkv = qkv.view(*new_qkv_shape)
        
        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.attention.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.attention.head_size : 2 * self.attention.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.attention.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.attention.rotary_ndims]
        query_pass = query[..., self.attention.rotary_ndims :]
        key_rot = key[..., : self.attention.rotary_ndims]
        key_pass = key[..., self.attention.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        offset = 0
        if has_layer_past:
            offset = layer_past[0].shape[-2]
            seq_len += offset
        cos, sin = self.attention.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, offset=offset)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # Compute attention
        #attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        qkv = torch.concat([query.unsqueeze(2), key.unsqueeze(2), value.unsqueeze(2)], dim = 2).permute(0, 3, 2, 1, 4).half()
        attn_output = self.flash_self_attention(qkv)
        attn_weights = None

        # Reshape outputs
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), self.attention.num_attention_heads * self.attention.head_size)
        attn_output = self.attention.dense(attn_output)
        
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    set_seed(training_args.seed)
    model = GPTNeoXForCausalLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.mask_token
    max_positions = model_args.max_positions
    tokenizer.model_max_length = max_positions
    for each in model.gpt_neox.layers:
        original_emb = each.attention.rotary_emb
        each.attention.rotary_emb = RotaryEmbedding(each.attention.rotary_ndims,max_positions,10000)
        each.attention.bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                    1, 1, max_positions, max_positions
                )
        each.attention = FlashAttentionWrapper(each.attention, max_seqlen = max_positions)

    # patching for the random contiguous tensors bug
    for p in model.parameters():
        p = p.contiguous()

    def merge_questions_and_answers(examples):
        out = tokenizer([question + " " + answer for question, answer in zip(examples["input"], examples["output"])])
        return out

    block_size = tokenizer.model_max_length
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    

    if data_args.dataset_name == "pile":
        base_url = "https://the-eye.eu/public/AI/pile/"
        data_files = {
            "train": [base_url + "train/"+ f"{idx:02d}.jsonl.zst" for idx in range(30)],
            "validation": base_url + "val.jsonl.zst",
            "test": base_url + "test.jsonl.zst",
        }
        datasets = load_dataset("json", data_files=data_files, streaming=True)
        datasets = datasets.filter(lambda x: len(x["text"])>=max_positions)
        tokenized_datasets = datasets.map(
            lambda examples: tokenizer(examples["text"]),
            batched=True,
        )
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
        )
        lm_datasets = lm_datasets.filter(lambda x: len(x["input_ids"])>=max_positions)
    elif data_args.dataset_name == "qasper":
        datasets = load_dataset("tau/scrolls", "qasper")
        datasets.pop("test")
        tokenized_datasets = datasets.map(
           merge_questions_and_answers,
           batched=True,
           num_proc = 1,
           remove_columns = datasets["train"].column_names,
           desc="Running tokenizer on dataset",
        )

        lm_datasets = tokenized_datasets.map(
           group_texts,
           batched=True,
           num_proc=1,
           desc=f"Grouping texts in chunks of {block_size}",
        )
    else:
        raise Exception("Sorry, please the dataset specified can not be recognized")
    
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset= train_dataset,
        eval_dataset= eval_dataset,
        tokenizer = tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    else:
        checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    max_train_samples = (len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if __name__  == "__main__":
    main()
