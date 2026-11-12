"""Qwen2.5-14B-Instruct 4-bit QLoRA on compact L3 dialogue data.

This is a local validation of the L3 data at 14B scale; final hosting is separate.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

HERE = Path(__file__).resolve().parent
BASE = os.environ.get("BASE_MODEL", "models/qwen_Qwen2.5-14B-Instruct")
OUT = Path(os.environ.get("OUT_DIR", HERE / "lora_l3_14b_qlora"))
MAX_LEN = int(os.environ.get("MAX_LEN", "512"))
EPOCHS = float(os.environ.get("EPOCHS", "2"))
TRAIN_FILE = Path(os.environ.get("TRAIN_FILE", str(HERE / "l3_compact_train.jsonl")))
VAL_FILE = Path(os.environ.get("VAL_FILE", str(HERE / "l3_compact_val.jsonl")))


def tokenize(row: dict, tokenizer) -> dict:
    messages = row["messages"]
    # transformers 5.x apply_chat_template(tokenize=True) 返回 BatchEncoding/Encoding，
    # 且分段 tokenize 有边界合并问题 → 模板出字符串 + offsets 精确遮罩 prompt 部分
    try:
        full_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    except TypeError:
        full_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    try:
        prompt_str = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt_str = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    enc = tokenizer(full_str, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    plen = len(prompt_str)
    labels = [tid if off[1] <= plen else -100 for tid, off in zip(input_ids, enc["offset_mapping"])]
    input_ids = input_ids[:MAX_LEN]
    labels = labels[:MAX_LEN]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def main() -> None:
    ds = load_dataset("json", data_files={
        "train": str(TRAIN_FILE),
        "validation": str(VAL_FILE),
    })
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tokenized = ds.map(lambda row: tokenize(row, tokenizer), remove_columns=ds["train"].column_names, desc="tokenize")
    tokenized = tokenized.filter(lambda row: any(x != -100 for x in row["labels"]))

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(OUT),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=EPOCHS,
        logging_steps=10,
        eval_strategy="no",
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(OUT))
    tokenizer.save_pretrained(str(OUT))
    print(f"saved adapter to {OUT}")


if __name__ == "__main__":
    main()
