# Unified Colab Script for MVM2 QLoRA Training
import os
print("Installing dependencies...")
os.system("pip install -q -U torch datasets trl peft transformers unsloth")

import json
from datasets import load_dataset

print("1. Generating Dataset locally (No API limits!)...")
dataset = load_dataset("gsm8k", "main", split="train")

def format_gsm8k(example):
    parts = example["answer"].split("####")
    reasoning = [step.strip() for step in parts[0].strip().split('\n') if step.strip()]
    final_answer = parts[1].strip() if len(parts) > 1 else ""
    
    json_data = {
        "final_answer": final_answer,
        "reasoning_trace": reasoning,
        "confidence_explanation": "Deterministic symbolic steps logically verified."
    }
    
    return {
        "messages": [
            {"role": "system", "content": "You are an MVM2 math reasoning agent. You strictly output JSON triplets: {final_answer, reasoning_trace, confidence_explanation}."},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": json.dumps(json_data)}
        ]
    }

print("Mapping dataset to MVM2 Triplets...")
formatted_dataset = dataset.map(format_gsm8k, remove_columns=["question", "answer"])
# Use 1000 samples for a fast, targeted training run
small_dataset = formatted_dataset.select(range(1000))

print("2. Starting Unsloth Training on T4 GPU...")
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

def format_chatml(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

train_data = small_dataset.map(format_chatml, batched=True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_data,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        max_steps = 60, # 60 steps for a very fast 5-minute training demonstration
        learning_rate = 2e-4,
        fp16 = True,
        logging_steps = 10,
        optim = "adamw_8bit",
        output_dir = "outputs",
    ),
)

trainer.train()

model.save_pretrained("mvm2_lora_model")
tokenizer.save_pretrained("mvm2_lora_model")
print("\n✅ Training Complete! The LoRA adapter is saved to 'mvm2_lora_model'.")
