import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

def main():
    print("Initializing Unsloth QLoRA Fine-Tuning Pipeline for MVM²...")

    # Configuration
    max_seq_length = 2048 # Good default for math problems
    dtype = None # Auto detects Float16, Bfloat16
    load_in_4bit = True # Use 4-bit quantization to fit on consumer GPUs (e.g. RTX 3090, 4090, T4)
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit" # Excellent 8B model pre-quantized
    
    # 1. Load Model with Unsloth (up to 2x faster, 70% less VRAM)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    
    # 2. Add LoRA Adapters
    # We only train 1-5% of the weights, targeting the specific layers that handle reasoning
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Optimization: 0 is faster
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # 3. Load the generated MVM2 dataset
    dataset_path = "models/local_mvm2_adapter/mvm2_training_data.jsonl"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing dataset {dataset_path}. Run generate_math_dataset.py first!")
        
    dataset = load_dataset('json', data_files=dataset_path, split='train')
    
    # Format the messages using the model's chat template
    def format_chatml(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}
        
    dataset = dataset.map(format_chatml, batched=True)

    # 4. Supervised Fine-Tuning (SFT) Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60, # Set to roughly 1-2 epochs based on dataset size for real training
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )

    # 5. Start Training
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    print("\nStarting QLoRA Fine-Tuning...")
    trainer_stats = trainer.train()
    
    # 6. Save Model
    save_path = "models/local_mvm2_adapter/lora_model"
    print(f"\nSaving LoRA adapters to {save_path}...")
    model.save_pretrained(save_path) # Local saving
    tokenizer.save_pretrained(save_path)
    
    print("\n✅ Fine-Tuning Complete! You can now run the MVM2 Engine completely offline by switching 'use_local_model=True' in llm_agent.py.")

if __name__ == "__main__":
    main()
