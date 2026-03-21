import os
import json
import random
import time
from datasets import load_dataset
import google.generativeai as genai

def generate_fine_tuning_dataset(num_samples=1000, output_file="models/local_mvm2_adapter/mvm2_training_data.jsonl"):
    """
    Downloads GSM8K and uses Gemini 2.5 Flash to automatically convert the solutions
    into the strict MVM2 JSON Triplet format for QLoRA fine-tuning.
    """
    print(f"Loading GSM8K to generate {num_samples} training examples...")
    dataset = load_dataset("gsm8k", "main", split="train")
    indices = random.sample(range(len(dataset)), num_samples)
    
    # Configure Gemini as our "Teacher" model to generate synthetic JSON traces
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBM0LGvprdpevZXTE4IqlSLv0y74aBGhRc")
    genai.configure(api_key=GEMINI_API_KEY)
    teacher = genai.GenerativeModel('gemini-2.5-flash')
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    success_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, idx in enumerate(indices):
            problem = dataset[idx]["question"]
            raw_solution = dataset[idx]["answer"]
            
            prompt = f"""
            You are creating a fine-tuning dataset for a math reasoning model.
            Convert this problem and solution into the strict MVM2 Triplet schema.
            
            Problem: {problem}
            Raw Solution: {raw_solution}
            
            You MUST return ONLY a raw JSON object matching this schema:
            {{
                "final_answer": "The numerical final answer",
                "reasoning_trace": ["step 1 equation", "step 2 equation"],
                "confidence_explanation": "A statement confirming the algebraic steps are sound."
            }}
            """
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = teacher.generate_content(prompt)
                    text = response.text.replace("```json", "").replace("```", "").strip()
                    json_data = json.loads(text)
                    
                    # Create the instruction-tuned row
                    training_row = {
                        "messages": [
                            {"role": "system", "content": "You are an MVM2 math reasoning agent. You strictly output JSON triplets: {final_answer, reasoning_trace, confidence_explanation}."},
                            {"role": "user", "content": problem},
                            {"role": "assistant", "content": json.dumps(json_data)}
                        ]
                    }
                    
                    f.write(json.dumps(training_row) + "\n")
                    success_count += 1
                    print(f"Successfully generated triplet for problem {i+1}/{num_samples}")
                    break
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        print(f"Rate limited. Sleeping 20s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(20)
                    else:
                        print(f"Skipping problem {i+1} due to parsing error: {e}")
                        break
                        
    print(f"\n✅ Dataset generation complete! Created {success_count} perfect MVM2 triplets at {output_file}")
    print("You can now run `python scripts/train_qlora_math_agent.py` to commence fine-tuning.")

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    generate_fine_tuning_dataset(num_samples=100) # Quick start with 100 samples
