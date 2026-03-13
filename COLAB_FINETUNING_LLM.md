# Google Colab: LLM Fine-Tuning for Football Commentary

This guide walks you through fine-tuning a language model to generate better football commentary.

---

## **Cell 1: Install Dependencies** (2-3 min)

```python
!pip install -q torch torchvision transformers datasets accelerate bitsandbytes peft pynvml
print("✓ Dependencies installed!")
```

---

## **Cell 2: Mount Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/My Drive')
print("✓ Google Drive mounted!")
```

---

## **Cell 3: Clone Repository**

```python
!git clone https://github.com/sya9423/football-commentary.git
%cd football-commentary
print("✓ Repo cloned!")
print("Files:", os.listdir())
```

---

## **Cell 4: Data & Model Preparation**
*This single block handles dataset creation, model loading, formatting, and tokenization.*

```python
import json
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Generate 1000 lines of robust training data using StatsBomb
# First, install the statsbomb python package we need
import os
os.system('pip install statsbombpy')

# Next, run the script we just wrote! 
# (This uses real match events + an LLM style template to build our JSONL file)
print("Running synthetic data generation...")
os.system('python generate_synthetic_data.py')

print("✓ 1500 pieces of Data created!")

# 2. Load Model & Tokenizer
# Qwen2.5-0.5B is a modern, small but powerful model released in 2024.
# It is far superior to GPT-2 for understanding structured prompts.
model_name = "Qwen/Qwen2.5-0.5B"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
print(f"✓ Model loaded: {model_name} ({model.num_parameters() / 1e6:.0f}M params)")

# 3. Format & Tokenize
dataset = load_dataset('json', data_files='training_data/commentary.jsonl')

def format_example(example):
    return {'text': f"Game: {example['game_state']}\nAction: {example['action']}\nCommentary: {example['commentary']}"}

def tokenize_function(examples):
    tokens = tokenizer(examples['text'], max_length=128, truncation=True, padding='max_length')
    # CRITICAL: Copy input_ids to labels, then mask padding tokens with -100
    # so the model does NOT try to learn the "empty space" at the end of short sentences.
    labels = []
    for ids in tokens['input_ids']:
        label = [tok if tok != tokenizer.pad_token_id else -100 for tok in ids]
        labels.append(label)
    tokens['labels'] = labels
    return tokens

formatted_data = dataset['train'].map(format_example)
tokenized_dataset = formatted_data.map(tokenize_function, remove_columns=['text'], batched=True)

print(f"✓ Data fully formatted and tokenized! Ready for training.")
```

---

## **Cell 5: Training & Saving** (5-10 mins)
*This block configures the trainer, trains the model, and then saves it automatically.*

```python
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# 1. Training Setup
print("🚀 Configuring Trainer...")
training_args = TrainingArguments(
    output_dir='./commentary_model_finetuned',
    num_train_epochs=10,            # More epochs for rich data convergence
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-5,             # Lower LR for a smarter model
    weight_decay=0.01,
    warmup_steps=50,
    bf16=True,                      # Use bfloat16 (matches Qwen2.5 weights)
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    train_dataset=tokenized_dataset,
)

# 2. Train!
print("🔥 Starting training...")
trainer.train()
print("✓ Training COMPLETE!")

# 3. Save Model
model.save_pretrained('./commentary_model_finetuned')
tokenizer.save_pretrained('./commentary_model_finetuned')
print(f"✓ Model successfully saved to: {os.listdir('./commentary_model_finetuned')}")
```

---

## **Cell 6: Test Model Outputs**
*This loads your updated model and tests it with new hypothetical football scenarios so you can review the results before downloading.*

> [!NOTE]
> The model is now trained on 1000 synthetic examples using real StatsBomb match data and a modern Qwen2.5 base model. Results should be significantly more coherent and football-specific.

```python
# 1. Quick Test
print("⚽ Running Model Test...")
fine_tuned_model = AutoModelForCausalLM.from_pretrained('./commentary_model_finetuned')
fine_tuned_tokenizer = AutoTokenizer.from_pretrained('./commentary_model_finetuned')

if torch.cuda.is_available():
    fine_tuned_model = fine_tuned_model.cuda()

def generate_commentary(game_state, action):
    prompt = f"Game: {game_state}\nAction: {action}\nCommentary:"
    inputs = fine_tuned_tokenizer.encode(prompt, return_tensors='pt')
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    
    # We use strict generation parameters to attempt to stop rambling
    outputs = fine_tuned_model.generate(
        inputs, 
        max_new_tokens=40,          # Don't let it talk forever
        num_return_sequences=1, 
        temperature=0.6,            # Lower temp for more predictable text
        top_p=0.9, 
        do_sample=True, 
        pad_token_id=fine_tuned_tokenizer.eos_token_id,
        repetition_penalty=1.2      # Stop it from repeating itself
    )
    
    # Extract only the newly generated text (ignore the prompt itself)
    full_text = fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_commentary = full_text.split("Commentary:")[1].strip() if "Commentary:" in full_text else full_text
    
    return generated_commentary

for state, action in [("1-0, 89th minute", "Player misses open goal"), ("0-0, 5th minute", "Harsh tackle")]:
    print(f"\nPrompt: {state} | {action}\nModel says: {generate_commentary(state, action)}")
```

---

## **Cell 7: Export Model (Optional)**
*If you are satisfied with the model outputs from Cell 6, run this cell to download the model zip file.*

```python
from google.colab import files
import shutil

# 1. Download Model to PC
print("\n📦 Zipping model files...")
shutil.make_archive('commentary_model_finetuned', 'zip', '.', 'commentary_model_finetuned')
print("📥 Triggering download... (check your browser downloads folder)")
files.download('commentary_model_finetuned.zip')
```

---

## **Workflow Complete!**

Once downloaded, you can extract the `.zip` file on your local machine and point your `full_system.py` directly to the `commentary_model_finetuned` directory to use this model.
