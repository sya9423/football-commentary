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

# 1. Create Data
os.makedirs('training_data', exist_ok=True)
training_data = [
    {"game_state": "0-0, 5th minute", "action": "Player passes to teammate", "commentary": "Nice possession football here, the team is keeping it tight!"},
    {"game_state": "1-0, 15th minute", "action": "Player shoots and scores", "commentary": "GOAAAAAL! What a brilliant finish! The striker makes no mistake!"},
    {"game_state": "1-1, 30th minute", "action": "Goalkeeper makes save", "commentary": "Fantastic save by the keeper! That could have been 2-0!"},
    {"game_state": "1-1, 45th minute", "action": "Player gets yellow card", "commentary": "That's a warning for the player - he needs to be careful now"},
    {"game_state": "2-1, 60th minute", "action": "Player dribbles past defender", "commentary": "Magnificent skill! He glides past two defenders like they're not there!"},
    {"game_state": "2-1, 75th minute", "action": "Corner kick taken", "commentary": "In comes the corner... It's a dangerous delivery!"},
    {"game_state": "2-2, 85th minute", "action": "Substitute comes on", "commentary": "Fresh legs on the pitch - the manager's making a tactical change"},
    {"game_state": "2-2, 90th minute", "action": "Whistle for end of match", "commentary": "And that's the final whistle! What a thrilling encounter!"},
    {"game_state": "0-0, 10th minute", "action": "Player makes tackle", "commentary": "Strong defending! That's a proper block!"},
    {"game_state": "1-0, 40th minute", "action": "Offside decision", "commentary": "No, he's offside! The flag is up!"},
]

with open('training_data/commentary.jsonl', 'w') as f:
    for item in training_data:
        f.write(json.dumps(item) + '\n')
print("✓ Data created!")

# 2. Load Model & Tokenizer
model_name = "gpt2" # Lightweight for Colab
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("✓ Model & Tokenizer loaded!")

# 3. Format & Tokenize
dataset = load_dataset('json', data_files='training_data/commentary.jsonl')

def format_example(example):
    return {'text': f"Game: {example['game_state']}\nAction: {example['action']}\nCommentary: {example['commentary']}"}

def tokenize_function(examples):
    return tokenizer(examples['text'], max_length=128, truncation=True, padding='max_length')

formatted_data = dataset['train'].map(format_example)
tokenized_dataset = formatted_data.map(tokenize_function, remove_columns=['text'], batched=True)

print(f"✓ Data fully formatted and tokenized! Ready for training.")
```

---

## **Cell 5: Training & Saving** (5-10 mins)
*This block configures the trainer, trains the model, and then saves it automatically.*

```python
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

# 1. Training Setup
print("🚀 Configuring Trainer...")
training_args = TrainingArguments(
    output_dir='./commentary_model_finetuned',
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Small batch to prevent OOM
    gradient_accumulation_steps=2,
    save_steps=50,
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=50,
    # Note: 'overwrite_output_dir' is excluded here to prevent a TypeError 
    # depending on your specific transformers version in Colab.
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorWithPadding(tokenizer),
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

## **Cell 6: Test & Export**
*This loads your updated model, tests it with new hypothetical football scenarios, and downloads the zip file locally.*

```python
from google.colab import files
import shutil

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
    
    outputs = fine_tuned_model.generate(
        inputs, max_length=100, num_return_sequences=1, 
        temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=fine_tuned_tokenizer.eos_token_id
    )
    return fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)

for state, action in [("1-0, 89th minute", "Player misses open goal"), ("0-0, 5th minute", "Harsh tackle")]:
    print(f"\nPrompt: {state} | {action}\nModel says: {generate_commentary(state, action)}")

# 2. Download Model to PC
print("\n📦 Zipping model files...")
shutil.make_archive('commentary_model_finetuned', 'zip', '.', 'commentary_model_finetuned')
print("📥 Triggering download... (check your browser downloads folder)")
files.download('commentary_model_finetuned.zip')
```

---

## **Workflow Complete!**

Once downloaded, you can extract the `.zip` file on your local machine and point your `full_system.py` directly to the `commentary_model_finetuned` directory to use this model.
