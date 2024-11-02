from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from tqdm import tqdm
import json

# Set the device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load a small model
model_name = "google-t5/t5-small"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Prepare the dataset
dataset = load_dataset("json", data_files="data/english-pirate-translations.json")

# Split the dataset into train and validation sets
dataset = dataset["train"].train_test_split(test_size=0.1)

def preprocess_function(examples):
    # Add a prefix to help T5 understand the task
    inputs = ["translate English to Pirate: " + text for text in examples["en"]]
    targets = examples["pr"]
    
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

print("Preprocessing datasets...")
tokenized_train = dataset["train"].map(preprocess_function, batched=True)
tokenized_validation = dataset["test"].map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./pirate_translator",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    save_steps=10,
    save_total_limit=2,
    logging_steps=1,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
)

print("Starting training...")
trainer.train()
print("Training completed!")

def translate_to_pirate(text):
    # Add the task prefix
    input_text = "translate English to Pirate: " + text
    
    # Prepare the input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate translation with specific parameters
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=0.7
    )
    
    # Decode the output
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return translation

# Test examples
test_texts = [
    "Hello, how are you?",
    "I need to find my ship",
    "Let's go on an adventure"
]

print("\nTesting the translator:")
print("-----------------------")
for text in tqdm(test_texts, desc="Translating"):
    pirate_text = translate_to_pirate(text)
    print(f"English: {text}")
    print(f"Pirate: {pirate_text}\n")