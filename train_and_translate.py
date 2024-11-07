from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from tqdm import tqdm
import os
import json

# Set the device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Define paths
MODEL_PATH = "./pirate_translator"
CHECKPOINT_PATH = os.path.join(MODEL_PATH, "best_model")

def train_model():
    # Load a small model
    model_name = "google-t5/t5-small"  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # Prepare the dataset
    dataset = load_dataset("json", data_files="data/pirate2.json")
    
    # Split the dataset into train and validation sets (90-10 split)
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    def preprocess_function(examples):
        inputs = ["translate English to Pirate: " + text for text in examples["en"]]
        targets = [text if text is not None else "" for text in examples["pr"]]  # Replace None with empty strings

        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length").input_ids
        model_inputs["labels"] = labels
        return model_inputs

    print("Preprocessing datasets...")
    tokenized_train = dataset["train"].map(preprocess_function, batched=True)
    tokenized_validation = dataset["test"].map(preprocess_function, batched=True)

    # Calculate training steps
    num_epochs = 20
    batch_size = 4
    num_training_steps = (len(tokenized_train) // batch_size) * num_epochs

    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        eval_strategy="steps",           # Evaluate during training
        eval_steps=50,                   # Evaluate every 50 steps
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        save_strategy="steps",           # Save during training
        save_steps=50,                   # Save every 50 steps
        save_total_limit=3,             # Keep only the 3 best checkpoints
        logging_steps=10,
        logging_dir="./logs",
        learning_rate=5e-5,             # Slightly higher learning rate
        warmup_ratio=0.1,               # 10% of steps for warmup
        weight_decay=0.01,
        load_best_model_at_end=True,    # Load the best model when training ends
        metric_for_best_model="eval_loss"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
    )

    print(f"Starting training with {len(tokenized_train)} examples...")
    trainer.train()
    
    # Save the best model and tokenizer
    trainer.save_model(CHECKPOINT_PATH)
    tokenizer.save_pretrained(CHECKPOINT_PATH)
    print("Training completed and model saved!")

def load_translator():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT_PATH).to(device)
    return model, tokenizer

def translate_to_pirate(text, model, tokenizer):
    input_text = "translate English to Pirate: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=4,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        early_stopping=True,
        length_penalty=1.0
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Check if we need to train or can load existing model
    if not os.path.exists(CHECKPOINT_PATH):
        print("No existing model found. Starting training...")
        train_model()
    
    # Load the model
    model, tokenizer = load_translator()
    
    # Test examples
    test_texts = [
        "Hello, how are you?",
        "I need to find my ship",
        "Let's go on an adventure",
        "The treasure is buried on the island",
        "Watch out for the enemy ships!"
    ]

    print("\nTesting the translator:")
    print("-----------------------")
    for text in tqdm(test_texts, desc="Translating"):
        pirate_text = translate_to_pirate(text, model, tokenizer)
        print(f"English: {text}")
        print(f"Pirate: {pirate_text}\n")
    
    # Interactive mode
    print("\nInteractive Mode (type 'quit' to exit):")
    while True:
        text = input("\nEnter text to translate: ")
        if text.lower() == 'quit':
            break
        pirate_text = translate_to_pirate(text, model, tokenizer)
        print(f"Pirate: {pirate_text}")