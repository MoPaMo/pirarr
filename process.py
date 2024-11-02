from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset
import json

# Load a small model
model_name = "google-t5/t5-small"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Prepare the dataset
dataset = load_dataset("json", data_files="data/english-pirate-translations.json")

# Split the dataset into train and validation sets
dataset = dataset["train"].train_test_split(test_size=0.1)  # 10% for validation

# Tokenize inputs
def preprocess_function(examples):
    inputs = examples["en"]
    targets = examples["pr"]
    model_inputs = tokenizer(inputs, max_length=32, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=32, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

# Process both train and validation sets
tokenized_train = dataset["train"].map(preprocess_function, batched=True)
tokenized_validation = dataset["test"].map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./pirate_translator",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,  # Added eval batch size
    num_train_epochs=3,
    save_steps=10,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,  # Added evaluation dataset
)

trainer.train()