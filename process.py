from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset
import json

# Load a small model
model_name = "google-t5/t5-small"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Prepare the dataset
# load data from .json
with open('data/english-pirate-translationsjson', 'r') as file:
    # Load the content of the file
    data = json.load(file)

dataset = load_dataset("json", data_files=data)

# Tokenize inputs
def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples]
    targets = [ex["pr"] for ex in examples]
    model_inputs = tokenizer(inputs, max_length=32, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=32, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

tokenized_data = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./pirate_translator",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=10,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
)

trainer.train()
