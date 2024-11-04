# gradio for huggingface
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer from the checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "./best_model"

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT_PATH).to(device)

# Gradio function to translate English to Pirate
def translate_to_pirate(text):
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

# Gradio Interface
iface = gr.Interface(
    fn=translate_to_pirate,
    inputs="text",
    outputs="text",
    title="Pirate Translator",
    description="Translate English phrases into Pirate speak!"
)

iface.launch()
