import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, RobertaConfig
from datasets import load_dataset, load_from_disk
import numpy as np
import pandas as pd
import math
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24,backend:native,garbage_collection_threshold:0.9,expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Configurazione
MODEL_NAME = "./model/roberta-base-latin-v2"  # Sostituisci con il tuo modello
DATASET_PATH = "./dataset_light_50000_test/tokenized_dataset"      # Percorso dataset locale
MASK_PROBABILITY = 0.15           # Probabilità di mascheramento

# Caricamento modello e tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_per_process_memory_fraction(0.95)
torch.cuda.empty_cache()
model.to(device)

# Caricamento dataset personalizzato
#dataset = load_dataset("text", data_files=DATASET_PATH)

# Tokenizzazione del dataset
#def tokenize_function(examples):
#    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

#tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = load_from_disk(DATASET_PATH)

# Mascheramento dinamico
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=MASK_PROBABILITY
)

# Preparazione dati
samples = tokenized_datasets[:70]  # Usa 100 frasi per esempio
batch = data_collator(samples['input_ids'])
#batch = data_collator([samples[i] for i in range(len(samples))])

# Spostamento su GPU/CPU
input_ids = batch['input_ids'].clone().detach().to(device)
labels = batch['labels'].clone().detach().to(device)
#attention_mask = torch.tensor(batch['attention_mask']).to(device)

# Inference
with torch.no_grad():
    outputs = model(input_ids, labels=labels)
    logits = outputs.logits

# Calcolo Accuracy
masked_indices = labels != -100  # Filtra solo i token mascherati
predictions = torch.argmax(logits, dim=-1)  # Token predetti
correct = (predictions[masked_indices] == labels[masked_indices]).sum().item()
total = masked_indices.sum().item()
accuracy = correct / total

# Calcolo Perplexity
loss = outputs.loss
perplexity = math.exp(loss.item())

# Salvataggio risultati
result_df = pd.DataFrame({
    "Accuracy": [accuracy],
    "Perplexity": [perplexity]
})
result_df.to_csv("mlm_metrics.csv", index=False)

print(f"Predict Total: {total:.2f}")
print(f"Predict Correct: {correct:.2f}")
print(f"Loss: {loss:.2f}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Perplexity: {perplexity:.2f}")