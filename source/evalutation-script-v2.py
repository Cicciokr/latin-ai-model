#https://lewtun.github.io/blog/til/nlp/huggingface/transformers/2021/01/01/til-data-collator.html
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, RobertaConfig, DataCollatorForWholeWordMask
from datasets import load_dataset, load_from_disk, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24,backend:native,garbage_collection_threshold:0.9,expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Configurazione
MODEL_NAME = "./model/roberta-base-latin-v2"  # Sostituisci con il tuo modello
DATASET_PATH = "./dataset_light_50000_test/tokenized_dataset"      # Percorso dataset locale
MASK_PROBABILITY = 0.15           # Probabilità di mascheramento

# Caricamento modello e tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_per_process_memory_fraction(0.9)
torch.cuda.empty_cache()
model.to(device)

# Caricamento dataset personalizzato
#tokenized_datasets = load_from_disk(DATASET_PATH)
dataset = load_dataset("parquet", data_dir="./parquet", trust_remote_code=True)
dataset = dataset['train'].train_test_split(test_size=0.0001, shuffle=True)
#dataset = load_dataset("Cicciokr/CC-100-Latin", revision="refs/convert/parquet", split="train[0:50]")
tokenizer = AutoTokenizer.from_pretrained("./model/roberta-base-latin-v2")
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=tokenizer.model_max_length, return_special_tokens_mask=True)

tokenized_dataset = dataset['test'].map(preprocess_function, batched=True)



# Mascheramento dinamico
data_collator = DataCollatorForWholeWordMask(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=MASK_PROBABILITY,
    return_tensors = "pt"
)


# Preparazione dati
#batch = data_collator(tokenized_dataset)

def loss_per_example(batch):
    batch = data_collator(batch['input_ids'])
    input_ids = batch['input_ids'].clone().detach().to(device)
    #attention_mask = torch.tensor(batch["attention_mask"], device=device)
    labels = batch['labels'].clone().detach().to(device)

    with torch.no_grad():
        output = model(input_ids, labels=labels)
        batch["predicted_label"] = torch.argmax(output.logits, axis=1)

    loss = torch.nn.functional.cross_entropy(output.logits, labels, reduction="mean")
    batch["custom_loss"] = loss
    
    # datasets requires list of NumPy array data types
    for k, v in batch.items():
        batch[k] = v.cpu().numpy()

    return batch


#processed_dataset = Dataset.from_dict({"input_ids": samples['input_ids']})
#processed_dataset.set_format("torch")
losses_ds = tokenized_dataset.map(loss_per_example, batched=True, batch_size=8)

pd.set_option("display.max_colwidth", None)

losses_ds.set_format('pandas')
losses_df = losses_ds[:][['labels', 'predicted_label', 'custom_loss']]
# add the text column removed by the trainer
losses_df['text'] = tokenized_dataset['text']
losses_df.sort_values("custom_loss", ascending=False).head()
result_df = pd.DataFrame(losses_df)
result_df.to_excel("evaluation-script-v2.xlsx", index=False, header=True)