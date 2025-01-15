import math
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# Configurazione del modello e parametri
MODEL_NAME = "./model/roberta-base-latin-v2"  # Sostituisci con il tuo modello
DATASET_PATH = "dataset.txt"      # Percorso del tuo dataset
MASK_PROBABILITY = 0.15
BATCH_SIZE = 1
EPOCHS = 3
LEARNING_RATE = 1e-5

# Configura il dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Caricamento del tokenizer e del modello
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
torch.cuda.set_per_process_memory_fraction(0.95)
torch.cuda.empty_cache()
model.to(device)

# Caricamento del dataset personalizzato
# Caricamento del dataset personalizzato
dataset = load_dataset("parquet", data_dir="./parquet", trust_remote_code=True)
#dataset = load_dataset("Cicciokr/CC-100-Latin", revision="refs/convert/parquet")
dataset = dataset['train'].train_test_split(test_size=0.00001, shuffle=True)

# Tokenizzazione del dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Tokenizza i dati
tokenized_datasets = dataset['test'].map(tokenize_function, batched=True)

# Creazione del data collator per MLM con mascheramento dinamico
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=MASK_PROBABILITY
)

# Configurazione per l'addestramento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    save_total_limit=2,
    learning_rate=LEARNING_RATE,
    logging_dir="./logs",
    logging_steps=50,
    fp16=True,
    weight_decay=0.01,
    gradient_accumulation_steps=8,
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none"  # Disabilita report a strumenti come WandB
)

# Creazione del trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Avvia l'addestramento
trainer.train()

# Valutazione finale del modello
results = trainer.evaluate()
print(f"Risultati valutazione finale: {results}")