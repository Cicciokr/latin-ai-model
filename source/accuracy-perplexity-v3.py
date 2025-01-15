import math
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from datasets import load_dataset
from sklearn.metrics import accuracy_score, recall_score
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

# Configurazione del modello e parametri
#MODEL_NAME = "/content/drive/MyDrive/Colab Notebooks/model/bert-base-latin-uncased"  # Sostituisci con il tuo modello
#MODEL_NAME = "ClassCat/roberta-base-latin-v2"
#MODEL_NAME = "pstroe/roberta-base-latin-cased"
#DATASET_PATH = "dataset.txt"      # Percorso del tuo dataset
MODEL_NAME = "FacebookAI/xlm-roberta-large"
MASK_PROBABILITY = 0.15
BATCH_SIZE = 8

criterion = CrossEntropyLoss()

# Caricamento del tokenizer e del modello
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_per_process_memory_fraction(0.95)
torch.cuda.empty_cache()
model.to(device)
model.eval()

# Caricamento del dataset personalizzato
#Dataset locale
#dataset = load_dataset("parquet", data_dir="./parquet", trust_remote_code=True)
#Dataset caricato da huggingface
#dataset = load_dataset("Cicciokr/CC-100-Latin", revision="refs/convert/parquet")
#Dataset CC 100 lavorato
dataset = load_dataset("text", data_files="./dataset/thelatinlibrary_cleaned.txt")
#dataset = load_dataset("pstroe/cc100-latin", data_files="la.nolorem.tok.latalphabetonly.v2.json", field="train")

wandb.login(key="e115f18967efc277b293c042cb216e776d82a741")
wandb.init(project="metric-calculation-latinlibrary", config={"model_name": MODEL_NAME, "mask_probability": MASK_PROBABILITY, "batch_size": BATCH_SIZE})


# Creazione del data collator per MLM con mascheramento dinamico
data_collator = DataCollatorForWholeWordMask(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=MASK_PROBABILITY
)



# Tokenizzazione del dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)


# Funzione per calcolare Accuratezza e Perplessità
def evaluate_model(dataloader):
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    all_predictions = []  # Lista per accumulare le predizioni
    all_labels = []  # Lista per accumulare i valori reali
    all_probabilities = []  # Lista per accumulare le probabilità

    for batch in dataloader:
        # Prepara un batch di esempio
        #batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        examples = [tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt") for text in batch["text"]]

        # Combina i batch in un unico dizionario
        input_ids = torch.cat([ex["input_ids"] for ex in examples], dim=0).to(device)
        attention_mask = torch.cat([ex["attention_mask"] for ex in examples], dim=0).to(device)
        labels = input_ids.clone()


        # Applica il data collator per mascherare dinamicamente
        #Sposto i tensori sulla cpu perchè altrimenti il DataCollatorForWholeWordMask non funziona
        batch = data_collator([{
            "input_ids": input_ids[i].cpu().tolist(),
            "attention_mask": attention_mask[i].cpu().tolist(),
            "labels": labels[i].cpu().tolist()
        } for i in range(input_ids.size(0))])

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)


        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            logits = outputs.logits
            loss = outputs.loss


        if not torch.isnan(loss):
            total_loss += loss.item() * input_ids.size(0)


            loss_cross_entropy = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Calcola l'accuratezza
            predictions = torch.argmax(logits, dim=-1)

            mask = labels != -100  # Maschera per selezionare solo i token mascherati

            correct = (predictions[mask] == labels[mask]).sum().item()
            total_correct += correct
            total_tokens += mask.sum().item()
            # Accumula predizioni e labels
            all_predictions.extend(predictions[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())

    # Accuratezza
    accuracy = total_correct / total_tokens

    # Perplessità
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    recall = recall_score(all_labels, all_predictions, average='weighted')

    return accuracy, perplexity, total_correct, total_loss, total_tokens, recall, loss_cross_entropy


# Esegui la valutazione
def calculate_metric(dataset, test_size = 0.0001):
  #dataset_split = dataset['train'].train_test_split(test_size=test_size, shuffle=True)
  # Tokenizza i dati
  dataset_split = dataset['train'][:test_size]
  tokenized_datasets = dataset_split.map(tokenize_function, batched=True)
  eval_dataloader = DataLoader(tokenized_datasets, batch_size=BATCH_SIZE, shuffle=True)
  accuracy, perplexity, total_correct, total_loss, total_tokens, recall, loss_cross_entropy = evaluate_model(eval_dataloader)
  """
  Metriche più adatte per MLM:

  Accuratezza: Misura la percentuale di token mascherati previsti correttamente dal modello.
  Perplessità: Rappresenta la capacità del modello di prevedere la prossima parola in una sequenza. Una perplessità più bassa indica una migliore prestazione del modello.
  Recall: Misura la proporzione di token mascherati che il modello è in grado di identificare correttamente tra tutti i token mascherati effettivamente presenti nel testo.
  """

  """
  Stampa i risultati > 80% Ottimo modello || = 70 Buon modello || < 70% Mediocre """
  print(f"Accuratezza: {accuracy*100:.2f}")

  """ Se la loss è 0, la perplessità sarà 1, indicando previsioni perfette. ||| Perplessità > 1 indica un grado di incertezza crescente """
  print(f"Perplessità: {perplexity:.4f}")
  print(f"Recall: {recall:.4f}")  # Stampa il valore del Recall
  print(f"Cross Entropy Loss: {loss_cross_entropy:.2f}")
  print(f"Total correct: {total_correct:.2f}")
  """
  Calcola il loss totale per la perplessità
  Valore basso (es. 0.1 - 0.5): Il modello sta facendo previsioni accurate.
  Valore medio (es. 1 - 2): Prestazioni accettabili, ma con margine di miglioramento.
  Valore alto (> 2): Modello con scarsa performance; potrebbe richiedere miglioramenti nei dati o nell'addestramento.
  """
  print(f"Total Loss: {total_loss:.2f}")
  print(f"Total Token: {total_tokens:.2f}")
  print(f"Length Dataset: {len(tokenized_datasets):.2f}")

  wandb.log({
      "accuracy": accuracy,
      "perplexity": perplexity,
      "recall": recall,
      "cross_entropy_loss": loss_cross_entropy,
      "total_correct": total_correct,
      "total_loss": total_loss,
      "total_tokens": total_tokens,
      "dataset_size": len(tokenized_datasets)
  })


calculate_metric(dataset, 5000)

wandb.finish()