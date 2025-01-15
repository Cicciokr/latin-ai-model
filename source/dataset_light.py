import pandas as pd
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import RobertaTokenizerFast, RobertaConfig, RobertaForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
import evaluate
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
#dataset testo
#dataset = load_dataset('text', data_files='./dataset/la.txt')
#dataset parquet
dataset = load_dataset("parquet", data_dir="./parquet", trust_remote_code=True)
#dataset = load_dataset("Cicciokr/CC-100-Latin", revision="refs/convert/parquet")
print(dataset)
#tokenizer = RobertaTokenizerFast(
#    vocab_file="./latinroberta-vocab.json",
#    merges_file="./latinroberta-merges.txt",
#)
tokenizer = AutoTokenizer.from_pretrained("./model/roberta-base-latin-v2")
print(tokenizer.mask_token)

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=tokenizer.model_max_length, return_special_tokens_mask=True)

tokenized_dataset = dataset['train'].select(range(50000, 100000)).map(preprocess_function, batched=True)
tokenized_dataset.save_to_disk("./dataset_light_50000_test/tokenized_dataset")

