#import pandas as pd
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import RobertaTokenizerFast, RobertaConfig, RobertaForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, logger
from transformers.integrations import TrainerCallback, is_tensorboard_available
from sklearn.model_selection import train_test_split
import torch
import gc
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
#import evaluate
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24,backend:native,garbage_collection_threshold:0.7,expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
#dataset testo
#dataset = load_dataset('text', data_files='la.txt')
#dataset parquet
#dataset = load_dataset("Cicciokr/CC-100-Latin", revision="refs/convert/parquet")
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.device(0))
device = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(0.9)
torch.cuda.empty_cache()
gc.collect()


def custom_rewrite_logs(d, mode):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if mode == 'eval' and k.startswith(eval_prefix):
            if k[eval_prefix_len:] == 'loss':
                new_d["combined/" + k[eval_prefix_len:]] = v
        elif mode == 'test' and k.startswith(test_prefix):
            if k[test_prefix_len:] == 'loss':
                new_d["combined/" + k[test_prefix_len:]] = v
        elif mode == 'train':
            if k == 'loss':
                new_d["combined/" + k] = v
    return new_d


class CombinedTensorBoardCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [TensorBoard](https://www.tensorflow.org/tensorboard).
    Args:
        tb_writer (`SummaryWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, tb_writers=None):
        has_tensorboard = is_tensorboard_available()
        if not has_tensorboard:
            raise RuntimeError(
                "TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or"
                " install tensorboardX."
            )
        if has_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: F401

                self._SummaryWriter = SummaryWriter
            except ImportError:
                try:
                    from tensorboardX import SummaryWriter

                    self._SummaryWriter = SummaryWriter
                except ImportError:
                    self._SummaryWriter = None
        else:
            self._SummaryWriter = None
        self.tb_writers = tb_writers

    def _init_summary_writer(self, args, log_dir=None):
        log_dir = log_dir or args.logging_dir
        if self._SummaryWriter is not None:
            self.tb_writers = dict(train=self._SummaryWriter(log_dir=os.path.join(log_dir, 'train')),
                                   eval=self._SummaryWriter(log_dir=os.path.join(log_dir, 'eval')))

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        log_dir = None

        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                log_dir = os.path.join(args.logging_dir, trial_name)

        if self.tb_writers is None:
            self._init_summary_writer(args, log_dir)

        for k, tbw in self.tb_writers.items():
            tbw.add_text("args", args.to_json_string())
            if "model" in kwargs:
                model = kwargs["model"]
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    tbw.add_text("model_config", model_config_json)
            # Version of TensorBoard coming from tensorboardX does not have this method.
            if hasattr(tbw, "add_hparams"):
                tbw.add_hparams(args.to_sanitized_dict(), metric_dict={})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writers is None:
            self._init_summary_writer(args)

        for tbk, tbw in self.tb_writers.items():
            logs_new = custom_rewrite_logs(logs, mode=tbk)
            for k, v in logs_new.items():
                if isinstance(v, (int, float)):
                    tbw.add_scalar(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            tbw.flush()

    def on_train_end(self, args, state, control, **kwargs):
        for tbw in self.tb_writers.values():
            tbw.close()
        self.tb_writers = None





tokenizer = RobertaTokenizerFast(
    vocab_file="./latinroberta-vocab.json",
    merges_file="./latinroberta-merges.txt",
)
config = RobertaConfig(
    vocab_size=len(tokenizer),
    max_position_embeddings=514,  # Lunghezza massima della sequenza
    hidden_size=180,
    num_attention_heads=6,
    num_hidden_layers=3
)
print(tokenizer.mask_token)

#def preprocess_function(examples):
#    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# Applicare la tokenizzazione
model = RobertaForMaskedLM(config=config)
model.to(device)
#dataset = load_dataset("parquet", data_dir="./parquet", trust_remote_code=True)
#tokenized_dataset = dataset.map(preprocess_function, batched=True)
#tokenized_dataset.save_to_disk("./dataset/tokenized_dataset")
tokenized_dataset = load_from_disk("./dataset_light_50000/tokenized_dataset")


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,                   # Abilita il mascheramento
    mlm_probability=0.15        # Percentuale di token da mascherare
)

#il 20% dei dati viene usato come test e l'80% viene usato come train, per evitare overfitting
#split_dataset = tokenized_dataset['train'].train_test_split(test_size=0.2)
#train_dataset = split_dataset["train"]
#eval_dataset = split_dataset["test"]
print("Allocated Memory:", torch.cuda.memory_allocated(0))
print("Max Memory Allocated:", torch.cuda.max_memory_allocated(0))

training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=True,
    gradient_accumulation_steps=4,
    report_to="tensorboard",
    push_to_hub=False
)

#metric = evaluate.load("accuracy")
#def compute_metrics(eval_pred):
#    logits, labels = eval_pred
#   predictions = np.argmax(logits, axis=-1)
#    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
#    train_dataset=tokenized_dataset['train'],
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[CombinedTensorBoardCallback]
)

trainer.train()

model.save_pretrained("lat-bert")
tokenizer.save_pretrained("lat-bert")


results = trainer.predict(
    test_dataset=tokenized_dataset
)
print(results)