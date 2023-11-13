from transformers import Trainer, TrainingArguments, AutoTokenizer, EsmForMaskedLM, TrainerCallback
from torch.utils.data import DataLoader, Dataset, RandomSampler
import pandas as pd
import torch
from torch.optim import AdamW
import wandb
import numpy as np


class ProteinDataset(Dataset):
    def __init__(self, file, tokenizer):
        data = pd.read_csv(file)
        self.tokenizer = tokenizer
        self.proteins = data["Receptor Sequence"].tolist()
        self.peptides = data["Binder"].tolist()

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        protein_seq = self.proteins[idx]
        peptide_seq = self.peptides[idx]

        masked_peptide = '<mask>' * len(peptide_seq)
        complex_seq = protein_seq + masked_peptide

        # Tokenize and pad the complex sequence
        complex_input = self.tokenizer(complex_seq, return_tensors="pt", padding="max_length", max_length = 552, truncation=True)

        input_ids = complex_input["input_ids"].squeeze()
        attention_mask = complex_input["attention_mask"].squeeze()

        # Create labels
        label_seq = protein_seq + peptide_seq
        labels = self.tokenizer(label_seq, return_tensors="pt", padding="max_length", max_length = 552, truncation=True)["input_ids"].squeeze()

        # Set non-masked positions in the labels tensor to -100
        labels = torch.where(input_ids == self.tokenizer.mask_token_id, labels, -100)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        


model_name = "esm2_t33_650M_UR50D"
model = EsmForMaskedLM.from_pretrained("facebook/" + model_name)
tokenizer = AutoTokenizer.from_pretrained("facebook/" + model_name)
lr = 0.0007984276816171436

training_args = TrainingArguments(
    output_dir='./output/',
    num_train_epochs = 5,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 16,
    warmup_steps = 501,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    save_strategy='epoch',
    metric_for_best_model='eval/loss',
    save_total_limit = 5,
    gradient_accumulation_steps=2
)

train_dataset = ProteinDataset("train.csv", tokenizer)
test_dataset = ProteinDataset("test.csv", tokenizer)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    optimizers=(AdamW(model.parameters(), lr=lr), None),
)

trainer.train()
