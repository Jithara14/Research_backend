#!/usr/bin/env python3
"""
Sinhala Emotion Classification Training Script
Model: sinbert_sinhala_best

Includes:
- Dataset loading (Sinhala + labels)
- Label normalization
- Train/val/test split
- XLM-RoBERTa model fine-tuning
- Accuracy, F1 scores, confusion matrix
- Best model saving
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
DATASET_PATH = "sinhala_emotion_dataset_6000_unique.xlsx"
MODEL_NAME = "xlm-roberta-base"       # Same architecture used previously
SAVE_DIR = "sinbert_sinhala_best"

MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 6
LR = 2e-5

LABEL_MAP = {
    "happy": 0,
    "sad": 1,
    "fear": 2,
    "angry": 3,
    "surprise": 4,
    "neutral": 5
}

# Reverse map for saving in config
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}


# ------------------------------------------------------
# DATASET CLASS
# ------------------------------------------------------
class SinhalaEmotionDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):

        sentence = str(self.sentences[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------
print("📥 Loading dataset:", DATASET_PATH)
df = pd.read_excel(DATASET_PATH)

# Dataset columns: 'id', 'emotion_set', 'tweets'
df["text"] = df["tweets"].astype(str)

# Normalize labels
df["label"] = df["emotion_set"].str.lower().str.strip()
df["label"] = df["label"].map(LABEL_MAP)

df = df.dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Total samples:", len(df))
print(df.head())


# ------------------------------------------------------
# TRAIN / VAL / TEST SPLIT
# ------------------------------------------------------
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


# ------------------------------------------------------
# TOKENIZER & DATALOADERS
# ------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = SinhalaEmotionDataset(
    train_df["text"].tolist(),
    train_df["label"].tolist(),
    tokenizer
)

val_dataset = SinhalaEmotionDataset(
    val_df["text"].tolist(),
    val_df["label"].tolist(),
    tokenizer
)

test_dataset = SinhalaEmotionDataset(
    test_df["text"].tolist(),
    test_df["label"].tolist(),
    tokenizer
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# ------------------------------------------------------
# MODEL
# ------------------------------------------------------
print("🔧 Loading XLM-R model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_MAP),
    id2label=ID2LABEL,
    label2id=LABEL_MAP
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print("🔥 Using device:", device)


# ------------------------------------------------------
# OPTIMIZER & SCHEDULER
# ------------------------------------------------------
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss()


# ------------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------------
def validate(loader):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().numpy())
            trues.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(trues, preds)
    return acc


best_val_acc = 0

print("\n🚀 Training started\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    val_acc = validate(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {total_loss:.4f}  Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print("💾 Saving best model to:", SAVE_DIR)
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)


# ------------------------------------------------------
# FINAL TEST EVALUATION
# ------------------------------------------------------
print("\n📊 Testing model on final test set...")

model.eval()
preds, trues = [], []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1)

        preds.extend(pred.cpu().numpy())
        trues.extend(batch["labels"].cpu().numpy())

print("\nClassification Report:\n")
print(classification_report(trues, preds, target_names=list(LABEL_MAP.keys())))

print("\nConfusion Matrix:\n")
print(confusion_matrix(trues, preds))

print("🎉 Training complete! Model saved to:", SAVE_DIR)
