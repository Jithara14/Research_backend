#!/usr/bin/env python3
"""
Tamil/Sinhala Emotion Classifier - Corrected Version
- Handles imbalanced data
- Uses class weights and scheduler
- Proper preprocessing
- Train / Validate / Test split
- Saves best model automatically
"""

import os
import re
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, AdamW, get_linear_schedule_with_warmup
)
from tqdm import tqdm
import emoji

# ================= CONFIG =================
DATA_PATH = "Tamil_Emotion_tweets_clean_updated.xlsx"
TEXT_COL = "tweets"
LABEL_COL = "Emotion Set"

MODEL_NAME = "xlm-roberta-base"  # change to 'xlm-roberta-large' if GPU memory allows
BATCH_SIZE = 16
EPOCHS = 8
LR = 3e-5
MAX_LEN = 128
SEED = 42
SAVE_DIR = "best_emotion_model"
USE_GPU = torch.cuda.is_available()

os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ================= DATA CLEANING =================
def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = emoji.demojize(text)  # convert emojis to text
    text = re.sub(r"[^0-9a-zA-Z\u0B80-\u0BFF\u0D80-\u0DFF.,!?\\s]", " ", text)  # keep Tamil & Sinhala
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ================= LOAD DATA =================
df = pd.read_excel(DATA_PATH)
df = df.dropna(subset=[TEXT_COL, LABEL_COL])
df[TEXT_COL] = df[TEXT_COL].apply(clean_text)
df[LABEL_COL] = df[LABEL_COL].str.lower().str.strip()

labels = sorted(df[LABEL_COL].unique())
label2id = {lbl: i for i, lbl in enumerate(labels)}
id2label = {i: lbl for lbl, i in label2id.items()}
df["label_id"] = df[LABEL_COL].map(label2id)

print("Labels:", labels)
print(df[LABEL_COL].value_counts())

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label_id"], random_state=SEED)

# ================= TOKENIZER & DATASET =================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
collator = DataCollatorWithPadding(tokenizer=tokenizer)

class EmotionDataset(Dataset):
    def __init__(self, df):
        self.texts = df[TEXT_COL].tolist()
        self.labels = df["label_id"].tolist()
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], truncation=True, max_length=MAX_LEN, padding=False, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_loader = DataLoader(EmotionDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
val_loader = DataLoader(EmotionDataset(val_df), batch_size=BATCH_SIZE, collate_fn=collator)
test_loader = DataLoader(EmotionDataset(test_df), batch_size=BATCH_SIZE, collate_fn=collator)

# ================= MODEL =================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id
).to(DEVICE)

# ================= LOSS & OPTIMIZER =================
counts = train_df["label_id"].value_counts().sort_index().values
weights = torch.tensor(train_df.shape[0]/(len(labels)*counts), dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

# ================= TRAIN / EVAL =================
def evaluate(loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k,v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=1)
            y_true.extend(batch["labels"].cpu())
            y_pred.extend(preds.cpu())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, f1

best_f1 = 0
patience = 3
patience_counter = 0
grad_clip = 1.0

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        batch = {k:v.to(DEVICE) for k,v in batch.items()}
        optimizer.zero_grad()
        logits = model(**batch).logits
        loss = criterion(logits, batch["labels"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    val_acc, val_f1 = evaluate(val_loader)
    print(f"\nEpoch {epoch} | Train Loss={avg_loss:.4f} | Val Acc={val_acc:.4f} | Val F1={val_f1:.4f}")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        print("🔥 Saving best model...")
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⛔ Early stopping triggered")
            break

# ================= FINAL TEST =================
test_acc, test_f1 = evaluate(test_loader)
print("\n🎉 TEST RESULTS")
print("Accuracy:", test_acc)
print("Macro F1:", test_f1)
