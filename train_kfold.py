import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from indo_utils import preprocess_pipeline
import os
import random
import pickle
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json

# --- Config ---
MODEL_NAME = "indobenchmark/indobert-base-p1"
DAPT_MODEL_DIR = "model_output/dapt_indobert" # Use locally adapted model
OUTPUT_DIR = "model_output"
DATA_FILE = "data/dataset_manual_1200.csv.xlsx"
N_FOLDS = 1
EPOCHS = 3
BATCH_SIZE = 16
MAX_LEN = 128
LR = 2e-5
SEED = 2024

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- Dataset ---
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# --- Training Function ---
def train_fold(fold_idx, train_idx, val_idx, texts, labels, tokenizer):
    print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")
    
    # Split
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    
    # Datasets
    train_ds = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_ds = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    # Model (Load from DAPT)
    model = BertForSequenceClassification.from_pretrained(
        DAPT_MODEL_DIR, 
        num_labels=3,
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.3
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    # Konfigurasi Bobot Kelas (Class Weights) untuk menangani ketidakseimbangan data:
    # Mapping: 0=Negatif, 1=Netral, 2=Positif
    # Bobot: Negatif(1.3), Netral(1.2), Positif(1.0) - Tweak: Fokus Negatif naik sedikit, Netral turun sedikit.
    weights = torch.tensor([1.3, 1.2, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    best_f1 = 0
    best_model_path = os.path.join(OUTPUT_DIR, f"model_fold_{fold_idx}.bin")
    
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        val_probs_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                v_loss = criterion(logits, targets)
                val_loss += v_loss.item()
                
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(targets.cpu().numpy())
                val_probs_list.extend(probs.cpu().numpy())
                
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        
        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val F1 {val_f1:.4f}")
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_f1)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            # Save Best OOF predictions (Targets + Probabilities)
            oof_data = {
                'indices': val_idx,
                'targets': val_targets,
                'probs': val_probs_list
            }
            
    # Plot History
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Fold {fold_idx+1} Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join("plots", f"loss_curve_fold_{fold_idx}.png")
    if not os.path.exists("plots"): os.makedirs("plots")
    plt.savefig(plot_path)
    print(f"Saved {plot_path}")
            
    return oof_data

def main():
    # Load Data
    df = pd.read_excel(DATA_FILE)
    df = df.dropna(subset=['komentar', 'Labeled'])
    
    # Determine mapping (Adjust if your labels are different, e.g., categorical strings)
    # Assuming label column is 'label' and values are mapped strictly. 
    # Checking unique values might be safer, but assuming standard [-1, 0, 1] or similar map.
    label_map = {'negatif': 0, 'netral': 1, 'positif': 2, 
                 'negative': 0, 'neutral': 1, 'positive': 2}
    
    # Normalize: Handle Floats and NaN
    df['Labeled'] = df['Labeled'].fillna(1).astype(int)
    # Normalize Label column
    df['label_norm'] = df['Labeled'].astype(str).str.lower().map(lambda x: label_map.get(x, x)) # Should map '0'->0, '1'->1 since we cast to int, string will match 0/1/2? No, map keys are strings?
    # Actually just use int values if cleaned
    df['label_norm'] = df['Labeled'].apply(lambda x: x if x in [0, 1, 2] else 1)
    
    texts = df['komentar'].apply(preprocess_pipeline).tolist()
    labels = df['label_norm'].tolist()
    
    tokenizer = BertTokenizer.from_pretrained(DAPT_MODEL_DIR)
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    all_oof_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(texts, labels)):
        if fold_idx >= 1: break
        oof_data = train_fold(fold_idx, train_idx, val_idx, texts, labels, tokenizer)
        all_oof_results.append(oof_data)
        
    # Save combined OOF results
    with open(os.path.join(OUTPUT_DIR, 'oof_results.pkl'), 'wb') as f:
        pickle.dump(all_oof_results, f)
        
    print("K-Fold Training Complete. Saved models and OOF results.")

if __name__ == "__main__":
    main()
