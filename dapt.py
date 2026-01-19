import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import os
import random
import numpy as np
from indo_utils import preprocess_pipeline

# --- Config ---
MODEL_NAME = "indobenchmark/indobert-base-p1"
OUTPUT_DIR = "model_output/dapt_indobert"
FILE_UNLABELED = "data/Dataset Komentar Instagram @pemkotsemarang.xlsx"
FILE_LABELED = "data/dataset_manual_1200.csv.xlsx"
EPOCHS = 1
BATCH_SIZE = 8
SEED = 42
LR = 5e-5

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()
        
        # Create Mask
        labels = input_ids.clone()
        prob_matrix = torch.full(labels.shape, 0.15)
        
        # Correctly get special tokens mask for the whole sequence
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
        prob_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        # padding mask
        prob_matrix.masked_fill_(input_ids == self.tokenizer.pad_token_id, value=0.0)
        
        masked_indices = torch.bernoulli(prob_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        # 80% MASK, 10% random, 10% original
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        # The rest 10% stays original
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def run_dapt():
    print("Loading datasets for DAPT...")
    # Load Unlabeled
    df_u = pd.read_excel(FILE_UNLABELED)
    texts_u = df_u['komentar'].dropna().apply(preprocess_pipeline).tolist()
    
    # Load Labeled
    df_l = pd.read_excel(FILE_LABELED)
    texts_l = df_l['komentar'].dropna().apply(preprocess_pipeline).tolist()
    
    all_texts = texts_u + texts_l
    print(f"Total phrases for DAPT: {len(all_texts)}")
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForMaskedLM.from_pretrained(MODEL_NAME)
    
    dataset = TextDataset(all_texts, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    optimizer = AdamW(model.parameters(), lr=LR)
    
    print(f"Starting DAPT Training on {device}...")
    
    step = 0
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
            if step % 50 == 0:
                print(f"Epoch {epoch+1} | Step {step} | Loss {loss.item():.4f}")
        
        print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(loader):.4f}")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Saving DAPT model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("DAPT Complete.")

if __name__ == "__main__":
    run_dapt()
