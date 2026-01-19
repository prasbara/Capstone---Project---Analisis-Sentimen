import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from indo_utils import preprocess_pipeline
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from collections import Counter
import argparse

# --- Config ---
MODEL_DIR = "model_output"
PLOT_DIR = "plots"
DAPT_DIR = "model_output/dapt_indobert"
DEFAULT_INPUT = "data/komentar_1bulan.csv"
OUTPUT_COMMENTS = "output_comments.csv"
OUTPUT_TRENDS = "output_trends.csv"
OUTPUT_KEYWORDS = "output_keywords.csv"
BATCH_SIZE = 32
MAX_LEN = 128
N_FOLDS = 1

class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
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
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten()
        }

def optimize_thresholds(oof_results):
    print("Optimizing classification thresholds (Biases)...")
    all_targets = []
    all_probs = []
    
    # Aggregate OOF
    for fold_data in oof_results:
        all_targets.extend(fold_data['targets'])
        all_probs.extend(fold_data['probs'])
        
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    best_f1 = 0
    best_bias = [0, 0, 0]
    
    # Grid Search for Bias [0, bias_neu, 0] etc
    # We essentially want to boost the weak class (often Neutral or Negative)
    # Search range: -0.2 to 0.4
    
    # Simple randomized search for speed
    for _ in range(200):
        # Random biases
        bias = np.random.uniform(-0.1, 0.3, 3)
        
        # Apply bias
        adjusted_probs = all_probs + bias
        preds = np.argmax(adjusted_probs, axis=1)
        
        f1 = f1_score(all_targets, preds, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_bias = bias
            
    print(f"Best OOF F1: {best_f1:.4f} with Bias: {best_bias}")
    # Print report
    adjusted_probs = all_probs + best_bias
    final_preds = np.argmax(adjusted_probs, axis=1)
    print("\nClassification Report (OOF Optimized):")
    print(classification_report(all_targets, final_preds, target_names=['Negatif', 'Netral', 'Positif']))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(all_targets, final_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Netral', 'Positif'], yticklabels=['Negatif', 'Netral', 'Positif'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (OOF Optimized)')
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
    plt.savefig(os.path.join(PLOT_DIR, 'confusion_matrix_oof.png'))
    print(f"Saved confusion_matrix_oof.png to {PLOT_DIR}")
    
    return best_bias

def generate_outputs(df, preds, confidences, args):
    # Map back to labels
    label_map_inv = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
    df['Predicted_Sentiment'] = [label_map_inv[p] for p in preds]
    df['Confidence_Score'] = confidences
    
    # 1. Comments
    out_comments_path = args.output
    print(f"Saving Comments CSV to {out_comments_path}...")
    
    cols = ['username', 'komentar', 'timestamp', 'Predicted_Sentiment', 'Confidence_Score']
    if 'username' not in df.columns: df['username'] = 'user'
    if 'timestamp' not in df.columns: df['timestamp'] = ''
    
    # Final cleanup of columns
    out_df = df.copy()
    # Ensure critical columns exist for output
    if 'Predicted_Sentiment' in out_df.columns:
         out_df.rename(columns={'Predicted_Sentiment': 'sentiment'}, inplace=True)
    if 'Confidence_Score' in out_df.columns:
         out_df.rename(columns={'Confidence_Score': 'confidence'}, inplace=True)
    if 'komentar' in out_df.columns:
         out_df.rename(columns={'komentar': 'comment'}, inplace=True)
         
    # Keep original columns + new ones, but prioritize standard naming if standard cols exist
    target_order = ['username', 'comment', 'timestamp', 'sentiment', 'confidence']
    remaining_cols = [c for c in out_df.columns if c not in target_order]
    x_cols = [c for c in target_order if c in out_df.columns]
    
    out_df = out_df[x_cols + remaining_cols]
    
    out_df.to_csv(out_comments_path, index=False)
    
    # 2. Trends
    print(f"Saving Trends CSV to {OUTPUT_TRENDS}...")
    if 'timestamp' in df.columns and not df['timestamp'].eq('').all():
        df['dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['date'] = df['dt'].dt.date
        trend_pivot = pd.crosstab(df['date'], df['Predicted_Sentiment'])
        # Ensure all columns
        for col in ['Negatif', 'Netral', 'Positif']:
            if col not in trend_pivot.columns: trend_pivot[col] = 0
            
        trend_df = trend_pivot.reset_index()
        trend_df.columns = ['date'] + [c.lower() if c in ['Negatif', 'Netral', 'Positif'] else c for c in trend_df.columns[1:]]
        # Rename specifically to match format
        trend_df = trend_df.rename(columns={'negatif':'negative', 'netral':'neutral', 'positif':'positive'})
        trend_df.to_csv(OUTPUT_TRENDS, index=False)
    else:
        # Create dummy trend if timestamp missing
        print("Warning: Timestamps missing or empty. Creating dummy trend.")
        with open(OUTPUT_TRENDS, 'w') as f: f.write("date,positive,negative,neutral\n")

    # 3. Keywords (Reuse logic)
    print(f"Saving Keywords CSV to {OUTPUT_KEYWORDS}...")
    # Basic stopwords (same as before)
    stop_words = {'yang', 'di', 'dan', 'ini', 'itu', 'ke', 'dari', 'yg', 'gak', 'ada', 'sama', 'bisa', 'kalo', 'buat', 'mau', 'tapi', 'aku', 'saya', 'kamu'}
    
    all_words = []
    word_sent = []
    
    # Process original texts for keywords
    texts_for_kw = df['komentar'].apply(preprocess_pipeline)
    for t, s in zip(texts_for_kw, df['Predicted_Sentiment']):
        ws = [w for w in t.split() if w not in stop_words and len(w)>2]
        all_words.extend(ws)
        for w in ws: word_sent.append((w, s))
        
    counts = Counter(all_words).most_common(50)
    kw_data = []
    
    # Determine dominant sentiment for each keyword
    sent_df = pd.DataFrame(word_sent, columns=['word', 'sentiment'])
    
    for word, count in counts:
        sub = sent_df[sent_df['word'] == word]
        if not sub.empty:
            dom = sub['sentiment'].mode()[0]
        else:
            dom = 'Netral'
        kw_data.append({'keyword': word, 'count': count, 'sentiment': dom})
        
    pd.DataFrame(kw_data).to_csv(OUTPUT_KEYWORDS, index=False)


def main():
    parser = argparse.ArgumentParser(description='Run Sentiment Analysis Inference')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT, help='Path to input file (xlsx or csv)')
    parser.add_argument('--output', type=str, default=OUTPUT_COMMENTS, help='Path to output csv file')
    args = parser.parse_args()

    # 1. Optimize
    try:
        with open(os.path.join(MODEL_DIR, 'oof_results.pkl'), 'rb') as f:
            oof_results = pickle.load(f)
        best_bias = optimize_thresholds(oof_results)
    except Exception as e:
        print(f"Warning: Could not load OOF Results ({e}). Using default bias [0,0,0].")
        best_bias = [0, 0, 0]
    
    # 2. Inference Ensemble
    print(f"Loading Unlabeled Data from: {args.input}")
    
    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
    else:
        df = pd.read_excel(args.input)

    # Rename timestamp column if needed
    if 'tanggal_komentar' in df.columns:
        df.rename(columns={'tanggal_komentar': 'timestamp'}, inplace=True)

    # Validate column
    if 'komentar' not in df.columns:
        print(f"ERROR: Input file must contain a 'komentar' column. Found: {df.columns.tolist()}")
        print("Please rename your text column to 'komentar' using --input file.")
        exit(1)

    df = df.dropna(subset=['komentar']) # Drop NaN comments to align with texts
    texts = df['komentar'].apply(preprocess_pipeline).tolist()
    
    tokenizer = BertTokenizer.from_pretrained(DAPT_DIR)
    dataset = InferenceDataset(texts, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    avg_probs = np.zeros((len(texts), 3))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Running Ensemble Inference...")
    for i in range(N_FOLDS):
        model_path = os.path.join(MODEL_DIR, f"model_fold_{i}.bin")
        if not os.path.exists(model_path):
             print(f"Warning: {model_path} not found. Skipping.")
             continue

        print(f"Loading Fold {i} model...")
        model = BertForSequenceClassification.from_pretrained(DAPT_DIR, num_labels=3)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        fold_probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=mask)
                probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
                fold_probs.extend(probs)
        
        avg_probs += np.array(fold_probs)
        
    avg_probs /= N_FOLDS
    
    # --- PERBAIKAN BERBASIS KEYWORD (HYBRID) ---
    from indo_utils import POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS, NEUTRAL_KEYWORDS
    
    print("Menerapkan Keyword Refinement...")
    keyword_bias = 0.8 # Boost kuat
    
    for idx, text in enumerate(texts):
        # Cek Negatif
        if any(w in text for w in NEGATIVE_KEYWORDS):
            avg_probs[idx][0] += keyword_bias
            
        # Cek Netral
        if any(w in text for w in NEUTRAL_KEYWORDS):
             avg_probs[idx][1] += keyword_bias 
             
        # Cek Positif
        if any(w in text for w in POSITIVE_KEYWORDS):
            avg_probs[idx][2] += keyword_bias
            
    # Apply Bias
    final_conf = np.max(avg_probs, axis=1) 
    
    biased_probs = avg_probs + best_bias
    final_preds = np.argmax(biased_probs, axis=1)
    
    # Generate Output
    generate_outputs(df, final_preds, final_conf, args)
    print("Ensemble Pipeline Complete.")

if __name__ == "__main__":
    main()
