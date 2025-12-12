import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import os
import matplotlib.pyplot as plt
from model import ISRM_Architected
from tqdm import tqdm

CONFIG = {
    "model_name": "distilbert-base-uncased",
    "latent_dim": 8,
    "batch_size": 16,      
    "epochs": 20,      
    "learning_rate": 2e-5, 
    "data_path": "dataset/isrm_dataset_final.json",
    "save_path": "model/isrm/isrm_v3_finetuned.pth",
    "plot_path": "plots/model_plots/training_curve.png"
}

class ISRMDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=128):
        with open(json_path, 'r') as f: self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"Context: {item.get('dialogue_history','')} \n User: {item.get('user_last_message','')}"
        enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        vec = torch.tensor(item['state_vector'], dtype=torch.float32)
        return {'ids': enc['input_ids'].squeeze(0), 'mask': enc['attention_mask'].squeeze(0), 'vec': vec}


mse_loss = nn.MSELoss(reduction='mean')

def loss_fn(recon_x, x, mu, logvar):
    MSE = mse_loss(recon_x, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + 0.0001 * KLD 

def train():
    print("🚀 Starting Fine-Tuning (Last 2 Layers Unfrozen)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    full_dataset = ISRMDataset(CONFIG['data_path'], tokenizer)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    model = ISRM_Architected(CONFIG['model_name'], CONFIG['latent_dim']).to(device)
    
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=len(train_loader)*CONFIG['epochs'])
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            optimizer.zero_grad()
            ids, mask, vec = batch['ids'].to(device), batch['mask'].to(device), batch['vec'].to(device)
            z, mu, logvar = model(ids, mask)
            loss = loss_fn(z, vec, mu, logvar)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                ids, mask, vec = batch['ids'].to(device), batch['mask'].to(device), batch['vec'].to(device)
                z, mu, logvar = model(ids, mask)
                loss = loss_fn(z, vec, mu, logvar)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1:02}: Train Loss {avg_train_loss:.6f} | Val Loss {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CONFIG['save_path'])
            print("   ✅ Saved Best Model")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    os.makedirs(os.path.dirname(CONFIG['plot_path']), exist_ok=True)
    plt.savefig(CONFIG['plot_path'])
    print(f"Saved training plot to {CONFIG['plot_path']}")