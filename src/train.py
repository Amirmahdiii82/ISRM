import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
import os
import matplotlib.pyplot as plt
from model import ISRM_Architected
from tqdm import tqdm
import numpy as np

CONFIG = {
    "model_name": "distilbert-base-uncased",
    "latent_dim": 3,
    "batch_size": 16,
    "epochs": 15,      
    "learning_rate": 1e-4, 
    "data_path": "dataset/pad_training_data.json",
    "save_path": "model/isrm/pad_encoder.pth",
    "plot_path": "/plots/model_plots/dataset_plots/training_metrics.png"
}

class ISRMDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=128):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"Context: {item.get('dialogue_history','')} \n User: {item.get('user_last_message','')}"
        state_vec = item.get('state_vector')

        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'ids': enc['input_ids'].squeeze(0),
            'mask': enc['attention_mask'].squeeze(0),
            'vec': torch.tensor(state_vec, dtype=torch.float)
        }

def vae_loss_fn(recon_x, x, mu, logvar, kl_weight):

    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + (kl_weight * KLD), MSE, KLD

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on {device} with KL Annealing...")
    
    os.makedirs(os.path.dirname(CONFIG['save_path']), exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG['plot_path']), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    dataset = ISRMDataset(CONFIG['data_path'], tokenizer)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])

    model = ISRM_Architected(CONFIG['model_name'], latent_dim=CONFIG['latent_dim']).to(device)
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])

    history = {'train_loss': [], 'val_loss': [], 'mse': [], 'kld': []}
    best_val_loss = float('inf')

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_train_loss = 0
        total_mse = 0
        total_kld = 0
        
        if epoch < 5:
            kl_weight = 0.0
        elif epoch < 20:
            kl_weight = (epoch - 5) / 15 * 0.001
        else:
            kl_weight = 0.001
            
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{CONFIG['epochs']} [KL:{kl_weight:.5f}]")
        
        for batch in loop:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            vec = batch['vec'].to(device)

            optimizer.zero_grad()
            
            recon_vec, mu, logvar = model(ids, mask)
            
            loss, mse, kld = vae_loss_fn(recon_vec, vec, mu, logvar, kl_weight)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_mse += mse.item()
            total_kld += kld.item()
            
            loop.set_postfix(mse=mse.item()/len(ids))

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_mse = total_mse / len(train_loader.dataset)
        avg_kld = total_kld / len(train_loader.dataset)
        
        history['train_loss'].append(avg_train_loss)
        history['mse'].append(avg_mse)
        history['kld'].append(avg_kld)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['ids'].to(device)
                mask = batch['mask'].to(device)
                vec = batch['vec'].to(device)
                recon, mu, logvar = model(ids, mask)
                v_loss, _, _ = vae_loss_fn(recon, vec, mu, logvar, kl_weight)
                val_loss += v_loss.item()
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(avg_val_loss)

        print(f"   Stats -> MSE: {avg_mse:.4f} | KLD: {avg_kld:.4f} | Best: {best_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CONFIG['save_path'])
            print("   ✅ Model Saved!")

    plot_training(history)

def plot_training(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train Total')
    ax1.plot(history['val_loss'], label='Val Total')
    ax1.set_title('Total Loss')
    ax1.legend()
    
    ax2.plot(history['mse'], label='Reconstruction (MSE)', color='blue')
    ax2.set_ylabel('MSE', color='blue')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(history['kld'], label='Regularization (KLD)', color='orange', linestyle='--')
    ax2_twin.set_ylabel('KLD', color='orange')
    ax2.set_title('MSE vs KLD Breakdown')
    
    plt.savefig(CONFIG['plot_path'])
    print(f"📊 Plots saved to {CONFIG['plot_path']}")

if __name__ == "__main__":
    train()