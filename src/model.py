import torch
import torch.nn as nn
from transformers import AutoModel

class ISRM_Architected(nn.Module):
    def __init__(self, base_model_name="distilbert-base-uncased", latent_dim=8):
        super(ISRM_Architected, self).__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
                
        for param in self.encoder.parameters():
            param.requires_grad = False

        for layer in self.encoder.transformer.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
                
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)
        self.sigmoid = nn.Sigmoid()

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        mu = self.fc_mu(cls_token)
        logvar = self.fc_logvar(cls_token)
        z = self.reparameterize(mu, logvar)
        
        return self.sigmoid(z), mu, logvar