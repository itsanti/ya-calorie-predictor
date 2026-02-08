import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel

class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, text_emb, image_emb):
        attn_out, _ = self.cross_attn(text_emb, image_emb, image_emb)
        fused = self.norm(text_emb + attn_out)
        fused = self.norm2(fused + self.ffn(fused))
        return fused

class FiLMAdapter(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gamma = nn.Linear(hidden_dim, hidden_dim)
        self.beta = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, condition):
        return self.gamma(condition) * x + self.beta(condition)

class MultimodalModel(nn.Module):
    def __init__(self, text_unfreeze, image_unfreeze, hidden_dim, dropout):
        super().__init__()

        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        set_requires_grad(self.text_model, text_unfreeze, verbose=True)
        self.text_proj = nn.Linear(768, hidden_dim)

        self.image_model = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
        image_out_dim = self.image_model.num_features
        set_requires_grad(self.image_model, image_unfreeze)
        self.image_proj = nn.Linear(image_out_dim, hidden_dim)

        self.text_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.cross_attention = CrossAttention(hidden_dim, num_heads=4, dropout=dropout)

        self.film = FiLMAdapter(hidden_dim)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, images, mass):
        text_outputs = self.text_model(input_ids=input_ids,
                                       attention_mask=attention_mask)
        text_emb = text_outputs.last_hidden_state
        text_emb = self.text_proj(text_emb)

        img_features = self.image_model(images)
        img_emb = self.image_proj(img_features)

        text_weights = self.text_attention(text_emb).softmax(dim=1)
        text_pooled = (text_emb * text_weights).sum(
            dim=1)


        fused = self.cross_attention(text_pooled.unsqueeze(1),
                                     img_emb.unsqueeze(1))
        fused = fused.squeeze(1)

        img_gated = self.film(img_emb,
                              text_pooled)

        combined = torch.cat([fused, img_gated, mass], dim=1)
        output = self.regressor(combined).squeeze(1)

        return output

def set_requires_grad(module, unfreeze_pattern, verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    patterns = unfreeze_pattern.split("|")
    for name, param in module.named_parameters():
        if any(p in name for p in patterns):
            param.requires_grad = True
            if verbose:
                print(f"Unfrozen: {name}")
        else:
            param.requires_grad = False
