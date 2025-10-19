import torch
import torch.nn as nn
from .torch_baselayer import MultiHeadAttention, EncoderLayer, TimeShiftedPositionalEncoding

class sahp_Encoder(nn.Module):
    """Encoder part of Self-Attentive Hawkes Process"""
    
    def __init__(self, num_types, d_model, n_head, n_layers, dropout, pad_token_id):
        super(sahp_Encoder, self).__init__()
        
        self.d_model = d_model
        self.num_types = num_types
        self.pad_token_id = pad_token_id
        
        self.type_emb = nn.Embedding(num_types, d_model, padding_idx=pad_token_id)
        self.position_emb = TimeShiftedPositionalEncoding(d_model=d_model, device=None)
        
        self.stack_layers = nn.ModuleList([
            EncoderLayer(
                d_model,
                MultiHeadAttention(n_head, d_model, d_model, dropout, output_linear=False),
                use_residual=True,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, event_types, event_times, non_pad_mask=None):
        batch_size, seq_len = event_times.size()
        device = event_times.device

        self.position_emb.to(device)
        
        time_delta = torch.zeros_like(event_times)
        time_delta[:, 1:] = event_times[:, 1:] - event_times[:, :-1]
        
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), 
            diagonal=1
        ).unsqueeze(0)  # [1, seq_len, seq_len]
        
        if non_pad_mask is not None:
            padding_mask = ~(non_pad_mask.unsqueeze(-1) & non_pad_mask.unsqueeze(-2))  # [B, seq, seq]
            attention_mask = causal_mask | padding_mask
        else:
            attention_mask = causal_mask.expand(batch_size, -1, -1)
        
        position_embedding = self.position_emb(event_times, time_delta)
        enc_output = position_embedding
        
        if event_types is not None:
            type_embedding = self.type_emb(event_types)
            enc_output += type_embedding
        
        for enc_layer in self.stack_layers:
            enc_output = enc_layer(enc_output, mask=attention_mask)
        
        enc_output = self.norm(enc_output)
        
        return enc_output