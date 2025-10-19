import math
import torch
from torch import nn

from .torch_baselayer import MultiHeadAttention, EncoderLayer


class attnhp_Encoder(nn.Module):
    def __init__(self, num_types, num_layers, d_model, n_head, dropout, pad_token_id):
        super().__init__()
        self.num_types = num_types
        self.d_model = d_model
        self.n_layers = num_layers
        self.n_head = n_head
        self.pad_token_id = pad_token_id

        self.type_emb = nn.Embedding(num_types, d_model, padding_idx=pad_token_id)
        self.dropout = nn.Dropout(p=dropout)
        self.time_encoding = TimePositionalEncoding(d_model)

        # Create independent heads
        self.heads = nn.ModuleList()
        for _ in range(self.n_head):
            head_layers = nn.ModuleList([
                EncoderLayer(
                    d_model=d_model,
                    self_attn=MultiHeadAttention(
                        n_head=1,  # Each head is single-headed
                        d_input=d_model,
                        d_model=d_model,
                        dropout=dropout,
                        output_linear=True
                    ),
                    feed_forward=None,
                    use_residual=True,
                    dropout=dropout
                ) for _ in range(self.n_layers)
            ])
            self.heads.append(head_layers)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, event_types, event_times, non_pad_mask):
        batch_size = event_times.size(0)
        seq_len = event_times.size(1)
        
        # Get embeddings
        time_embed = self.time_encoding(event_times)
        if event_types is not None:
            type_embed = self.type_emb(event_types)
        else:
            type_embed = torch.zeros_like(time_embed)
        
        enc_input = time_embed + type_embed
        enc_input = self.dropout(enc_input)

        # Create attention mask
        if non_pad_mask is not None:
            if non_pad_mask.dim() == 2:
                non_pad_mask = non_pad_mask.unsqueeze(-1)
            attn_mask = non_pad_mask.squeeze(-1).unsqueeze(1)
            attn_mask = attn_mask.expand(-1, seq_len, -1)
        else:
            attn_mask = None

        # Process through each head independently
        head_outputs = []
        for head_idx in range(self.n_head):
            current_input = enc_input
            for layer_idx in range(self.n_layers):
                current_input = self.heads[head_idx][layer_idx](current_input, attn_mask)
            head_outputs.append(current_input)

        # Combine head outputs (average pooling)
        enc_output = torch.stack(head_outputs).mean(dim=0)
        enc_output = self.layer_norm(enc_output)
        
        return enc_output



class TimePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device='cpu'):
        super().__init__()
        i = torch.arange(0, d_model, 1, device=device)
        div_term = (2 * (i // 2).float() * -(math.log(10000.0) / d_model)).exp()
        self.register_buffer('div_term', div_term)

    def forward(self, x):
        result = x.unsqueeze(-1) * self.div_term.to(x.device)
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result
