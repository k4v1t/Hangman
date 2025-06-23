import torch.nn as nn
import torch

class HangmanTransformer(nn.Module):

    def __init__(self, vocab_size=28, max_len=10, d_model=256, nhead=4, num_layers=4,
                 dim_feedforward=512, dropout=0.1, ngram_dim=35, aux_dim=19):
        super(HangmanTransformer, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)  # Token dropout

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim + 26 * 4 + ngram_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_model)
        )

        self.cls_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 26)
        )

    def forward(self, input_ids, masked_idx, norm_features, char_multi_hot, ngram_vector):

        # Positional encoding
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = self.embedding_dropout(token_emb + pos_emb)

        x = x.transpose(0, 1)  # Transformer expects seq_len x batch x embed
        transformer_output = self.transformer_encoder(x)
        x = transformer_output.transpose(0, 1)  # Back to batch x seq_len x embed

        # Masked token representation
        masked_idx = masked_idx.bool()
        masked_token_emb = (x * masked_idx.unsqueeze(-1)).sum(dim=1) / masked_idx.sum(dim=1, keepdim=True).clamp(min=1)

        # Flatten and combine auxiliary features
        aux_features = torch.cat([
            norm_features,
            char_multi_hot,
            ngram_vector
        ], dim=1)
        aux_emb = self.aux_mlp(aux_features)

        # Final classifier input: [masked_token_emb || aux_emb]
        cls_input = torch.cat([masked_token_emb, aux_emb], dim=1)
        logits = self.cls_head(cls_input)

        return logits
