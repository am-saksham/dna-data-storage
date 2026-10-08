import math
import torch
import torch.nn as nn

class DNAVocabulary:
    """
    Handles tokenization of DNA strings into integer tensors for the neural network.
    Includes special tokens for padding and sequence boundaries.
    """
    def __init__(self):
        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2
        
        self.char2idx = {'A': 3, 'C': 4, 'G': 5, 'T': 6}
        self.idx2char = {3: 'A', 4: 'C', 5: 'G', 6: 'T'}
        self.vocab_size = 7
        
    def encode(self, sequence, max_len=None):
        indices = [self.sos_token] + [self.char2idx[c] for c in sequence if c in self.char2idx] + [self.eos_token]
        if max_len is not None:
            if len(indices) < max_len:
                indices += [self.pad_token] * (max_len - len(indices))
            else:
                indices = indices[:max_len-1] + [self.eos_token]
        return indices
        
    def decode(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return "".join([self.idx2char[idx] for idx in indices if idx in self.idx2char])


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding to give the Transformer a global sense of order.
    """
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class KMerFeatureStem(nn.Module):
    """
    World-Class Upgrade: 1D Convolutional Stem.
    Instead of just embedding individual letters, this layer looks at overlapping 5-mers (5 bases at a time).
    This makes the AI inherently robust to biological frame-shifts (insertions/deletions) 
    because it learns local structural motifs before applying global attention.
    """
    def __init__(self, vocab_size, d_model, kernel_size=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # padding='same' ensures sequence length doesn't change
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, 
                              kernel_size=kernel_size, padding='same')
        self.activation = nn.GELU() # State-of-the-art activation function
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x) # (batch, seq_len, d_model)
        
        # Conv1d expects shape (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.activation(x)
        x = x.transpose(1, 2) # Back to (batch, seq_len, d_model)
        
        return self.layer_norm(x)


class DNADenoiserTransformer(nn.Module):
    """
    State-of-the-Art DNA Sequence-to-Sequence Architecture.
    Combines a K-Mer Convolutional Stem with a Deep Transformer for biological error correction.
    """
    def __init__(self, vocab_size=7, d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Advanced Convolutional Embeddings
        self.src_stem = KMerFeatureStem(vocab_size, d_model)
        self.tgt_stem = KMerFeatureStem(vocab_size, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Core Transformer (Encoder-Decoder)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def generate_square_subsequent_mask(self, sz, device):
        """Prevents the decoder from looking into the future during training."""
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt, pad_idx=0):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, src.device)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)

        src_padding_mask = (src == pad_idx)
        tgt_padding_mask = (tgt == pad_idx)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt)

        # 1. K-Mer Local Feature Extraction + 2. Global Positional Encoding
        src_emb = self.pos_encoder(self.src_stem(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_stem(tgt) * math.sqrt(self.d_model))

        # 3. Global Self-Attention & Cross-Attention
        outs = self.transformer(
            src=src_emb, 
            tgt=tgt_emb, 
            src_mask=src_mask, 
            tgt_mask=tgt_mask, 
            memory_mask=None,
            src_key_padding_mask=src_padding_mask, 
            tgt_key_padding_mask=tgt_padding_mask, 
            memory_key_padding_mask=src_padding_mask
        )
        
        return self.fc_out(outs)
