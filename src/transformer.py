import torch
from torch import nn
import torch.nn.functional as F

from src.base import TransformerBlock

# ================================
# Text Classifier Transformer
# ================================


class CTransformer(nn.Module):
    def __init__(self, k, heads, depth, max_seq_length, vocab_size, num_classes):
        super().__init__()

        self.token_emb = nn.Embedding(
            vocab_size, k
        )  # look up table for word embedding: (vocab_size x k)
        self.position_em = nn.Embedding(
            max_seq_length, k
        )  # look up table for positions

        tblock = []
        for i in range(depth):
            tblock.append(TransformerBlock(k=k, heads=heads, mask=False))

        self.tblock = nn.Sequential(*tblock)

        self.toprobs = nn.Linear(k, num_classes)

    def forward(self, x):
        token_emb = self.token_emb(x)
        b, t, k = token_emb.size()

        positions = torch.arange(t)
        position_emb = self.position_em(positions)[None, :, :].expand(b, t, k)

        x = token_emb + position_emb
        # print('x size inside ctransformer', x.size())

        x = self.tblock(x)
        # average pooling
        x = x.mean(dim=1)
        x = self.toprobs(x)

        return F.log_softmax(x, dim=1)


# test CTransformer cladd
# ctransformer = CTransformer(15, 8, 3, 5, 5, 2)
# x = torch.randn(4, 5, 15)
# res = ctransformer(x)
# print(res.shape)

# class GTransformer(nn.Module):
#     def __init__(self):
#         pass
