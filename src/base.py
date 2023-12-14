import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, k, heads=8, mask=False):
        super().__init__()
        self.k = k #embeding size
        self.heads = heads
        self.mask = mask

        self.to_queries = nn.Linear(k, heads * k, bias=False)
        self.to_keys = nn.Linear(k, heads * k, bias=False)
        self.to_values = nn.Linear(k, heads * k, bias=False)

        self.unifyheads = nn.Linear(heads * k, k)

    def forward(self, x):
        # x dimention is (b, t, k)
        # print('x size inside self attention block', x.size())
        b, t, k = x.size()

        h = self.heads

        keys = self.to_keys(x).view(b, t, h, k)  # the result before view method is (b, t, h * k)
        queries = self.to_queries(x).view(b, t, h, k)
        values = self.to_values(x).view(b, t, h, k)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k) 
        values = values.transpose(1, 2).contiguous().view(b * h, t, k) / (k ** (1 / 4))

        weights = torch.bmm(
            queries, keys.transpose(1, 2)
        )  # the result would be (b * h, t, t) dimension

        weights = weights / (k ** (1 / 2))

        if self.mask:
            indices = torch.triu_indices(t, t, offset=1)
            weights[:, indices[0], indices[1]] = float("-inf")

        weights = F.softmax(weights, dim=2)
        out = torch.bmm(weights, values).view(b, h, t, k)

        out = out.transpose(1, 2).contiguous().view(b, t, h * k)
        out = self.unifyheads(out)

        return out


# ===========================
# test SelfAttention Class
# ===========================
# selfattend = SelfAttention(10)
# x = torch.randn(5, 15, 10)
# result = selfattend(x)
# print(result.shape)
# exit()


class TransformerBlock(nn.Module):
    def __init__(self, k, heads, mask=False):
        super().__init__()

        self.selfattention = SelfAttention(k, heads, mask=mask)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(nn.Linear(k, 4 * k), nn.ReLU(), nn.Linear(4 * k, k))

    def forward(self, x):
        attended = self.selfattention(x)
        x = self.norm1(attended + x)
        feedforward = self.ff(x)
        x = self.norm2(feedforward + x)

        return x


# =============================
# Test TransformerBlock Class
# ============================

# formerblock = TransformerBlock(12, 8, False)
# x = torch.randn(3, 5, 12)
# result = formerblock(x)
# print(result.shape)
