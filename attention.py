import torch
from torch import nn

class Attention(nn.Module):
    def __init__(
        self,
        d_model=512,
        n_head=8,
        dropout=0.1,
    ):
        super(Attention, self).__init__()

        assert d_model % n_head == 0

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.d = (self.d_head) ** 0.5
        self.w_o = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: (B, L, D)
        mask(optional): (B, L, L)
        """

        bs = q.size(0)

        q = self.w_q(q).view(bs, -1, self.n_head, self.d_head)
        k = self.w_k(k).view(bs, -1, self.n_head, self.d_head)
        v = self.w_v(v).view(bs, -1, self.n_head, self.d_head)

        """
        q, k, v: (B, L, H, D)
        -> qk: (B, H, L, L)
        """
        q = q.transpose(1, 2) # (B, L, H, D) -> (B, H, L, D)
        k = k.permute(0, 2, 3, 1) # (B, L, H, D) -> (B, H, D, L)

        score = torch.matmul(q, k) / self.d # (B, H, L, L)

        if mask is not None:
            """
            1: mask, 0: not mask
            """
            mask = mask.unsqueeze(1).expand_as(score).float()
            mask = mask[mask > 0] = -1e9
            score = score + mask

        """
        score: (B, H, L, L)
        v: (B, L, H, D)
        -> out: (B, H, L, D)
        """
        score = self.dropout(self.softmax(score))
        v = v.permute(0, 2, 1, 3) # (B, L, H, D) -> (B, H, L, D)
        out = torch.matmul(score, v) # (B, H, L, D)

        out = self.dropout(self.w_o(out.transpose(1,2).reshape(bs, -1, self.d_model)))

        return out

if __name__ == "__main__":
    s = torch.rand((2, 4, 512))
    attn = Attention()
    attn(s,s,s)
    print(s)




