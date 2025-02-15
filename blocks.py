import torch
import torch.nn as nn
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ROPE(nn.Module):
    def __init__(self,embedding_dim:int,base : int = 10_000):

        super().__init__()

        self.base = base
        self.d = embedding_dim
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self,x:torch.Tensor):

        # If the cos and sin values are already cached no need to calculate each time
        if self.cos_cached != None and x.shape[0] <= self.cos_cached.shape[0]:
            return 
        
        seq_length = x.shape[0]

        theta = 1. / (self.base ** torch.arange(0,self.d,2).float() / self.d).to(DEVICE)
        
        seq_idx = torch.arange(0,seq_length,device=DEVICE).float().to(DEVICE)

        idx_theta = torch.einsum('n,d->nd',seq_idx,theta)

        idx_theta2 = torch.cat([idx_theta,idx_theta],dim = 1)

        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x:torch.Tensor):

        d_2 = self.d // 2

        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim = -1)
    
    def forward(self,x:torch.Tensor):

        self._build_cache(x)

        x_rope, x_pass = x[...,:self.d],x[..., self.d:]

        neg_half_x = self._neg_half(x_rope)

        x_rope = (x_rope*self.cos_cached[: x.shape[0]]) + (neg_half_x * self.sin_cached[: x.shape[0]])

        return torch.cat((x_rope,x_pass),dim = -1)
    


class ROPEMHA(nn.MultiheadAttention):
    def __init__(self, heads: int, d_model: int, rope_percentage: float = 0.5, dropout_prob: float = 0.0):

        
        super().__init__(d_model, heads,dropout_prob)


        self.d_rope = int((d_model // heads) * rope_percentage)
        self.query_pos = ROPE(self.d_rope)
        self.key_pos = ROPE(self.d_rope)

    def get_scores(self,query: torch.Tensor, key: torch.Tensor):

        return torch.einsum('ibhd,jbhd->ijbh',self.query_pos(query),self.key_pos(key))
    

class FeedForward(nn.Module):
    def __init__(self,embedding_dim,ff_dim):

        super().__init__()

        self.linear1 = nn.Linear(embedding_dim,ff_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ff_dim,embedding_dim)

    def forward(self,x: torch.Tensor):

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x
    

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim,heads,ff_dim):

        super().__init__()

        self.attn = ROPEMHA(heads,embedding_dim)
        self.ff = FeedForward(embedding_dim,ff_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self,x: torch.Tensor):
        
        norm1 = self.norm1(x)
        attn,_ = self.attn(norm1,norm1,norm1)
        attn = attn + x

        norm2 = self.norm2(attn)
        ff = self.ff(norm2)
        ff = ff + attn

        return ff

class DecoderBlock(nn.Module):
    def __init__(self, n_layers, decoder, embedding_dim, vocab_size):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  
        self.layers = nn.ModuleList([decoder for _ in range(n_layers)])
        self.final = nn.Linear(embedding_dim,vocab_size)
        self.norm = nn.LayerNorm(vocab_size)  

    def forward(self, x: torch.Tensor):
        
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x)

        x = self.final(x)
        x = self.norm(x)
        
        return x