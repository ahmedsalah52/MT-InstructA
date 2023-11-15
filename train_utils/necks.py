import torch.nn as nn
import torch
import math
from torch import nn, Tensor 
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)
        #self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)].to(x.device)
        return self.dropout(x)
    
class transformer_encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emp_size = args.emp_size
        self.pos_encoder  = PositionalEncoding(args.emp_size, dropout=args.neck_dropout, max_len=args.neck_max_len)
        encoder_layer     = nn.TransformerEncoderLayer(d_model=args.emp_size,dropout=args.neck_dropout, nhead=args.n_heads)
        self.encoder      = nn.TransformerEncoder(encoder_layer, num_layers=args.neck_layers)
        

    def forward(self, embeddings):
        shape = embeddings.shape
        embeddings = embeddings.reshape(shape[0],-1,self.emp_size)
        embeddings = embeddings.permute(1,0,2)
        
        embeddings = embeddings * math.sqrt(self.emp_size)
        embeddings = self.pos_encoder(embeddings)
        embeddings = self.encoder(embeddings)
        return embeddings.permute(1,0,2).reshape(*shape)
    

      
    def get_opt_params(self):
        return  [
            {"params": self.encoder.parameters()}
             ]

class CrossAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values  = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys    = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out  = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys     = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = query.reshape(N, query_len, self.num_heads, self.head_dim)

        values  = self.values(values)
        keys    = self.keys(keys)
        queries = self.queries(queries)



        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class CrossAttentionEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super(CrossAttentionEncoderLayer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout,batch_first=True) 
        #CrossAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, conditional_src , src_mask=None):
     

        cross_attended_src,attn_output_weights = self.cross_attention(src, conditional_src, src, src_mask)
        
       
        x = self.dropout(self.norm1(cross_attended_src + src))
        forward = self.feed_forward(x)

        out = self.dropout(self.norm2(forward + x))
        return out


class CrossAttentionEncoder(nn.Module):
    def __init__(
        self,
        embed_size,
        num_layers,
        num_heads,
        dropout,
        max_length,
    ):
        super(CrossAttentionEncoder, self).__init__()

        self.embed_size = embed_size
        self.pos_encoder  = PositionalEncoding(embed_size, dropout=dropout, max_len=max_length)

        self.layers = nn.ModuleList(
            [
                CrossAttentionEncoderLayer(
                    embed_size, num_heads, dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, conditional_src, mask=None):
        src = src.permute(1,0,2)
        conditional_src = conditional_src.permute(1,0,2)

        src = self.pos_encoder(src)

        for layer in self.layers:
            src = layer(src, conditional_src, mask)
        
        
        src = src.permute(1,0,2)

        return src

class CrossAttentionNeck(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emp_size = args.emp_size
       
        self.encoder = nn.ModuleList([CrossAttentionEncoder(embed_size=args.emp_size,
                                                            num_layers=args.neck_layers, 
                                                            num_heads=args.n_heads, 
                                                            dropout=args.neck_dropout,
                                                            max_length=len(args.cams))   for i in range(len(args.cams))])
        
                                                             
        self.flatten = nn.Flatten()
    def forward(self, input_x):

        images_emps,text_emps,pos_emps = input_x
        text_emps = text_emps[:,None,:]
        
        images_emps = images_emps.reshape(images_emps.shape[0],images_emps.shape[1],-1,self.emp_size)
        text_emps   = text_emps.reshape(text_emps.shape[0],-1,self.emp_size)

        batch_size , cams , seq_length, emps = images_emps.shape

        images_emps  = [self.encoder[i](images_emps[:,i],text_emps) for i in range(cams)]
        images_emps  = torch.stack(images_emps,dim=1)
        text_images_embeddings = torch.cat([images_emps,text_emps[:,None,:,:]],dim=1)
        text_images_embeddings = self.flatten(text_images_embeddings)


        return torch.cat([text_images_embeddings,pos_emps],dim=1)
         
    def get_opt_params(self):
        return  [
            {"params": self.encoder.parameters()}
             ]

