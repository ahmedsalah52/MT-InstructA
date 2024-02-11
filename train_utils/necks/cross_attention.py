import torch.nn as nn
import torch
import math
from torch import nn, Tensor 
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class CrossAttentionEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super(CrossAttentionEncoderLayer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout,batch_first=True) 
        self.norm1 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(embed_size, embed_size),
        )
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, conditional_src , src_mask=None):
     

        cross_attended_src,attn_output_weights = self.cross_attention(query=src, key=conditional_src, value=conditional_src)
        
        src = src + self.dropout(cross_attended_src) 
        src = self.norm1(src)


        src = src + self.dropout(self.feed_forward(src))
        src = self.norm2(src)
        return src


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
        self.pos_encoder  = PositionalEncoding(embed_size, max_length)

        self.layers = nn.ModuleList(
            [
                CrossAttentionEncoderLayer(embed_size, num_heads, dropout) for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, conditional_src, mask=None):
        src = self.pos_encoder(src)
        conditional_src = self.pos_encoder(conditional_src)
        
        for layer in self.layers:
            src = layer(src, conditional_src, mask)
        
        

        return src

class CrossAttentionNeck(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emp_size = args.emp_size
        self.cams = args.cams

        """self.encoder = nn.ModuleDict({str(cam):CrossAttentionEncoder(embed_size=args.emp_size,
                                                            num_layers=args.neck_layers, 
                                                            num_heads=args.n_heads, 
                                                            dropout=args.neck_dropout,
                                                            max_length=args.neck_max_len)   
                                                            
                                                            for cam in self.cams})"""
        self.encoder = CrossAttentionEncoder(embed_size=args.emp_size,
                                            num_layers=args.neck_layers, 
                                            num_heads=args.n_heads, 
                                            dropout=args.neck_dropout,
                                            max_length=len(self.cams) ) 

        self.norm = nn.LayerNorm((args.imgs_emps * len(self.cams))
                                 +args.pos_emp
                                 +args.instuction_emps)
        
        self.instruct_dropout = nn.Dropout(args.instruct_dropout)
                                                    
        self.flatten = nn.Flatten()
    def forward(self, input_x,cat=True):
        images_emps,text_emps,pos_emps = input_x
        text_emps = text_emps[:,None,:]
        text_emps   = text_emps.reshape(text_emps.shape[0],-1,self.emp_size)
      
        images_emps = self.encoder(images_emps,text_emps)
        
        if not cat:
            return self.flatten(images_emps),self.flatten(text_emps),pos_emps
        
        
        text_emps = self.instruct_dropout(text_emps)
        
        text_images_embeddings = torch.cat([images_emps,text_emps[:,None,:]],dim=1)
        text_images_embeddings = self.flatten(text_images_embeddings)

      
        return self.norm(torch.cat([text_images_embeddings,pos_emps],dim=1))
         
    def get_opt_params(self):
        return  [
            {"params": self.parameters()}
             ]

