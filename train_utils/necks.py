import torch.nn as nn
class transformer_encoder(nn.Module):
    def __init__(self, args):
        super(transformer_encoder, self).__init__()
        self.att_head_emp = args.att_head_emp
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.att_head_emp, nhead=args.n_heads)
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=args.neck_layers)
        

    def forward(self, embeddings):
        embeddings = embeddings.reshape(embeddings.shape[0],-1,self.att_head_emp)
        embeddings = self.model(embeddings)
        return embeddings
    

      
    def get_opt_params(self,args):
        return  [
            {"params": self.model.parameters()}
             ]

