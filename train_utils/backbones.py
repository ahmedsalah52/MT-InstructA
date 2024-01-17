import torch.nn as nn
import torch
from torch import functional as F

import clip

class Open_AI_CLIP(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.model, self.preprocess_image = clip.load(args.op_image_model_name,jit=False)
        self.model = self.model.float()
        self.pos_emp = nn.Linear(8,args.pos_emp)
        self.flatten = nn.Flatten()
        self.normalize = nn.LayerNorm(args.pos_emp+(args.imgs_emps*len(args.cams))+args.instuction_emps)
        #self.grad_clip = nn.utils.clip_grad_norm_(self.parameters(), 0.5)
    def forward(self,batch,cat=True,vision=True,command=True,pos=True):
        image_features , text_features,pos_embeddings = None,None ,None
        if cat and not (vision and command and pos): 
            raise ValueError('cat is True but one of vision,command,pos is False')
        
        if vision:
            batch_size,cams,ch,h,w  = batch['images'].shape
            batch["images"] = torch.flatten(batch["images"], start_dim=0, end_dim=1)
            image_features = self.model.encode_image(batch['images'])
            image_features = torch.unflatten(image_features,dim = 0,sizes=(batch_size,cams))
            batch["images"] = torch.unflatten(batch["images"],dim = 0,sizes=(batch_size,cams))
        if pos:
            pos_embeddings = self.pos_emp(batch['hand_pos'])

        if command:
            text = clip.tokenize(batch['instruction']).to(self.device)
            text_features  = self.model.encode_text(text)
        
        
        if not cat: 
            return image_features,text_features,pos_embeddings
        
        text_images_embeddings = torch.cat([image_features,text_features[:,None,:]],dim=1)
        text_images_embeddings = self.flatten(text_images_embeddings)
        return self.normalize(torch.cat([text_images_embeddings,pos_embeddings],dim=1))
    
    @property
    def device(self):
        # Return the device of the first parameter of the model
        return next(self.parameters()).device
    def get_opt_params(self):
        return  [
            {"params": self.model.parameters(),"lr": self.args.clip_lr},
            {"params": self.pos_emp.parameters()}
             ]