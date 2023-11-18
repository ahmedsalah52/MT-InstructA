import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMBlock(nn.Module):
    def __init__(self,vision_dim , instruct_dim):
        super(FiLMBlock, self).__init__()

        # Linear layers for conditioning
        self.conditioning_linear1 = nn.Linear(instruct_dim, vision_dim)
        self.conditioning_linear2 = nn.Linear(vision_dim, vision_dim * 2)  # Two times for scale and shift
        self.Relu = nn.ReLU()
    def forward(self, vision_embeddings, instruction_vector):
        # Apply conditioning
        conditioning_output = self.conditioning_linear1(instruction_vector)
        conditioning_output = self.Relu(conditioning_output)
        conditioning_output = self.conditioning_linear2(conditioning_output)

        # Split the conditioning output into scale and shift parameters

        scale , shift = torch.split(conditioning_output, vision_embeddings.size(1), dim=1)
      
        # Apply FiLM modulation
        vision_embeddings = vision_embeddings * scale + shift

        return vision_embeddings

class ResidualBlock(nn.Module):
    def __init__(self, vision_dim , instruct_dim):
        super(ResidualBlock, self).__init__()

        # FiLM block
        self.film_block = FiLMBlock(vision_dim , instruct_dim)

        self.Linear1 = nn.Linear(vision_dim,vision_dim)
                        
        self.Linear2 = nn.Linear(vision_dim,vision_dim)
        self.batch_norm = nn.BatchNorm1d(vision_dim)
        self.Relu = nn.ReLU()
     

    def forward(self, vision_embeddings, instruction_vector):
        
        vision_embeddings = self.Linear1(vision_embeddings)
        vision_embeddings = self.Relu(vision_embeddings)

        x = self.Linear2(vision_embeddings)
        x = self.batch_norm(x)
        x = self.film_block(x,instruction_vector)
        x = self.Relu(x)

        # Add the residual connection
        return vision_embeddings + x

class Film(nn.Module):
    def __init__(self,args):
        super(Film, self).__init__()
        N_blocks = args.film_N_blocks
        vision_dim = args.imgs_emps*len(args.cams)
        instruct_dim = args.instuction_emps


        self.blocks = []
        for i in range(N_blocks):
            self.blocks.append(ResidualBlock(vision_dim , instruct_dim))

        self.blocks = nn.ModuleList(self.blocks)
        self.flatten = nn.Flatten()
    def forward(self, input_x,cat=True):

        vision_embeddings, instruction_vector, pos_emps = input_x
        vision_embeddings = self.flatten(vision_embeddings)

        for block in self.blocks:
            vision_embeddings = block(vision_embeddings, instruction_vector)

        if not cat:
            return vision_embeddings,instruction_vector,pos_emps
        
        return torch.cat([vision_embeddings,instruction_vector,pos_emps],dim=1)
        
    def get_opt_params(self):
        return  [
            {"params": self.parameters()}
             ]

