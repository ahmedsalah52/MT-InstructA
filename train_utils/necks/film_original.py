import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM_gen(nn.Module):
    def __init__(self,vision_dim , instruct_dim):
        super(FiLM_gen, self).__init__()

        # Linear layers for conditioning
        self.film = nn.Sequential(nn.Linear(instruct_dim, vision_dim),
                                  nn.ReLU(),
                                  nn.Linear(vision_dim, vision_dim * 2))
        self.vision_dim = vision_dim
        
    def forward(self, instruction_vector):
        # Apply conditioning
        conditioning_output = self.film(instruction_vector)

        # Split the conditioning output into scale and shift parameters
        scale , shift = torch.split(conditioning_output, self.vision_dim, dim=1)
      
        return scale , shift


class FiLM(nn.Module):
    def __init__(self):
        super(FiLM, self).__init__()

    def forward(self, x, gamma, beta):
        return gamma * x + beta

class ResidualBlock(nn.Module):
    def __init__(self, vision_dim ):
        super(ResidualBlock, self).__init__()

        self.film = FiLM()

        self.Linear1 = nn.Linear(vision_dim,vision_dim)
                        
        self.Linear2 = nn.Linear(vision_dim,vision_dim)
        self.batch_norm = nn.BatchNorm1d(vision_dim)
        self.Relu = nn.ReLU()
     

    def forward(self, vision_embeddings, gamma,beta):
        
        vision_embeddings = self.Linear1(vision_embeddings)
        vision_embeddings = self.Relu(vision_embeddings)

        x = self.Linear2(vision_embeddings)
        x = self.batch_norm(x) 
        x = self.film(x,gamma,beta)
        x = self.Relu(x)

        # Add the residual connection
        return vision_embeddings + x

class FilM_neck(nn.Module):
    def __init__(self,args):
        super(FilM_neck, self).__init__()
        N_blocks = args.film_N_blocks
        vision_dim = args.imgs_emps*len(args.cams)
        instruct_dim = args.instuction_emps

        self.film_gens   = nn.ModuleList([FiLM_gen(vision_dim , instruct_dim)  for i in range(N_blocks)])
        self.film_blocks = nn.ModuleList([ResidualBlock(vision_dim)            for i in range(N_blocks)])
        self.flatten = nn.Flatten()
    def forward(self, input_x,cat=True):

        vision_embeddings, instruction_vector, pos_emps = input_x
        vision_embeddings = self.flatten(vision_embeddings)

        gamma_beta = [block(instruction_vector) for block in self.film_gens] # calculate gammas and betas for N blocks


        for i,block in enumerate(self.film_blocks):
            gamma , beta = gamma_beta[i]
            vision_embeddings = block(vision_embeddings, gamma,beta)

        if not cat:
            return vision_embeddings,instruction_vector,pos_emps
        
        return torch.cat([vision_embeddings,instruction_vector,pos_emps],dim=1)
        
    def get_opt_params(self):
        return  [
            {"params": self.parameters()}
             ]

