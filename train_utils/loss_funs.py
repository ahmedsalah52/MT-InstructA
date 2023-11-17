import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativeMSELoss(nn.Module):
    def __init__(self,args):
        super(RelativeMSELoss, self).__init__()
        
        self.mag_weight = args.mag_weight
    def forward(self, predicted, ground_truth):
        # Normalize sequences
        #norm_predicted    = F.normalize(predicted,  dim=-1)
        #norm_ground_truth = F.normalize(ground_truth,dim=-1)
        norm_predicted       = predicted/torch.max(predicted,dim=-1,keepdim=True)[0]
        norm_ground_truth = ground_truth/torch.max(ground_truth,dim=-1)[0]
        # Calculate MSE on normalized sequences
        mse_loss = F.mse_loss(norm_predicted, norm_ground_truth)
        
        # calculate the magnitude of the predicted vector
        mag_predicted    = torch.norm(predicted, dim=-1)
        mag_ground_truth = torch.norm(ground_truth, dim=-1)
        mag_loss = F.mse_loss(mag_predicted, mag_ground_truth)

        return mse_loss + self.mag_weight * mag_loss

class MSE(nn.Module):
    def __init__(self,args):
        super(MSE, self).__init__()
        self.loss = nn.MSELoss()
    def forward(self, predicted, ground_truth):
        return self.loss(predicted, ground_truth)