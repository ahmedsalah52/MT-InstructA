import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativeMSELoss(nn.Module):
    def __init__(self,mag_weight=0.01):
        super(RelativeMSELoss, self).__init__()
        self.mag_weight = mag_weight
    def forward(self, predicted, ground_truth):
        # Normalize sequences
        norm_predicted    = F.normalize(predicted, p=100, dim=-1)
        norm_ground_truth = F.normalize(ground_truth, p=100,dim=-1)

        # Calculate MSE on normalized sequences
        mse_loss = F.mse_loss(norm_predicted, norm_ground_truth)
        
        # calculate the magnitude of the predicted vector
        mag_predicted    = torch.norm(predicted, dim=-1)
        mag_ground_truth = torch.norm(ground_truth, dim=-1)
        mag_loss = F.mse_loss(mag_predicted, mag_ground_truth)

        return mse_loss + self.mag_weight * mag_loss
