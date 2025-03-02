import torch.nn.functional as F
import torch  
import torchvision.models as models

#https://builtin.com/machine-learning/siamese-network
#Siamese network learns to minimize distance for same pets and maximize for different pets.

# https://www.sciencedirect.com/topics/computer-science/contrastive-loss#:~:text=Contrastive%20loss%20is%20a%20metric,the%20loss%20value%20is%20zero.
# Contrastive loss is a metric learning loss function as
# it calculates the Euclidean distance or cosine similarity between vector pairs.
# It then assigns a loss value based on a predefined margin threshold. 
# If the distance between two vectors is less than the margin threshold, the loss value is zero.



class ContrastiveLoss(torch.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        distance = F.pairwise_distance(emb1, emb2)
        loss = torch.mean(label * distance**2 + (1 - label) * torch.clamp(self.margin - distance, min=0.0)**2)
        return loss