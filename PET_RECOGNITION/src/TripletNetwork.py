import torch.nn as nn
from FeatureExtractot import FeatureExtractor 


class TripletNetwork(nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()
        self.feature_extractor = FeatureExtractor(model_name)
        self.fc = nn.Linear(self.feature_extractor.embedding_dim, 128)

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x

    def forward(self, anchor, positive, negative):
        anchor_emb = self.forward_once(anchor)
        positive_emb = self.forward_once(positive)
        negative_emb = self.forward_once(negative)
        return anchor_emb, positive_emb, negative_emb

# Test model
