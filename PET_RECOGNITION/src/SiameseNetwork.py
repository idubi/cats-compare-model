import torch.nn as nn
from FeatureExtractot import FeatureExtractor 

class SiameseNetwork(nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()
        self.feature_extractor = FeatureExtractor(model_name)
        self.fc = nn.Linear(self.feature_extractor.embedding_dim, 128)  # Reduce to 128D

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x

    def forward(self, img1, img2):
        emb1 = self.forward_once(img1)
        emb2 = self.forward_once(img2)
        return emb1, emb2

 