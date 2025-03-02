import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()
        
        # Load different models dynamically
        if model_name == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.model.fc = nn.Identity()  # Remove classification layer
            self.embedding_dim = 2048  # ResNet50 output size
        
        elif model_name == "resnet101":
            self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            self.model.fc = nn.Identity()  # Remove classification layer
            self.embedding_dim = 2048  # resnet101 output size            

        elif model_name == "efficientnet_b3":
            self.model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            self.model.classifier[1] = nn.Identity()
            self.embedding_dim = 1536  # EfficientNet-B3 output size
        
        elif model_name == "mobilenet_v3":
            self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            self.model.classifier[3] = nn.Identity()
            self.embedding_dim = 1024  # MobileNetV3 output size

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def forward(self, x):
        return self.model(x)

# Example: Load ResNet50

