
import torch
import torch.nn as nn
from torchvision import models
import config


class ResNet50(nn.Module):
    
    def __init__(self, num_classes=7, pretrained=True, dropout_rate=0.3):
        super(ResNet50, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        
        # Replace FC layer with Dropout + Linear
        if config.USE_DROPOUT and dropout_rate > 0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, num_classes)
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
    
    def get_feature_extractor(self):
        """Get feature extractor (without FC layer)"""
        return nn.Sequential(*list(self.backbone.children())[:-1])


def get_model(num_classes=7, pretrained=True):
   
    dropout_rate = config.DROPOUT_PROB if config.USE_DROPOUT else 0.0
    return ResNet50(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
