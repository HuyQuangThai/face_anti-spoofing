import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models.feature_extraction as feature_extraction
from torchvision.models.feature_extraction import create_feature_extractor


class AntiSpoofingModel(nn.Module):
    def __init__(
            self,
            input_channels=80,
            embedding_dim=128,
        ):
        super(AntiSpoofingModel, self).__init__()
        self.projector = nn.Conv2d(
            input_channels,
            embedding_dim,
            kernel_size=1,
        )
        self.embedding_dim = embedding_dim
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.embedding_dim,
            nhead = 8,
            dim_feedforward=256,
            batch_first=True,
            dropout=0.1,
        )
        
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=2
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64,1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.projector(x)
        x_flat = x.view(b, self.embedding_dim, h*w).permute(0, 2, 1)  # [B, H*W, C]
        features = self.encoder(x_flat)  # [B, H*W, C]
        features_reshaped = features.permute(0, 2, 1).view(b, self.embedding_dim, h, w)  # [B, C, H, W]
        spoofing_map = self.decoder(features_reshaped)  # [B, 1, H, W]
        return spoofing_map

class IdentificationModel(nn.Module):
    def __init__(self, input_dim=1280, embedding_dim=512):
        super(IdentificationModel, self).__init__()
        self.recognition = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        
    def forward(self, x, l2_norm=True):
        emb = self.recognition(x)
        if l2_norm:
            emb = F.normalize(emb, p=2, dim=1)
        return emb

    
class Model(nn.Module):
    def __init__(
        self,
        pretrained=True
    ):
        super(Model, self).__init__()
        base_model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        return_nodes = {
            'features.4': 'mid_features',
            'features.7': 'final_features'
        }
        self.backbone = create_feature_extractor(
            base_model,
            return_nodes=return_nodes
        )
        
        self.anti_spoofing = AntiSpoofingModel(
            input_channels=80,
            embedding_dim=128,
        )
        self.identification = IdentificationModel(
            input_dim = 1280,
            embedding_dim=512,
        )
    
    def forward(self, x, update = 'both'):
        features = self.backbone(x)
        mid_features = features['mid_features']  # [B, C, H, W]
        final_features = features['final_features']  # [B, C, H, W]
        
        b, c, h, w = mid_features.size()
        mid_features_reshaped = mid_features.view(b, c, h*w).permute(0, 2, 1)  # [B, H*W, C]
                
        spoofing_map = self.anti_spoofing(mid_features_reshaped, b, c, h, w)  # [B, 1, H', W']
        embeding_map = self.identification(final_features)
        
        if update == 'spoofing':
            embeding_map = embeding_map.detach()
        elif update == 'recognition':
            spoofing_map = spoofing_map.detach()
        else:
            pass
        return spoofing_map, embeding_map
        