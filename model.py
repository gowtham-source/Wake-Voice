"""
TAN-MSFF: Temporal Attention Network with Multi-Scale Feature Fusion
A novel architecture for wake word/phrase detection optimized for small datasets.

Key innovations:
1. Multi-scale temporal convolutions to capture phonetic patterns at different granularities
2. Squeeze-and-Excitation attention for channel recalibration
3. Temporal self-attention for capturing long-range dependencies in speech
4. Residual connections with learnable scaling for stable training
5. Mixup-ready architecture for improved generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SqueezeExcitation(nn.Module):
    """Channel attention mechanism"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        # x: (B, C, T)
        b, c, t = x.size()
        # Global average pooling
        y = x.mean(dim=2)  # (B, C)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.unsqueeze(2)


class MultiScaleConvBlock(nn.Module):
    """Multi-scale temporal convolution block"""
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        
        # Different kernel sizes for multi-scale feature extraction
        self.conv1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=7, padding=3)
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.se = SqueezeExcitation(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        # Multi-scale convolutions
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)
        
        # Concatenate multi-scale features
        out = torch.cat([out1, out3, out5, out7], dim=1)
        out = self.bn(out)
        out = F.gelu(out)
        out = self.se(out)
        out = self.dropout(out)
        
        # Residual connection with learnable scaling
        return self.residual(x) + self.scale * out


class TemporalSelfAttention(nn.Module):
    """Efficient temporal self-attention"""
    def __init__(self, channels, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)
        
        # Relative positional encoding
        self.rel_pos = nn.Parameter(torch.randn(1, num_heads, 1, 128) * 0.02)
    
    def forward(self, x):
        # x: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        B, T, C = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention with relative position bias
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative positional bias (truncate or pad as needed)
        if T <= 128:
            attn = attn + self.rel_pos[:, :, :, :T]
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        
        return out.transpose(1, 2)  # Back to (B, C, T)


class AttentionBlock(nn.Module):
    """Transformer-style attention block"""
    def __init__(self, channels, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = TemporalSelfAttention(channels, num_heads, dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout)
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        # x: (B, C, T)
        # Pre-norm attention
        x_norm = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.scale * self.attn(x_norm)
        
        # Pre-norm FFN
        x_t = x.transpose(1, 2)  # (B, T, C)
        x_t = x_t + self.scale * self.ffn(self.norm2(x_t))
        
        return x_t.transpose(1, 2)


class StatisticsPooling(nn.Module):
    """Statistics pooling layer - captures mean and std"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # x: (B, C, T)
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        return torch.cat([mean, std], dim=1)


class TANMSFF(nn.Module):
    """
    Temporal Attention Network with Multi-Scale Feature Fusion
    
    Architecture:
    1. Mel-spectrogram frontend (computed externally)
    2. Multi-scale convolutional encoder
    3. Temporal self-attention layers
    4. Statistics pooling
    5. Classification head
    """
    def __init__(
        self,
        n_mels: int = 64,
        num_classes: int = 18,
        channels: list = [32, 64, 128],
        num_attention_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Initial projection from mel features
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_mels, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(channels[0]),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Multi-scale convolutional encoder
        self.conv_blocks = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels:
            self.conv_blocks.append(MultiScaleConvBlock(in_ch, out_ch, dropout))
            self.conv_blocks.append(nn.MaxPool1d(2))
            in_ch = out_ch
        
        # Temporal self-attention layers
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(channels[-1], num_heads, dropout)
            for _ in range(num_attention_layers)
        ])
        
        # Statistics pooling
        self.pool = StatisticsPooling()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1] * 2, channels[-1]),
            nn.BatchNorm1d(channels[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, return_features=False):
        """
        Args:
            x: Mel-spectrogram tensor of shape (B, n_mels, T)
            return_features: If True, also return intermediate features
        
        Returns:
            logits: Classification logits of shape (B, num_classes)
            features: (optional) Feature tensor before classifier
        """
        # Input projection
        x = self.input_proj(x)
        
        # Multi-scale convolutions
        for block in self.conv_blocks:
            x = block(x)
        
        # Self-attention
        for attn in self.attention_blocks:
            x = attn(x)
        
        # Statistics pooling
        features = self.pool(x)
        
        # Classification
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def get_embedding(self, x):
        """Get feature embedding for similarity search"""
        _, features = self.forward(x, return_features=True)
        return F.normalize(features, p=2, dim=1)


class LightweightTANMSFF(nn.Module):
    """
    Lightweight version for edge deployment
    ~100K parameters
    """
    def __init__(
        self,
        n_mels: int = 40,
        num_classes: int = 18,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Depthwise separable convolutions for efficiency
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(n_mels, 32, 3, padding=1, groups=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 2 - Depthwise separable
            nn.Conv1d(32, 32, 3, padding=1, groups=32),
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 3 - Depthwise separable
            nn.Conv1d(64, 64, 3, padding=1, groups=64),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Dropout(dropout)
        )
        
        # Simple attention
        self.attention = nn.Sequential(
            nn.Conv1d(128, 1, 1),
            nn.Softmax(dim=2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        
        # Attention-weighted pooling
        attn = self.attention(x)
        x = (x * attn).sum(dim=2)
        
        # Classification
        return self.classifier(x)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    batch_size = 8
    n_mels = 64
    time_steps = 100
    num_classes = 18
    
    x = torch.randn(batch_size, n_mels, time_steps)
    
    # Full model
    model = TANMSFF(n_mels=n_mels, num_classes=num_classes)
    out = model(x)
    print(f"TAN-MSFF Output shape: {out.shape}")
    print(f"TAN-MSFF Parameters: {count_parameters(model):,}")
    
    # Lightweight model
    model_light = LightweightTANMSFF(n_mels=40, num_classes=num_classes)
    x_light = torch.randn(batch_size, 40, time_steps)
    out_light = model_light(x_light)
    print(f"Lightweight Output shape: {out_light.shape}")
    print(f"Lightweight Parameters: {count_parameters(model_light):,}")
