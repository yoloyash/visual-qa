import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch


def get_optimizer_and_scheduler(model, optim='adam', lr=0.001):
    if optim.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=False)
        return optimizer, lr_scheduler
    else:
        raise NotImplementedError(f"{optim} not implemented")


class VQAModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VQAModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.dropout2(x)
        return x
    
class VQAModelV2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VQAModelV2, self).__init__()

        self.block1 = nn.Sequential(*[nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3)])

        self.block2 = nn.Sequential(*[nn.Linear(hidden_dim, 2*hidden_dim),
            nn.LayerNorm(2*hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3)])

        self.block3 = nn.Sequential(*[nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.LayerNorm(2*hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3)])

        self.classifier = nn.Linear(2*hidden_dim, output_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        out = self.classifier(x)
        return out

class VQAModelV3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VQAModelV3, self).__init__()

        self.block1 = nn.Sequential(*[nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5)])

        self.block2 = nn.Sequential(*[nn.Linear(hidden_dim, 2*hidden_dim),
            nn.LayerNorm(2*hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5)])

        self.classifier = nn.Linear(2*hidden_dim, output_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        out = self.classifier(x)
        return out
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.bias = bias
        self.attention = nn.Linear(feature_dim, 1, bias=bias)
        
    def forward(self, x):
        # x shape: (batch_size, step_dim, feature_dim)
        eij = self.attention(x)
        
        # Compute softmax over the step dimension (time dimension) to get attention weights
        a = F.softmax(eij, dim=1)
        weighted_input = x * a
        
        # Sum over the step dimension to get the attended feature vector
        attended_features = weighted_input.sum(1)
        return attended_features

class VQAModelV3Attn(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VQAModelV3Attn, self).__init__()

        self.block1 = nn.Sequential(*[nn.Linear(input_dim, hidden_dim),
                                      nn.LayerNorm(hidden_dim),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.5)])

        self.block2 = nn.Sequential(*[nn.Linear(hidden_dim, 2*hidden_dim),
                                      nn.LayerNorm(2*hidden_dim),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.5)])

        # Attention layer
        self.attention = Attention(feature_dim=2*hidden_dim, step_dim=1)  # Adjust step_dim based on your actual input shape

        self.classifier = nn.Linear(2*hidden_dim, output_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        
        # Apply attention
        x = self.attention(x.unsqueeze(1))  # Unsqueeze to add a dummy step dimension
        
        out = self.classifier(x)
        return out