import torch
from torch import nn
import numpy as np

class AbstractClassifier(nn.Module):

    def __init__(self, type, embedding_size: tuple):
        """Classifier model for embedded Abstracts.
        Args:
            embedding_size (tuple): Tuple indicating embedding input size
        """        
        super(AbstractClassifier, self).__init__()

        # CNN TODO: shape checks
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 20, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(20, 50, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(50*44, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        """ 
        Forward method of the Classification model.
        Args:
            x (torch.tensor): (batch of) embeddings to be classified
        Returns:
            torch.tensor: logits
        """     
        out = self.layer1(x)
        out = self.layer2(out)

        out = out.reshape(out.size(0), -1)
        
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
        
        return out

        