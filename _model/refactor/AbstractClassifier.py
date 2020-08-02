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

        # linear 
        self.linear = nn.Linear(embedding_size, 1, bias=False)
        self.type = type

        # CNN TODO: shape checks
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        """ 
        Forward method of the Classification model.
        Args:
            x (torch.tensor): (batch of) embeddings to be classified
        Returns:
            torch.tensor: logits
        """     
        if self.type == "cnn":
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.drop_out(out)
            out = self.fc1(out)
            out = self.fc2(out)
        return out
        elif self.type == "fcn": 
            out = self.linear(x)
            return out
        else: 
            raise NotImplementedError
        