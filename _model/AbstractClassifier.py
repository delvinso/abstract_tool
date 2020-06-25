import torch
from torch import nn
import numpy as np

class AbstractClassifier(nn.Module):

    def __init__(self, embedding_size: tuple):
        """Classifier model for embedded Abstracts.

        Args:
            embedding_size (tuple): Tuple indicating embedding input size
        """        
        super(AbstractClassifier, self).__init__()
        self.linear = nn.Linear(embedding_size, 1, bias=False)

    def forward(self, x):
        """ 
        Forward method of the Classification model.
        Args:
            x (torch.tensor): (batch of) embeddings to be classified

        Returns:
            torch.tensor: logits
        """        
        out = self.linear(x)
        return out
        