import torch
import pandas as pd
import numpy as np
from torch.utils import data
from keras.preprocessing.sequence import pad_sequences

class AbstractDataset(data.Dataset):
  def __init__(self, data, list_IDs: list, labels: dict, metadata: False, max_len: int=128):
    """Create custom torch Dataset.
  
    Arguments:
    data {array-like} --  DataFrame containing dataset.
    list_IDs {list} -- List of data IDs to be loaded.
    labels {dict} -- Map of data IDs to their labels.
    
    Returns:
    X,Y,z -- data, metadata, and label.
    """
    self.data = data
    self.max_len = max_len
    self.labels = labels
    self.list_IDs = list_IDs
    self.metadata = metadata

  def __len__(self):
      return len(self.list_IDs)

  def __getitem__(self, index):
      # Select sample
      ID = self.list_IDs[index]

      # Load data
      X = self.data[self.data[0] == ID][1].values

      if self.metadata:
        Y = self.data[self.data[0] == ID][2].values.tolist()
        z = self.labels[ID]

        return self.list_IDs, torch.tensor(X[0]['input_ids']), Y, torch.tensor(z) 

      else: 
        y = self.labels[ID]

        return self.list_IDs, torch.tensor(X[0]['input_ids']), torch.tensor(y) 

      