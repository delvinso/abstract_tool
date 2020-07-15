import torch
import pandas as pd
import numpy as np
from torch.utils import data
from keras.preprocessing.sequence import pad_sequences

class AbstractDataset(data.Dataset):
  def __init__(self, data, labels, is_embedding: bool, list_IDs: list=None,  max_len: int=128):
    """Create custom torch Dataset.
    
    Arguments:
    data {array-like} --  DataFrame containing dataset.
    is_embedding {bool} -- whether embeddings or raw data sent in
    list_IDs {list} -- List of data IDs to be loaded.
    labels {dict} -- Map of data IDs to their labels.
    
    Returns:
    X,Y,z -- data, metadata, and label.
    """
    self.data = data
    self.max_len = max_len
    self.labels = labels
    self.list_IDs = list_IDs
    self.is_embedding = is_embedding

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    if self.is_embedding: 
      X = self.data
      # print(self.labels)
      # print(ID)
      y = self.labels
      
      return torch.tensor(X), torch.tensor(y)
    else: 
      # Load data and get metadata
      X = self.data[self.data[0] == ID][1].values
      Y = self.data[self.data[0] == ID][2].values.tolist()

      # X = pad_sequences(X, maxlen=self.max_len, dtype="long", truncating="post", padding="post")

      # Load label
      # z = np.reshape(self.labels[ID], (1,1))
      z = self.labels[ID]

      # X: data, Y: metadata, z: label
      return self.list_IDs, torch.tensor(X[0]['input_ids']), Y, torch.tensor(z) 
