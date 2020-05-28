import io
import torch
import logging
import json
import argparse
from pprint import pprint

# 3rd party libraries
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils import data
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, auc, average_precision_score, precision_score, recall_score
from transformers import *
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torchtext 

# custom
from AbstractDataset import AbstractDataset
from BertForAbstractScreening import BertForAbstractScreening

def load_data(csv_file, tokenizer, proportion: float=0.7, max_len: int=128, partition: dict=None, labels: dict=None):
    """Load data using PyTorch DataLoader.

    Keyword Arguments:
        csv_file {string} -- path to load raw data
        tokenizer {AutoModel.tokenizer} -- BERT-specific tokenizer for preprocessing
        proportion {float} -- proportion for splitting up train and test. (default: {0.7})
        max_len {int} -- maximum token length for a text. (default: {128})
        partition {dict} -- maps lists of training and validation data IDs (default: {None})
        labels {dict} -- (default: {None})

    Returns:
        torch.utils.data.Dataset -- dataset
    """

    # columns: [0] unique ID, [1] text, [2] label, [3] metadata 
    dataset = pd.read_csv(csv_file, header=None, sep='\t')

    # create list of train/valid IDs if not provided
    if not partition and not labels:
        ids = list(dataset.iloc[:,0])
        total_len = len(ids)
        np.random.shuffle(ids)

        labels = {}
        # metadata = {}
    
        partition = {'train': ids[ :int(total_len * 0.7)],
                     'valid': ids[int(total_len * 0.7): ]
                     }
        for i in dataset.iloc[:,0]:
            labels[i] = dataset.iloc[i][3]

    # set parameters for DataLoader -- num_workers = cores
    params = {'batch_size': 2,
              'shuffle': True,
              'num_workers': 0
              }
    # glove for metadata preprocessing 
    glove = torchtext.vocab.GloVe(name="6B", dim=50)  

    # NOTE: the tokenizer.encocde_plus function does the token/special/map/padding/attention all in one go
    dataset[1] = dataset[1].apply(lambda x: tokenizer.encode_plus(x, \
                                                                  max_length=256, \
                                                                  add_special_tokens=True, \
                                                                  pad_to_max_length=True))
    #  print(dataset[2])
    dataset[2] = dataset[2].apply(lambda y: __pad__(str(y).split(" "), 30))
    dataset[2] = dataset[2].apply(lambda z: __glove_embed__(z, glove))

    train_data = dataset[dataset[0].isin(partition['train'])]
    valid_data = dataset[dataset[0].isin(partition['valid'])]

    # create train/valid generators
    training_set = AbstractDataset(train_data, partition['train'], labels, max_len)
    training_generator = DataLoader(training_set, **params)

    validation_set = AbstractDataset(valid_data, partition['valid'], labels,  max_len)
    validation_generator = DataLoader(validation_set, **params)

    return training_generator, validation_generator


def __pad__(sequence, max_l):
        if max_l - len(sequence) < 0:
            sequence = sequence[:max_l]
        else: 
            sequence = np.pad(sequence, (0, max_l - (len(sequence))), 'constant', constant_values=(0))
        return sequence


def __glove_embed__(sequence, model):
    embedded = []
    for word in sequence:
        embedded.append(model[word])
    return embedded


def get_embeddings(data_generator, embedding_model: torch.nn.Module):
    """Get BERT embeddings from a dataloader generator.

    Arguments:
        data_generator {data.Dataset} -- dataloader generator (AbstractDataset).
        embedding_model {torch.nn.Module} -- embedding model. 

    Returns:
        embeddings {dict} -- dictionary containing ids, augmented embeddings, and labels. 
    """    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    with torch.set_grad_enabled(False):
        embeddings = {'ids': [],
                      'embeddings': [],
                      'labels': []
                     }
        
        # get BERT training embeddings
        for local_ids, local_data, local_meta, local_labels in data_generator:
            local_data, local_meta, local_labels =  local_data.to(device).long().squeeze(1), \
                                                    local_meta, \
                                                    local_labels.to(device).long()

            augmented_embeddings = embedding_model(local_data, local_meta)

            embeddings['ids'].extend(local_ids)
            embeddings['embeddings'].extend(np.array(augmented_embeddings))
            embeddings['labels'].extend(np.array(local_labels.tolist()))

    return embeddings


def metrics(metric_type: str, preds: list, labels: list):
    """ Provides various metrics between predictions and labels.

    Arguments:
        metric_type {str} -- type of metric to use ['flat_accuracy', 'f1', 'roc_auc', 'precision', 'recall']
        preds {list} -- predictions.
        labels {list} -- labels.

    Returns:
        int -- prediction accuracy
    """
    assert metric_type in ['flat_accuracy', 'f1', 'roc_auc', 'ap'], 'Metrics must be one of the following: \
                                                                    [\'flat_accuracy\', \'f1\', \'roc_auc\'] \
                                                                    \'precision\', \'recall\', \'ap\']'
    labels = np.array(labels)
    # preds = np.concatenate(np.asarray(preds))

    if metric_type == 'flat_accuracy':
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)
    elif metric_type == 'f1':
        return f1_score(labels, preds)
    elif metric_type == 'roc_auc':
        return roc_auc_score(labels, preds)
    elif metric_type == 'precision':
        return precision_score(labels, preds)
    elif metric_type == 'recall':
        return recall_score(labels, preds)
    elif metric_type == 'ap':
        return average_precision_score(labels, preds)
