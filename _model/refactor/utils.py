import io
import os
import torch
import logging
import json
import pickle
import argparse
from pprint import pprint

# 3rd party libraries
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils import data
from sklearn import decomposition
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
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
from AbstractBert import AbstractBert

def load_data(config, vocab, proportion: float=0.7, max_len: int=256, partition: dict=None, labels: dict=None):
    """Load data using PyTorch DataLoader.
    Keyword Arguments:
        config {string} -- config file containing data paths and tokenizer information
        metadata {bool} -- whether the data contains metadata for augmented embeddings
        proportion {float} -- proportion for splitting up train and test. (default: {0.7})
        max_len {int} -- maximum token length for a text. (default: {128})
        partition {dict} -- maps lists of training and validation data IDs (default: {None})
        labels {dict} -- (default: {None})
    Returns:
        partition {dict} -- list of ids in train and valid datasets
        torch.utils.data.Dataset -- dataset
    """

    # columns if meta: [0] unique ID, [1] text, [2] metadata, [3] label
    # columns if no meta: [0] unique ID, [1] text, [2] label

    if config["metadata"]:
        unique_id_col = 0
        text_col = 1
        metadata_col = 2
        label_col = 3
    else: 
        unique_id_col = 0
        text_col = 1
        label_col = 2

    dataset = pd.read_csv(config['train_file'], header=None, sep='\t')
    # below fix null values wrecking encode_plus

    # convert labels to integer and drop nas
    dataset.iloc[:, label_col] = pd.to_numeric(dataset.iloc[:, label_col], errors = 'coerce' )
    dataset = dataset[~ dataset[text_col].isnull()]

    # recreate the first column with the reset index.
    dataset = dataset[(dataset.iloc[:, label_col] == 1) | (dataset.iloc[:, label_col] == 0)] \
        .reset_index().reset_index().drop(columns = ['index', 0]).rename(columns = {'level_0': 0})

    # create list of train/valid IDs if not provided
    if not partition and not labels:
        ids = list(dataset.iloc[:,unique_id_col])
        total_len = len(ids)
        np.random.shuffle(ids)

        labels = {}
        # metadata = {}
    
        partition = {'train': ids[ :int(total_len * 0.7)],
                     'valid': ids[int(total_len * 0.7): ]
                     }
        for i in dataset.iloc[:, unique_id_col]:
            labels[i] = dataset.iloc[i][label_col]

    # set parameters for DataLoader -- num_workers = cores
    params = {'batch_size': 32,
              'shuffle': True,
              'num_workers': 0
              }

    tokenizer = AutoTokenizer.from_pretrained(vocab)
    dataset[text_col] = dataset[text_col].apply(lambda x: tokenizer.encode_plus(str(x), \
                                                                  max_length=max_len, \
                                                                  add_special_tokens=True, \
                                                                  pad_to_max_length=True, \
                                                                  truncation=True))

    if config['metadata']: # glove for metadata preprocessing 
        glove = torchtext.vocab.GloVe(name="6B", dim=50)
        dataset[metadata_col] = dataset[metadata_col].apply(lambda y: __pad__(str(y).split(" "), 30))
        dataset[metadata_col] = dataset[metadata_col].apply(lambda z: __glove_embed__(z, glove))

    train_data = dataset[dataset[unique_id_col].isin(partition['train'])]
    valid_data = dataset[dataset[unique_id_col].isin(partition['valid'])]

    # create train/valid generators
    training_set = AbstractDataset(data=train_data, labels=labels, metadata=config['metadata'], list_IDs=partition['train'])
    training_generator = DataLoader(training_set, **params)

    validation_set = AbstractDataset(data=valid_data, labels=labels, metadata=config['metadata'], list_IDs=partition['valid'])
    validation_generator = DataLoader(validation_set, **params)

    return partition, training_generator, validation_generator


def __pad__(sequence, max_l):
    """ Padding function for 1D sequences """
    if max_l - len(sequence) < 0:
        sequence = sequence[:max_l]
    else: 
        sequence = np.pad(sequence, (0, max_l - (len(sequence))), 'constant', constant_values=(0))
    return sequence


def __glove_embed__(sequence, model):
    """ Embed words in a sequence using GLoVE model """
    embedded = []
    for word in sequence:
        embedded.append(model[word])
    return embedded


def load_embeddings(config, name, vocab, training_generator, validation_generator):
    """Load embeddings either from cache or from scratch
    Args:
        config (json) -- file configurations.
        name --
        vocab -- 
        training_generator --
        validation_generator -- 
    Returns:
        embedding_shape, train_embeddings, valid_embeddings
    """    
    if os.listdir(config['cache']+"/"+name):
        with open(config['cache']+name+'_training_embeddings.p', 'rb') as cache:
            train_embeddings = pickle.load(cache)

        with open(config['cache']+name+'_validation_embeddings.p', 'rb') as cache:
            valid_embeddings = pickle.load(cache)
    else:
        # get embeddings from scratch
        tokenizer = AutoTokenizer.from_pretrained(vocab)
        embedding_model = AbstractBert(vocab) 

        if torch.cuda.device_count() > 1:
            print("GPUs Available: ", torch.cuda.device_count())
            embedding_model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
       
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        embedding_model.eval().to(device)

        logger.info(' Getting BERT embeddings...')
        train_embeddings = _get_bert_embeddings(training_generator, embedding_model, config["metadata"])
        valid_embeddings = _get_bert_embeddings(validation_generator, embedding_model, config["metadata"])

        # save embeddings
        pickle.dump(train_embeddings, open(config['cache']+name+'_training_embeddings.p', 'wb'))
        pickle.dump(valid_embeddings, open(config['cache']+name+'_validation_embeddings.p', 'wb'))

        logger.info(' Saved full BERT embeddings.')

    embedding_shape = train_embeddings['embeddings'][1].shape[0]

    return embedding_shape, train_embeddings, valid_embeddings


def _get_bert_embeddings(data_generator, embedding_model: torch.nn.Module, metadata: False):
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
        if metadata:
            for local_ids, local_data, local_meta, local_labels in data_generator:
                local_data, local_meta, local_labels =  local_data.to(device).long().squeeze(1), \
                                                        local_meta, \
                                                        local_labels.to(device).long()

                augmented_embeddings = embedding_model(local_data, local_meta)

                embeddings['ids'] = local_ids
                embeddings['embeddings'].extend(np.array(augmented_embeddings.detach().cpu()))
                embeddings['labels'].extend(np.array(local_labels.detach().cpu().tolist()))
        else:
            for local_ids, local_data, local_labels in data_generator:
                local_data, local_labels =  local_data.to(device).long().squeeze(1), \
                                                local_labels.to(device).long()

                augmented_embeddings = embedding_model(local_data)

                embeddings['ids'] = local_ids
                embeddings['embeddings'].extend(np.array(augmented_embeddings.detach().cpu()))
                embeddings['labels'].extend(np.array(local_labels.detach().cpu().tolist()))
    
    return embeddings

def get_pca_embeddings(config, name, training_embedding: dict, validation_embedding: dict):
    """Reduced embeddings using PCA. 
    Args:
        training_embedding (dict) -- dictionary containing training embeddings
        validation_embedding (dict) -- dictionary containing validation embeddings
    Returns:
        generator -- Torch Dataloader
        tuple -- shape of embedding
    """    
    params = {'batch_size': 32,
              'shuffle': False,
              'num_workers': 0
              }

    if os.listdir(config['pca_cache']):
        logger.info(" Loading PCA-embeddings from cache ")
        with open(config['pca_cache']+name+'_pca_train_embeddings.p', 'rb') as cache:
            train_embeddings = pickle.load(cache)

        with open(config['pca_cache']+name+'_pca_valid_embeddings.p', 'rb') as cache:
            valid_embeddings = pickle.load(cache)
    else:
        logger.info(' Standardizing ')
        
        ss = StandardScaler()
        train_embed_ss = ss.fit_transform(training_embedding['embeddings'])
        valid_embed_ss = ss.transform(validation_embedding['embeddings'])

        # Dimension reduction: PCA or UMAP (?)
        logger.info(' Doing PCA...')
        pca_model = decomposition.PCA(n_components = 0.99) # this can be a parameter down the road, but for debugging it's fine
        train_reduc = pca_model.fit_transform(train_embed_ss)
        val_reduc = pca_model.transform(valid_embed_ss)

        
        training_embedding['embeddings'] = train_reduc
        validation_embedding['embeddings'] = val_reduc

        train_embeddings = training_embedding.copy()
        valid_embeddings = validation_embedding.copy()

        # save embeddings
        pickle.dump(train_embeddings, open(config['cache']+name+"/"+name+'_pca_train_embeddings.p', 'wb'))
        pickle.dump(valid_embeddings, open(config['cache']+name+"/"+name+'_pca_valid_embeddings.p', 'wb'))

    embedding_shape = len(train_embeddings['embeddings'][0])

    return embedding_shape, train_embeddings, valid_embeddings


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
