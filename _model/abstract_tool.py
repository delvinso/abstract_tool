import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import *
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from AbstractDataset import AbstractDataset
from pprint import pprint

def load_data(csv_file: str, tokenizer, max_len: int=128, partition: dict=None, labels: dict=None):
    """Load data using PyTorch DataLoader.

    Keyword Arguments:
        csv_file {string} -- path to load raw data
        tokenizer {AutoModel.tokenizer} -- BERT-specific tokenizer for preprocessing
        max_len {int} -- maximum token length for a text. (default: {128})
        partition {dict} -- maps lists of training and validation data IDs (default: {None})
        labels {dict} -- (default: {None})

    Returns:
        torch.utils.data.Dataset -- dataset
    """
    np.random.seed(2020)
    
    # columns: [0] unique ID, [1] text, [2] label
    dataset = pd.read_csv(csv_file, header=None, sep='\t')

    # create list of train/valid IDs if not provided
    if not partition and not labels: 
        ids = list(dataset.iloc[:,0])
        total_len = len(ids)
        np.random.shuffle(ids)

        labels = {}
        partition = {'train': ids[int(total_len*0.7): ],
                     'valid': ids[ :int(total_len*0.7)]
                    }
        for i in dataset.iloc[:,0]:
            labels[i] = dataset.iloc[i][2]

    # set parameters for DataLoader -- num_workers = cores
    params = {'batch_size': 4,
              'shuffle': True,
              'num_workers': 4
             }

    # tokenize, add [CLS], [SEP], **pad if necessary (in data loading function)***
    dataset[1] = dataset[1].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

    # TODO: create attention mask (to indicate padding or no padding)
    # mask = [] 

    # for seq in dataset[1].tolist():
    #     seq_mask = [float(i>0) for i in seq]
    #     attention_masks.append(seq_mask)

    train_data = dataset[dataset[0].isin(partition['train'])]
    valid_data = dataset[dataset[0].isin(partition['valid'])]

    # create train/valid generators
    training_set = AbstractDataset(train_data, partition['train'], labels, max_len)
    training_generator = DataLoader(training_set, **params)


    validation_set = AbstractDataset(valid_data, partition['valid'], labels,  max_len)
    validation_generator = DataLoader(validation_set, **params)

    return training_generator, validation_generator

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
 
    ## TODO: json file for args 
    FILE = '/Users/chantal/Desktop/systematic_review/abstract_tool/data/cleaned_data/Scaling_data.tsv'
    MAX_EPOCHS = 50
    MAX_LEN = 128

    # BERT-specific variables, load model + tokenizer, using pretrained SciBERT
    pretrained_weights = 'allenai/scibert_scivocab_uncased'
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

    # load data
    training_generator, validation_generator = load_data(FILE, tokenizer, MAX_LEN)

    for epoch in range(MAX_EPOCHS):
        # Training
        for local_batch, local_labels in training_generator:
            # Transfer to GPU if necessary
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            

            # Model computations
            

        # Validation
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in validation_generator:
                # Transfer to GPU if necessary
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # Model computations
                


if __name__ == '__main__':
    main()



