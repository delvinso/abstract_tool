import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import *
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from AbstractDataset import AbstractDataset
from pprint import pprint
import logging

''' TODO: 
    - add other models for comparison, set up different structure on top of BERT (e.g., use AutoModel+torch.nn instead of BERTForSequenceClassification)
    - remove hardcoded params
    - save/load trained model function
    - plot loss with tensorboard
    - testing function
    - Lines 55, 64, 106, 145, 150
'''

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
    
    # columns: [0] unique ID, [1] text, [2] label
    dataset = pd.read_csv(csv_file, header=None, sep='\t')

    # create list of train/valid IDs if not provided
    if not partition and not labels: 
        ids = list(dataset.iloc[:,0])
        total_len = len(ids)
        np.random.shuffle(ids)

        labels = {}
        partition = {'train': ids[ :int(total_len*0.7)],
                     'valid': ids[int(total_len*0.7): ]
                    }
        for i in dataset.iloc[:,0]:
            labels[i] = dataset.iloc[i][2]

    # set parameters for DataLoader -- num_workers = cores
    params = {'batch_size': 4,
              'shuffle': True,
              'num_workers': 4
             }

    ### NOTE: the tokenizer.encocde_plus function does the token/special/map/padding/attention all in one go

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

def flat_accuracy(preds: list, labels: list):
    """ Accuracy between predictions and labels.
    
    Arguments:
        preds {list} -- predictions.
        labels {list} -- labels.
    
    Returns:
        int -- prediction accuracy
    """    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def main():
    # setting up logger
    # logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
 
    ## TODO: json file for args 
    FILE = '/Users/chantal/Desktop/systematic_review/abstract_tool/data/cleaned_data/Scaling_data.tsv'
    MAX_EPOCHS = 10
    MAX_LEN = 128
    SEED = 2020

    # setting up seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)

    # BERT-specific variables, load model + tokenizer, using pretrained SciBERT
    pretrained_weights = 'allenai/scibert_scivocab_uncased'
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased')
    model.to(device)

    # load data
    training_generator, validation_generator = load_data(FILE, tokenizer, MAX_LEN)

    logger.info(f'\nNum. training samples: {len(training_generator)} \
                  \nNum. validation samples: {len(validation_generator)}')

    params = list(model.named_parameters())

    optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
    # total steps: [number of batches] x [number of epochs]
    total_steps = len(training_generator) * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, 
                                                num_training_steps = total_steps)

    for epoch in range(MAX_EPOCHS):
        logger.info(f'======== Epoch {epoch} / {MAX_EPOCHS} ========')

        ########################## Training ##########################
        logger.info(' Training')
        total_train_loss = 0

        model.train()
        for local_batch, local_labels in tqdm(training_generator):
            # TODO: remove .cpu() when running on GPU, adding in "channel" dim at pos -1
            local_batch, local_labels = local_batch.to(device).long().cpu().squeeze(1), \
                                        local_labels.to(device).long().cpu()

            model.zero_grad()
            # TODO: attention mask None for now

            output = model(local_batch)
            logits = output[0]
            
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), local_labels.view(-1)) 


            total_train_loss += loss.item()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_generator)
        logger.info(f"  Average training loss: {avg_train_loss:.2f}") 

        ########################## Validation ##########################
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        with torch.set_grad_enabled(False):
            logger.info(' Validating')
            for local_batch, local_labels in tqdm(validation_generator):
                local_batch, local_labels = local_batch.to(device).long().cpu(), local_labels.to(device).long().cpu()

                output = model(local_batch)  

                logits = output[0]

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, 2), local_labels.view(-1)) 

                total_eval_loss += loss.item()  
                       
                logits = logits.detach().cpu().numpy()
                label_ids = local_labels.to('cpu').numpy()     

                total_eval_accuracy += flat_accuracy(logits, label_ids)   

        avg_val_accuracy = total_eval_accuracy / len(validation_generator)
        logger.info(f"  Accuracy: {avg_val_accuracy:.2f}")

        avg_val_loss = total_eval_loss / len(validation_generator)
        logger.info(f"  Validation Loss: {avg_val_loss:.2f}")
                


if __name__ == '__main__':
    main()



