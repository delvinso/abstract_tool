import io
import torch
import logging
import json
import argparse
from pprint import pprint

# 3rd party libraries
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
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

# custom
from AbstractDataset import AbstractDataset
from BertForAbstractScreening import BertForAbstractScreening

def load_data(csv_file, tokenizer, max_len: int=128, partition: dict=None, labels: dict=None):
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

    # columns: [0] unique ID, [1] text, [2] label, [3] metadata 
    dataset = pd.read_csv(csv_file, header=None, sep='\t')

    # create list of train/valid IDs if not provided
    if not partition and not labels:
        ids = list(dataset.iloc[:,0])
        total_len = len(ids)
        np.random.shuffle(ids)

        labels = {}
        # metadata = {}
    
        partition = {'train': ids[ :int(total_len*0.7)],
                     'valid': ids[int(total_len*0.7): ]
                     }
        for i in dataset.iloc[:,0]:
            labels[i] = dataset.iloc[i][2]
            # metadata[i] = dataset.iloc[i][3]

    # set parameters for DataLoader -- num_workers = cores
    params = {'batch_size': 32,
              'shuffle': True,
              'num_workers': 16
              }

    # NOTE: the tokenizer.encocde_plus function does the token/special/map/padding/attention all in one go
    dataset[1] = dataset[1].apply(lambda x: tokenizer.encode_plus(x, add_special_tokens=True))

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
    labels = np.concatenate(labels)
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


def main():
    # logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG) # if you want to pipe the logging to a file
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', help='path to the json config file')
    args = parser.parse_args()

    # model configuration file 
    with open(args.config_dir) as f:
        config = json.load(f)

    FILE = config['file']
    TYPE = FILE.split('/')[-1][:-4] # retrieve file name without the extension
    MAX_EPOCHS = config['epochs']
    MAX_LEN = config['max_len']
    SEED = config['seed']

    # setting up seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)

    # load model + tokenizer, using pretrained SciBERT
    pretrained_weights = 'allenai/scibert_scivocab_uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    # model = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased')
    model = BertForAbstractScreening.from_pretrained(pretrained_weights) 

    if torch.cuda.device_count() > 1:
        print("GPUs Available: ", torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])

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
        labels_list = []
        preds_list = []
        total_train_roc_auc = 0

        model.train().to(device)

        for idx, local_batch, local_labels in enumerate(training_generator):
            # NOTE: remove .cpu() when running on GPU, adding in "channel" dim at pos -1
            local_batch, local_labels = local_batch.to(device).long().squeeze(1), \
                                        local_labels.to(device).long()

            model.zero_grad()

            output = model(local_batch)
            logits = output[0]

            loss_fct = CrossEntropyLoss()
            # print(logits.view(-1, 2))
            loss = loss_fct(logits.view(-1, 2), local_labels.view(-1))

            total_train_loss += loss.item()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            logits = torch.softmax(logits, dim = 1)
            logits = logits.detach().cpu().numpy()

            for i in range(logits.shape[0]):
                preds = logits[i][1] # get p(y == 1) from each sample
                preds_list.append(preds)

            # print(local_labels.detach().cpu().numpy())
            label_ids = local_labels.detach().cpu().numpy().flatten() #[0]
            labels_list.append(label_ids)

            if ((idx % 10 == 0) & (idx > 0)):
                running_roc_auc = metrics('roc_auc', preds = preds_list, labels = labels_list)
                running_ap = metrics('ap', preds = preds_list, labels = labels_list)
                logging.info('Training Batch: {}, AUC: {:3f}, AP {:3f}'.format(i, running_roc_auc, running_ap))
                # print(label_ids)
                # print(logits)
            

        avg_train_loss = total_train_loss / len(training_generator)
        total_train_roc_auc = metrics('roc_auc', preds = preds_list, labels = labels_list)
        total_train_ap = metrics('ap', preds = preds_list, labels = labels_list)
        logger.info(f"  Average training loss: {avg_train_loss:.3f}, Training AUC: {total_train_roc_auc:.3f},\
                        AP: {total_train_ap:.3f}")

        ########################## Validation ##########################
        model.eval().to(device)

        # various accuracy measures
        total_eval_accuracy = 0
        total_eval_f1 = 0
        total_eval_roc_auc = 0
        total_eval_precision = 0
        total_eval_recall = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        labels_list = []
        preds_list = []

        with torch.set_grad_enabled(False):
            logger.info(' Validating')

            for idx, local_batch, local_labels in enumerate(validation_generator):
                local_batch, local_labels = local_batch.to(device).long().squeeze(1), \
                                            local_labels.to(device).long()

                output = model(local_batch)
                logits = output[0]

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, 2), local_labels.view(-1))

                total_eval_loss += loss.item()

                logits = torch.softmax(logits, dim = 1)
                logits = logits.detach().cpu().numpy()

                label_ids = local_labels.detach().cpu().numpy().flatten()

                for i in range(logits.shape[0]):
                    preds = logits[i][1] # get p(y == 1) from each sample
                    preds_list.append(preds)
                labels_list.append(label_ids)


                # total_eval_f1 += metrics('f1', logits, label_ids)
                if ((idx % 10 == 0) & (idx > 0)):
                    # print(labels_list)
                    # print(preds_list)
                    running_roc_auc = metrics('roc_auc', preds = preds_list, labels = labels_list)
                    running_ap = metrics('ap', preds = preds_list, labels = labels_list)
                    logging.info('Validation Batch: {}, AUC: {:3f}, AP {:3f}'.format(i, running_roc_auc, running_ap))
               

        total_eval_roc_auc = metrics('roc_auc', preds = preds_list, labels = labels_list)
        avg_roc_auc = total_eval_roc_auc / len(validation_generator)


        avg_val_loss = total_eval_loss / len(validation_generator)
        total_roc_auc = metrics('roc_auc', preds = preds_list, labels = labels_list)
        total_ap = metrics('ap', preds = preds_list, labels = labels_list)
        logger.info(f"  Average validation loss: {avg_val_loss:.3f}, \
                        Validation AUC: {total_roc_auc:.3f}, \
                        AP: {total_ap:.3f}")
        
        # save the model 
        model.module.save_state_dict('scibert_'+TYPE+'.pt')


if __name__ == '__main__':
    main()



