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
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
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
from data_tools import load_data, metrics

'''
TODO: ADD PCA or some dimension reduction to speed up clustering
TODO: Add function to do UMAP/embedding visualization  
'''

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

    # default params 
    FILE = config['file']
    TYPE = FILE.split('/')[-1][:-4] # retrieve file name without the extension
    MAX_EPOCHS = config['epochs']
    MAX_LEN = config['max_len']
    SEED = config['seed']
    PRETRAINED_GMM = config['clustering_model']
    TRAIN_MODE = config['train']
    # model specific params
    PRETRAINED_WEIGHTS = 'allenai/scibert_scivocab_uncased'
    N_CLUSTERS = 2
    COV_TYPE = 'full'

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)

    if not TRAIN_MODE and not PRETRAINED_GMM:
        logger.error('Must provide path to a trained clustering model if training mode is False.')
    
    # load models and tokenizer
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_WEIGHTS)
    embedding_model = BertForAbstractScreening.from_pretrained(PRETRAINED_WEIGHTS) 

    if torch.cuda.device_count() > 1:
        print("GPUs Available: ", torch.cuda.device_count())
        embedding_model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])

    embedding_model.eval().to(device)
    # load data
    training_generator, validation_generator = load_data(FILE, tokenizer, MAX_LEN)

    logger.info(f'\nNum. training samples: {len(training_generator)} \
                  \nNum. validation samples: {len(validation_generator)}')

    train_augmented_embeddings = {}
    # valid_augmented_embeddings = {}

    if TRAIN_MODE: 
        # train new GMM
        clustering_model = mixture.GaussianMixture(n_components=N_CLUSTERS, \
                                                   covariance_type=COV_TYPE)
        with torch.set_grad_enabled(False):
            logger.info(' Getting BERT embeddings')

            # get BERT training embeddings
            for idx, local_batch, local_augmented, local_labels in enumerate(training_generator):
                local_batch, local_augmented, local_labels = local_batch.to(device).long().squeeze(1), \
                                                            local_augmented.to(device).long().squeeze(1), \
                                                            local_labels.to(device).long()

                augmented_embeddings = embedding_model(local_batch, local_augmented)
                # store batch with their labels 
                train_augmented_embeddings[idx] = [augmented_embeddings, local_labels]
    
            # fit training embeddings 
            clustering_model.fit(train_augmented_embeddings.values()[:, 0], \   
                                train_augmented_embeddings.values()[:, 1] )    # labels
            # save model 
            pickle.dump(clustering_model)
    else: 
        # load pretrained GMM (pickled) 
        clustering_model = pickle.load(PRETRAINED_GMM)

    all_predictions = {}
    # predict the posterior probability of the validation embeddings, check label.. 
    for idx, local_batch, local_augmented, local_labels in enumerate(validation_generator):
        local_batch, local_augmented, local_labels = local_batch.to(device).long().squeeze(1), \
                                                        local_augmented.to(device).long().squeeze(1), \
                                                        local_labels.to(device).long()

        augmented_embeddings = embedding_model(local_batch, local_augmented)

        all_predictions[key] = [clustering_model.predict_proba(augmented_embeddings), \
                                clustering_model.predict(augmented_embeddings), \
                                local_labels]
    
    # check prediction performance on the validation data 

    # save all models? 

    # while loop to constantly update and predict (phase 2)
        

if __name__ == '__main__':
    main()



