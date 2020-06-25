import io
import torch
import logging
import json
import argparse
import pickle
from pprint import pprint

# 3rd party libraries
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import cluster, datasets, mixture, decomposition
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier
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
from data_tools import load_data, metrics, get_embeddings

#TODO: clean this up.

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

    # load config file 
    with open(args.config_dir) as f:
        config = json.load(f)

    # TODO: We really dont need these variables lol, just use the config file directly 
    # default params 
    TRAIN_FILE = config['train_file']
    TEST_FILE = config['test_file']
    TYPE = TRAIN_FILE.split('/')[-1][:-4] # retrieve file name without the extension
    MAX_EPOCHS = config['epochs']
    MAX_LEN = config['max_len']
    SEED = config['seed']
    PRETRAINED_KNN = config['clustering_model']
    TRAIN_MODE = config['train']
    TEST_MODE = config['test']
    CACHED_EMBEDDINGS = config['embeddings']

    # model specific params
    PRETRAINED_WEIGHTS = 'allenai/scibert_scivocab_uncased'
    N_CLUSTERS = 5
    COV_TYPE = 'full'

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)

    # config checks
    if not TRAIN_MODE and not PRETRAINED_KNN:
        logger.error('Must provide path to a trained clustering model if training mode is False.')
    
    if TEST_MODE and not TEST_FILE:
        logger.error('Must provide path to a testing file with data if testing mode is True.')
    
    if CACHED_EMBEDDINGS: # check for cached embeddings
        # load train, valid
        reduced_train = CACHED_EMBEDDINGS['train']
        reduced_valid = CACHED_EMBEDDINGS['valid']

        # load labels
        label_train = CACHED_EMBEDDINGS['train_label']
        label_valid = CACHED_EMBEDDINGS['valid_label']

        if TEST_MODE: # if testing, also load test labels
            reduced_test = CACHED_EMBEDDINGS['test']
            label_test = CACHED_EMBEDDINGS['test_label']

    else: # obtain embeddings from scratch 
        tokenizer = BertTokenizer.from_pretrained(PRETRAINED_WEIGHTS)
        embedding_model = BertForAbstractScreening(PRETRAINED_WEIGHTS) 

        if torch.cuda.device_count() > 1:
            print("GPUs Available: ", torch.cuda.device_count())
            embedding_model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])

        embedding_model.eval().to(device)

        # load data
        training_generator, validation_generator = load_data(TRAIN_FILE, tokenizer, MAX_LEN)

        logger.info(f'\nNum. training samples: {len(training_generator)} \
                    \nNum. validation samples: {len(validation_generator)}')
    
        ######## BERT embeddings and PCA ########
        logger.info(' Getting BERT embeddings...')
        augmented_train = get_embeddings(training_generator, embedding_model)
        augmented_valid = get_embeddings(validation_generator, embedding_model)

        if TEST_MODE: # if testing, load the testing data as well
            test_generator, _ = load_data(TEST_FILE, tokenizer, MAX_LEN, proportion=0)
            logger.info(f'\nNum. testing samples: {len(test_generator)}')

            augmented_test = get_embeddings(test_generator, embedding_model)
            
            all_augmented = np.vstack((augmented_train['embeddings'], \
                                    augmented_valid['embeddings'], \
                                    augmented_test['embeddings']))
        else:
            all_augmented = np.vstack((augmented_train['embeddings'], augmented_valid['embeddings']))

        # Dimension reduction: PCA
        logger.info(' Doing PCA...')
        pca_model = decomposition.PCA(n_components='mle')
        all_reduced = pca_model.fit_transform(all_augmented)

        reduced_train = all_reduced[ :len(augmented_train['embeddings'])]
        reduced_train_label = augmented_train['labels']

        if TEST_MODE: 
            reduced_valid = all_reduced[len(augmented_train['embeddings']): -len(augmented_test['embeddings'])]
            reduced_valid_label = augmented_valid['labels']

            reduced_test = all_reduced[-len(augmented_test['embeddings']): ]
            reduced_test_label = augmented_test['labels']
        else:
            reduced_valid = all_reduced[len(augmented_train['embeddings']): ]
            reduced_valid_label = augmented_valid['labels']

        # save embeddings 
        embeddings = {'train' = ,
                      'valid' = ,
                      'test'  = ,
                      }
        ######## BERT embeddings and PCA ########

    ######## KNN ########
    if TRAIN_MODE: 
        # train new KNN
        logger.info(' Clustering...')
        clustering_model = KNeighborsClassifier(n_neighbors=N_CLUSTERS, \
                                                weights='distance')
        # fit training embeddings
        cv_scores = cross_val_score(clustering_model, \
                                    reduced_train, \
                                    augmented_train['labels'], \
                                    cv=5)
        clustering_model.fit(reduced_train, augmented_train['labels'])
        logger.info(f" Cross-validation score for clustering model: {np.mean(cv_scores)}")
    else: 
        # load pretrained KNN (pickled) 
        clustering_model = pickle.load( )
    ######## KNN ########


    # predict the posterior probability of the validation embeddings, check label.. 
    all_predictions = [clustering_model.predict_proba(reduced_valid), \
                       clustering_model.predict(reduced_valid), \
                       augmented_valid['labels']]

    # check prediction performance on the validation data 
    logger.info(' Getting metrics...')
    roc_auc = metrics('roc_auc', preds = all_predictions[1], labels = all_predictions[2])
    ap = metrics('ap', preds = all_predictions[1], labels = all_predictions[2])
    
    logger.info(f'AUROC: {np.mean(roc_auc)} \nAP: {np.mean(ap)}')

    # save all models
    pickle.dump(clustering_model, open('model', 'w+'))
    embedding_model.module.save_state_dict('embedding_BFAS_'+TYPE+'.pt')

    # testing data, one by one, evaluate confidence in prediction
    # present non_confident_set back to user for labeling 

    # # TODO: confidence score and threshold score following this paper: https://bit.ly/2Zg3mVL 
    # if TEST_MODE: 
    #     non_confident_set = {}

    #     for abstract_id, abstract in zip(reduced_test, augmented_test['embeddings']):
    #         predicted_label = clustering_model.predict(abstract)
    #         confidence = get_knn_confidence(abstract)

    #         if confidence < threshold: 
    #             non_confident_set.append([abstract_id, abstract, predicted_label])
        
    #     with open('to_be_labeled.txt', 'w+') as f:
    #         f.write(non_confident_set)


if __name__ == '__main__':
    main()



