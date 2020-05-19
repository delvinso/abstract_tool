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

'''
TODO: Add function to do UMAP/embedding visualization  
TODO: incorporate phase 2
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

    # load config file 
    with open(args.config_dir) as f:
        config = json.load(f)

    # default params 
    TRAIN_FILE = config['train_file']
    TEST_FILE = config['test_file']
    TYPE = FILE.split('/')[-1][:-4] # retrieve file name without the extension
    MAX_EPOCHS = config['epochs']
    MAX_LEN = config['max_len']
    SEED = config['seed']
    PRETRAINED_KNN = config['clustering_model']
    TRAIN_MODE = config['train']
    TEST_MODE = config['test']

    # model specific params
    PRETRAINED_WEIGHTS = 'allenai/scibert_scivocab_uncased'
    N_CLUSTERS = 2
    COV_TYPE = 'full'

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)

    # config checks
    if not TRAIN_MODE and not PRETRAINED_KNN:
        logger.error('Must provide path to a trained clustering model if training mode is False.')
    
    if TEST_MODE and not TEST_FILE:
        logger.error('Must provide path to a testing file with data if testing mode is True.')
    
    # load models and tokenizer
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_WEIGHTS)
    embedding_model = BertForAbstractScreening.from_pretrained(PRETRAINED_WEIGHTS) 

    if torch.cuda.device_count() > 1:
        print("GPUs Available: ", torch.cuda.device_count())
        embedding_model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])

    embedding_model.eval().to(device)

    # load data
    training_generator, validation_generator = load_data(TRAIN_FILE, tokenizer, MAX_LEN)

    logger.info(f'\nNum. training samples: {len(training_generator)} \
                  \nNum. validation samples: {len(validation_generator)}')
    
    if TEST_MODE:
        test_generator, _ = load_data(TEST_FILE, proportion=0, tokenizer, MAX_LEN)
        logger.info(f'\nNum. testing samples: {len(test_generator)}')

        augmented_test = get_embeddings(test_generator, embedding_model)

        # get BERT embeddings 
        logger.info(' Getting augmented BERT embeddings...')
        augmented_train = get_embeddings(training_generator, embedding_model)
        augmented_valid = get_embeddings(validation_generator, embedding_model)
        augmented_test = get_embeddings(test_generator, embedding_model)
        

        all_augmented = np.hstack((augmented_train, augmented_valid, augmented_test))

        # Dimension reduction: PCA or UMAP (?)
        pca_model = decomposition.PCA(n_components='mle')
        all_reduced = pca.fit_transform(all_augmented)

        print(all_reduced.shape)

        reduced_train = all_reduced[ :len(augmented_train['embeddings'])]
        reduced_valid = all_reduced[len(augmented_valid['embeddings']): -len(augmented_test['embeddings'])]
        reduced_test = all_reduced[-len(augmented_test['embeddings']): ]
    else:
        # get BERT embeddings 
        logger.info(' Getting augmented BERT embeddings...')
        augmented_train = get_embeddings(training_generator, embedding_model)
        augmented_valid = get_embeddings(validation_generator, embedding_model)

        all_augmented = np.hstack((augmented_train, augmented_valid))

        # Dimension reduction: PCA or UMAP (?)
        pca_model = decomposition.PCA(n_components='mle')
        all_reduced = pca.fit_transform(all_augmented)

        print(all_reduced.shape)

        reduced_train = all_reduced[ :len(augmented_train['embeddings'])]
        reduced_valid = all_reduced[len(augmented_valid['embeddings']): ]

    if TRAIN_MODE: 
        # train new KNN
        clustering_model = KNeighborsClassifier(n_neighbors=N_CLUSTERS, \
                                                weights='distance')
        # fit training embeddings 
        clustering_model.fit(reduced_train, \   
                             augmented_train['labels'])    # labels
    else: 
        # load pretrained KNN (pickled) 
        clustering_model = pickle.load(PRETRAINED_KNN)

    # predict the posterior probability of the validation embeddings, check label.. 
    all_predictions = [clustering_model.predict_proba(reduced_valid), \
                       clustering_model.predict(reduced_valid), \
                       augmented_valid['labels']]
    
    # check prediction performance on the validation data 
    roc_auc = metrics('roc_auc', preds = all_predictions[1], labels = all_predictions[2])
    ap = metrics('ap', preds = all_predictions[1], labels = all_predictions[2])
    
    logger.info(f'AUROC: {np.avg(roc_auc)} \nAP: {np.avg(ap)}')

    # save all models
    pickle.dump(clustering_model)
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



