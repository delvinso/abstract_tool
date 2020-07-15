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
from AbstractClassifier import AbstractClassifier
from AbstractBert import AbstractBert
from utils import load_data, metrics, load_embeddings, get_bert_embeddings, get_pca_embeddings


""" TODO: TESTING MODULE. """

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

    # set seeds
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    # torch.cuda.manual_seed_all(SEED)

    # load data     
    partition, training_generator, validation_generator = load_data(config, metadata=False)

    # get embeddings  
    train_embeddings, valid_embeddings = load_embeddings(config, training_generator, validation_generator)

    # do PCA (TODO: or augmented PCA)
    embed_shape, reduced_train_generator, reduced_valid_generator = get_pca_embeddings(partition, \
                                                                                       train_embeddings, \
                                                                                       valid_embeddings)

    ############################### Train Classifier ###############################
    classifier = AbstractClassifier(embedding_size=embed_shape[0])
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(config['epochs']):

        losses = 0.0

        for i, data in enumerate(reduced_train_generator, 0):
            print(data.shape)
            inputs, labels = data
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses += loss.item()

            if i % 2000 == 1999: 
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
    ###############################################################################

    logger.info("Classifier Training Complete")
            
    # save model
    torch.save(classifier.state_dict(), config["pretrained_classifier"])


    # check prediction performance on the validation data 
    correct = 0
    total = 0
    with torch.no_grad():
        for data in reduced_valid_generator:
            embeddings, labels = data
            outputs = net(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        logger.info(' Getting metrics...')
        roc_auc = metrics('roc_auc', preds = all_predictions[1], labels = all_predictions[2])
        ap = metrics('ap', preds = all_predictions[1], labels = all_predictions[2])
        
        logger.info(f'AUROC: {np.mean(roc_auc)} \nAP: {np.mean(ap)}')



if __name__ == '__main__':
    main()



