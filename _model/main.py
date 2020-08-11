import io
import torch
import logging
import json
import argparse
import pickle
import os
import time

# 3rd party libraries
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from sklearn import  decomposition
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer
import pandas as pd
import numpy as np


# custom
from AbstractDataset import AbstractDataset
from AbstractClassifier import AbstractClassifier
from AbstractBert import AbstractBert

from utils import load_data, metrics, load_embeddings, get_pca_embeddings
from run_model import train

# for dealing with multiprocessing/len(ancdata) error
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048*2, rlimit[1]))

'''
TODO: Add augmented PCA
TODO: ADD TEST FUNCTIONALITY 
TODO: 1D CNN arch. -- test full model
TODO: CNN --> set up hyperparams as config options
TODO: clean up train function
TODO: add option to load pretrained classifier
TODO: add FCN
TODO: Look into factory class pattern for abstract classifier (???)
'''

def main(args):

    # load config file 
    with open(args.config_dir) as f:
        config = json.load(f)

    # set seeds
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # set up cache/output directories and logger
    _set_dirs(config, args.name)
    logger = _set_logger(config, args.name)

    # get embeddings 
    embed_shape, train_embeddings, valid_embeddings = _get_data(config, logger, args.name)

    if config['train']:
        train(config, args.name, logger, train_embeddings, valid_embeddings, embed_shape)
    elif config['test']:
        # _test(config, args.name, logger)
        pass
    else:
        raise logger.error(f"Neither train nor test mode activated.")


def _set_logger(config, name):
    """ set up logging files """

    log_out = os.path.join(config['out_dir'], name+'_model.log')
    logging.basicConfig(level=logging.INFO, filename = log_out)
    logger = logging.getLogger(__name__)

    return logger


def _set_dirs(config, name):
    """ set up directories """
    
    # create output directories
    if not os.path.exists(config['out_dir']): os.makedirs(config['out_dir'])
    # make a cache specific for the experiment name
    if not os.path.exists(config['cache']+"/"+name): os.makedirs(config['cache']+"/"+name)
    if not os.path.exists(config['pca_cache']): os.makedirs(config['pca_cache'])

    return 


def _get_data(config, logger, name):
    """ get all the required data for training/testing the classifier """

    # load bert-related stuff
    bert_models = {'bert':'allenai/scibert_scivocab_uncased',
                   'roberta' : 'allenai/biomed_roberta_base'}
    vocab = bert_models[config['bert_model']]

    # load data
    partition, training_generator, validation_generator = load_data(config, vocab)

    # get the embeddings: either from scratch, or from cache
    logger.info(f" Getting {config['embedding_type']} embeddings ...")
    
    if config['embedding_type'] == 'bert':
        embed_shape, train_embeddings, valid_embeddings = load_embeddings(config, name, vocab, training_generator, validation_generator)
    elif config['embedding_type'] == 'specter':
        # load from filepath
        embed_shape, train_embeddings, valid_embeddings = pickle.load(os.path(config["embedding_path"]))
    else:
        raise logger.error("Only BERT and Specter embeddings accepted.")

    # dimension reduction: PCA (either from scratch, or from cache)
    if config["do_pca"]:
        logger.info(' Reducing embedding dimensions...')
        embed_shape, train_embeddings, valid_embeddings = get_pca_embeddings(config, name, train_embeddings, valid_embeddings)

    logger.info(' Dataset is: {} and PCA was performed: {}'.format(name, config["do_pca"]))
    logger.info(f'\n Num. training samples: {len(training_generator)} \
                  \n Num. validation samples: {len(validation_generator)}')

    return embed_shape, train_embeddings, valid_embeddings


if __name__ == '__main__':

    # check cuda
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', help='path to the json config file')
    parser.add_argument('--name', help='experiment name')

    args = parser.parse_args()

    main(args)



