import io
import torch
import logging
import json
import argparse
import pickle
import os
from pprint import pprint

# 3rd party libraries
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
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

# for dealing with multiprocessing/len(ancdata) error
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048*2, rlimit[1]))

'''
TODO: Add function to do UMAP/embedding visualization  
TODO: incorporate phase 2
'''

def main():

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
    TYPE = TRAIN_FILE.split('/')[-1][:-4] # retrieve file name without the extension
    MAX_EPOCHS = config['epochs']
    MAX_LEN = config['max_len']
    SEED = config['seed']
    PRETRAINED_KNN = config['clustering_model']
    TRAIN_MODE = config['train']
    TEST_MODE = config['test']
    OUT_DIR = config['out_dir']

    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)

    # logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG) # if you want to pipe the logging to a file
    #TODO take as parameter to json
    log_out = os.path.join(OUT_DIR, 'model.log')
    logging.basicConfig(level=logging.INFO, filename = log_out)
    logger = logging.getLogger(__name__)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

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
    embedding_model = BertForAbstractScreening(PRETRAINED_WEIGHTS)

    # multiple GPUs
    if torch.cuda.device_count() > 1:
        print("GPUs Available: ", torch.cuda.device_count())
        embedding_model = torch.nn.DataParallel(embedding_model, device_ids=[0, 1, 2])

    embedding_model.eval().to(device)

    # load data
    training_generator, validation_generator = load_data(TRAIN_FILE, tokenizer, max_len = MAX_LEN)
    logger.info('Dataset is {}'.format(TYPE))
    logger.info(f'\nNum. training samples: {len(training_generator)} \
                  \nNum. validation samples: {len(validation_generator)}')

    ######## BERT embeddings and PCA ########

    if TEST_MODE:
        test_generator, _ = load_data(TEST_FILE, tokenizer, max_len = MAX_LEN, proportion=0)
        logger.info(f'\nNum. testing samples: {len(test_generator)}')

        # get BERT embeddings
        logger.info(' Getting augmented BERT embeddings...')
        augmented_train = get_embeddings(training_generator, embedding_model)
        augmented_valid = get_embeddings(validation_generator, embedding_model)
        augmented_test = get_embeddings(test_generator, embedding_model)

        logger.info(' Standardizing ')
        # standardizing since KNN is non-parametric and is not robust to feature parameters
        ss = StandardScaler()
        train_embed_ss = ss.fit_transform(augmented_train['embeddings'])
        valid_embed_ss = ss.transform(augmented_valid['embeddings'])
        test_reduc_ss = ss.transform(augmented_test['embeddings'])

        # Dimension reduction: PCA or UMAP (?)
        logger.info(' Doing PCA...')
        pca_model = decomposition.PCA(n_components = 0.99) # this can be a parameter down the road, but for debugging it's fine
        train_reduc = pca_model.fit_transform(train_embed_ss)
        val_reduc = pca_model.transform(valid_embed_ss)
        test_reduc = pca_model.transform(test_reduc_ss)

        # all_augmented = np.hstack((augmented_train['embeddings'], \
        #                            augmented_valid['embeddings'], \
        #                            augmented_test['embeddings']))

        # Dimension reduction: PCA or UMAP (?)
        # FIX THIS.
        # logger.info(' Doing PCA...')
        # pca_model = decomposition.PCA(n_components='mle')
        # all_reduced = pca_model.fit_transform(all_augmented)
        #
        # reduced_train = all_reduced[ :len(augmented_train['embeddings'])]
        # reduced_valid = all_reduced[len(augmented_valid['embeddings']): -len(augmented_test['embeddings'])]
        # reduced_test = all_reduced[-len(augmented_test['embeddings']): ]
    else:
        # get BERT embeddings
        logger.info(' Getting augmented BERT embeddings...')
        augmented_train = get_embeddings(training_generator, embedding_model)
        augmented_valid = get_embeddings(validation_generator, embedding_model)

        # save down embeddings as a pickle
        with open(os.path.join(OUT_DIR, '{}_test_aug_valid.pkl'.format(TYPE)), 'wb') as handle:
            pickle.dump(augmented_valid, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(OUT_DIR, '{}_test_aug_train.pkl'.format(TYPE)), 'wb') as handle:
            pickle.dump(augmented_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(' Standardizing ')
        # standardizing since KNN is non-parametric and is not robust to feature parameters
        ss = StandardScaler()
        train_embed_ss = ss.fit_transform(augmented_train['embeddings'])
        valid_embed_ss = ss.transform(augmented_valid['embeddings'])

        logger.info(' Doing PCA...')
        pca_model = decomposition.PCA(n_components = 0.99) # this can be a parameter down the road, but for debugging it's fine
        train_reduc = pca_model.fit_transform(train_embed_ss)
        val_reduc = pca_model.transform(valid_embed_ss)

    ######## BERT embeddings and PCA ########

    ######## KNN ########
    if TRAIN_MODE:

        logger.info(' Clustering with grid search...')
        # initialize model
        clustering_model = KNeighborsClassifier(  n_jobs = 32)
        # optimal # of neighbours? this can be a dataset specific thing. no need to identify a single value..
        param_grid = {'n_neighbors': [3, 5, 7, 11, 13, 15],
                      'weights': ['distance']}
        # grid search arguments
        grids = GridSearchCV(clustering_model,
                             param_grid,
                             cv= 10 ,
                             scoring = {'average_precision', 'roc_auc'},
                             refit = 'roc_auc')
        # perform the grid search
        grids.fit(train_reduc, augmented_train['labels'])
        # take cv results into a dataframe and slice row with best parameters
        cv_res = pd.DataFrame.from_dict(grids.cv_results_)#.groupby('param_n_neighbors').agg('mean').reset_index()
        cv_best_res = cv_res[cv_res.rank_test_roc_auc == 1]

        logger.info((f"KNN Training:\n\tAUROC: {cv_best_res[['mean_test_roc_auc']].iloc[0]['mean_test_roc_auc']}\
                                    \n\tAP: {cv_best_res[['mean_test_average_precision']].iloc[0]['mean_test_average_precision']}")
                    )
    else:
        # load pretrained KNN (pickled)
        clustering_model = pickle.load(PRETRAINED_KNN)
    ######## KNN ########

    # predict the posterior probability of the validation embeddings, check label..
    all_predictions = [grids.predict_proba(val_reduc),
                       grids.predict(val_reduc),
                       augmented_valid['labels']]

    # check prediction performance on the validation data
    logger.info(' Getting metrics...')
    # AUROC and AUPRC are calculated with P(y == 1) so we want to use index 0 instead of 1 of all_predictions
    roc_auc = metrics('roc_auc', preds = [x[1] for x in all_predictions[0]], labels = all_predictions[2])
    ap = metrics('ap', preds = [x[1] for x in all_predictions[0]], labels = all_predictions[2])

    logger.info(f'KNN Validation:\n\tAUROC: {roc_auc} \n\tAP: {ap}')

    # --- saving down results for debugging ----
    run_res = pd.DataFrame([roc_auc,
                            ap,
                            cv_best_res[['mean_test_roc_auc']].iloc[0]['mean_test_roc_auc'],
                            cv_best_res[['mean_test_average_precision']].iloc[0]['mean_test_average_precision']])
    run_res['metrics'] = ['auroc', 'auprc', 'auroc', 'auprc']
    run_res['set'] = ['validation', 'validation', 'training', 'training']
    run_res['data'] = TYPE

    run_res.to_csv(os.path.join(OUT_DIR, '{}_results.csv'.format(TYPE)))


    # save all models
    #
    #     pickle.dump(clustering_model)# embedding_model.module.save_state_dict('embedding_BFAS_'+TYPE+'.pt')

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



