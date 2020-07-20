# done:
# - caching for the embeddings, load if they exist, specific to each model (ie. bert, roberta..)
# - dictionary for whatever huggingface transformer we want(need to test) but works for roberta and bert
# - dictionary for sklearn probabilistic classifier allowing for robustness
# - saving down predictions with labels and ids, as well as model metrics, for both cv and fitted values
# - swapping between PCA and no PCA
# - timing cross-validation and shape of X to output file
# - standardizing the CLS token
# - fixed get_embeddings() to properly return index
# - drop invalid abstracts from load_data()


# todo:
# - add code for linear layer/various other neural net layers
# - refactor PCA instead of having two objects (one for standardized, one for pca)
# - modularity?


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
from BertForAbstractScreening import BertForAbstractScreening
from data_tools import load_data, metrics, get_embeddings

# for dealing with multiprocessing/len(ancdata) error
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048*2, rlimit[1]))

def main():

    # ---- set up parser ------
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', help='path to the json config file')
    args = parser.parse_args()

    # ---- load config file ------
    with open(args.config_dir) as f:
        config = json.load(f)

    TRAIN_FILE = config['train_file']
    TEST_FILE = config['test_file']
    TYPE = TRAIN_FILE.split('/')[-1][:-4] # retrieve file name without the extension
    MAX_LEN = config['max_len']
    SEED = config['seed']
    PRETRAINED_KNN = config['clustering_model']
    TRAIN_MODE = config['train']
    TEST_MODE = config['test']
    OUT_DIR = config['out_dir']
    PCA = config['pca']
    MODEL_TYPE = config['model_type']
    BERT_MODEL = config['bert_model']
    embedding_cache = 'embeddings_cache'
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    if not os.path.exists(embedding_cache): os.makedirs(embedding_cache)


    # TODO: take as parameter to json
    # logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG) # if you want to pipe the logging to a file
    log_out = os.path.join(OUT_DIR, f'{MODEL_TYPE}_model.log')
    logging.basicConfig(level=logging.INFO, filename = log_out)
    logger = logging.getLogger(__name__)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")


    # ---- reproducibility -----
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)

    # config checks (not sure what this does)
    if not TRAIN_MODE and not PRETRAINED_KNN:
        logger.error('Must provide path to a trained clustering model if training mode is False.')

    if TEST_MODE and not TEST_FILE:
        logger.error('Must provide path to a testing file with data if testing mode is True.')

    # ---- load models and tokenizer -----
    # model specific params
    bert_models = {'bert':'allenai/scibert_scivocab_uncased',
                   'roberta' : 'allenai/biomed_roberta_base'}

    VOCAB = bert_models[BERT_MODEL]

    tokenizer = AutoTokenizer.from_pretrained(VOCAB)
    embedding_model = BertForAbstractScreening(VOCAB)

    # ---- multiple GPU check ----
    if torch.cuda.device_count() > 1:
        print("GPUs Available: ", torch.cuda.device_count())
        embedding_model = torch.nn.DataParallel(embedding_model, device_ids=[0, 1, 2])

    # ---- no fine tuning ---
    embedding_model.eval().to(device)

    # ---- load data -----
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

    else:
        # TODO: flexibility to what directory it needs to be saved in to, currently saved in running directory
        train_pickle = os.path.join(embedding_cache, '{}_{}_cls_embed_train.pkl'.format(TYPE, BERT_MODEL))
        valid_pickle = os.path.join(embedding_cache, '{}_{}_cls_embed_valid.pkl'.format(TYPE, BERT_MODEL))

        # caching the embeddings
        if os.path.exists(train_pickle):
            logger.info(' Loading pickles...')
            with open(train_pickle, "rb") as input_file:
                augmented_train = pickle.load(input_file)

            with open(valid_pickle, "rb") as input_file:
                augmented_valid = pickle.load(input_file)
        else:
            # get BERT embeddings
            logger.info(' Getting augmented BERT embeddings...')
            augmented_train = get_embeddings(training_generator, embedding_model)
            augmented_valid = get_embeddings(validation_generator, embedding_model)
            with open(os.path.join(valid_pickle), 'wb') as handle:
                pickle.dump(augmented_valid, handle, protocol = pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(train_pickle), 'wb') as handle:
                pickle.dump(augmented_train, handle, protocol = pickle.HIGHEST_PROTOCOL)

        logger.info(' Standardizing ')
        ss = StandardScaler()
        train_embed_ss, valid_embed_ss = ss.fit_transform(augmented_train['embeddings']), ss.transform(augmented_valid['embeddings'])

        if PCA == 'use_pca':
            logger.info(' Doing PCA...')
            pca_model = decomposition.PCA(n_components = 0.99) # this can be a parameter down the road, but for debugging it's fine
            train_reduc, val_reduc =  pca_model.fit_transform(train_embed_ss), pca_model.transform(valid_embed_ss)

    ######## BERT embeddings and PCA ########

    if TRAIN_MODE:

        logger.info(' Clustering with grid search...')
        # initialize model

        logger.info(MODEL_TYPE)
        # note to self: c is inverse of regularization strength!! (opposite of glmnet...)
        probabilistic_classifiers = {'knn' : (KNeighborsClassifier(n_jobs = 128),
                                              {'n_neighbors': [3, 5, 7, 11, 13, 15], 'weights': ['distance']}),

                                    'lasso': (LogisticRegression(n_jobs=128, penalty='l1', solver = 'saga',
                                                                 random_state=0, verbose = 1, max_iter = 5000, tol =  0.001),
                                              {'C': [0.001, 0.01, 0.1, 1, 10, 100] }),

                                     'ridge': (LogisticRegression(n_jobs=128, penalty='l2', solver = 'saga',
                                                                  random_state=0, verbose = 1, max_iter = 5000, tol =  0.001),
                                               {'C': [0.001, 0.01, 0.1, 1, 10, 100]}),

                                     'rf': (RandomForestClassifier(n_jobs = 128, verbose = 1, random_state = 0),
                                            {'n_estimators': [500, 750, 1000],
                                             'max_features': ["sqrt", 0.05, 0.1],
                                             'min_samples_split': [2, 4, 8]})}

        if MODEL_TYPE in probabilistic_classifiers:
            model, param_grid = probabilistic_classifiers[MODEL_TYPE]
            logger.info(model)
            logger.info(param_grid)
        else:
            raise ValueError('Model not implemented yet')

        # grid search arguments
        grids = GridSearchCV(model, param_grid, cv= 10, scoring = {'average_precision', 'roc_auc'}, refit = 'roc_auc')

        # perform the grid search
        start_time = time.time()
        if PCA == 'use_pca':
            print('Performing PCA...')
            grids.fit(train_reduc, augmented_train['labels'])
        else:
            grids.fit(train_embed_ss, augmented_train['labels'])
        end_time = time.time()

        cv_time = end_time - start_time
        logging.info('CV Parameter search took {} minutes'.format(cv_time/60)) # seconds

        # take cv results into a dataframe and slice row with best parameters
        cv_res = pd.DataFrame.from_dict(grids.cv_results_)
        cv_best_res = cv_res[cv_res.rank_test_roc_auc == 1]

        logger.info((f"Model: {MODEL_TYPE}\n\tBest CV Params: {cv_best_res.params}\n\tTraining:\n\tAUROC: {cv_best_res[['mean_test_roc_auc']].iloc[0]['mean_test_roc_auc']}\
                                    \n\tAP: {cv_best_res[['mean_test_average_precision']].iloc[0]['mean_test_average_precision']}"))
    else:
        # load pretrained KNN (pickled)
        clustering_model = pickle.load(PRETRAINED_KNN)


    # get predictions (posterior probability)

    #TODO: refactor
    if PCA == 'use_pca':
        all_predictions = [grids.predict_proba(val_reduc), grids.predict(val_reduc), augmented_valid['labels']]
        train_fitted =  [grids.predict_proba(train_reduc), grids.predict(train_reduc), augmented_train['labels']]
    else:
        all_predictions = [grids.predict_proba(valid_embed_ss), grids.predict(valid_embed_ss),  augmented_valid['labels']]
        train_fitted = [grids.predict_proba(train_embed_ss), grids.predict(train_embed_ss), augmented_train['labels']]
    # ----- save results for plotting AUROC ------


    train_df = pd.DataFrame({
        'model_probs': train_fitted[0][:, 1],
        'ground_truth': augmented_train['labels'],
        'set': 'training_fitted',
        'data': TYPE,
        'pca': PCA,
         'model_type': MODEL_TYPE,
        'bert_model': BERT_MODEL,
        'ids': augmented_train['ids']
    })

    val_df = pd.DataFrame({
        'model_probs': all_predictions[0][:, 1],
        'ground_truth': augmented_valid['labels'],
        'set': 'validation',
        'data': TYPE,
        'pca': PCA,
        'model_type': MODEL_TYPE,
        'bert_model': BERT_MODEL,
        'ids': augmented_valid['ids']
    })
    
    combined_df = pd.concat([train_df, val_df], axis = 0)

    combined_df.to_csv(os.path.join(OUT_DIR, '{}_preds.csv'.format(TYPE)))

    # check prediction performance on the validation data
    logger.info(' Getting metrics...')
    
    # AUROC and AUPRC are calculated with P(y == 1) so we want to use index 0 (probs) instead of 1 (thresholded classes) of all_predictions
    
    roc_auc = metrics('roc_auc', preds = all_predictions[0][:, 1], labels = all_predictions[2])
    ap = metrics('ap', preds = all_predictions[0][:, 1], labels = all_predictions[2])

    roc_auc2 = metrics('roc_auc', preds = train_fitted[0][:, 1], labels = train_fitted[2])
    ap2 = metrics('ap', preds = train_fitted[0][:, 1], labels = train_fitted[2])

    logger.info(f' Validation:\n\tAUROC: {roc_auc} \n\tAP: {ap}')

    # --- saving down results ----
    run_res = pd.DataFrame([roc_auc, ap,
                            roc_auc2, ap2,
                            cv_best_res[['mean_test_roc_auc']].iloc[0]['mean_test_roc_auc'],  cv_best_res[['mean_test_average_precision']].iloc[0]['mean_test_average_precision']])
    run_res['metrics'] = ['auroc', 'auprc', 'auroc', 'auprc', 'auroc', 'auprc']
    run_res['set'] = ['validation', 'validation', 'training_fitted', 'training_fitted', 'training', 'training']
    run_res['data'] = TYPE
    run_res['PCA'] = PCA
    run_res['model_type'] = MODEL_TYPE
    run_res['bert_model'] =  BERT_MODEL
    run_res['cv_time'] = cv_time
    run_res['X_shape'] = train_reduc.shape[1] if PCA == 'use_pca' else train_embed_ss.shape[1]
    run_res['params'] = str(param_grid) # the parameter grid input
    run_res['best_params'] = str(cv_best_res[['params']].iloc[0, 0])# the params for the best CV model
    run_res['grids'] = str(grids) # sanity check that all parameters were actually passed into gridsearchcv

    run_res.to_csv(os.path.join(OUT_DIR, '{}_results.csv'.format(TYPE)))


if __name__ == '__main__':
    main()


