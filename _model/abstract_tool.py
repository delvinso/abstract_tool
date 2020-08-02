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
from AbstractClassifier import AbstractClassifier
from AbstractBert import AbstractBert
from utils import load_data, metrics, load_embeddings, get_bert_embeddings, get_pca_embeddings, save_embeddings

# for dealing with multiprocessing/len(ancdata) error
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048*2, rlimit[1]))

'''
TODO: Add function to do UMAP/embedding visualization  
TODO: Add 1D CNN 
TODO: Add augmented PCA
'''

def main(args):

    # load config file 
    with open(args.config_dir) as f:
        config = json.load(f)

    # set seeds
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # create output directories
    if not os.path.exists(config['out_dir']): os.makedirs(config['out_dir'])
    if not os.path.exists(config['embedding_cache']): os.makedirs(fconfig['embedding_cache'])

    # logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG) # if you want to pipe the logging to a file
    log_out = os.path.join(config['out_dir'], 'model.log')
    logging.basicConfig(level=logging.INFO, filename = log_out)
    logger = logging.getLogger(__name__)

    # load data 
    bert_models = {'bert':'allenai/scibert_scivocab_uncased',
                   'roberta' : 'allenai/biomed_roberta_base'}
    vocab = bert_models[config['bert_model']]

    partition, training_generator, validation_generator = load_data(config, vocab)

    # get the embeddings (either from scratch, or from cache)
    logger.info(' Getting BERT embeddings...')
    train_embeddings, valid_embeddings = load_embeddings(config, bert_models, training_generator, validation_generator)
   
    # dimension reduction: PCA (either from scratch, or from cache)
    if config["do_pca"]:
        logger.info(' Reducing embedding dimensions...')
        embed_shape, train_embeddings, valid_embeddings = get_pca_embeddings(train_embeddings, valid_embeddings)

    logger.info(' Dataset is {} and PCA was performed: {}'.format(name, config["do_pca"]))
    logger.info(f'\n Num. training samples: {len(training_generator)} \
                  \n Num. validation samples: {len(validation_generator)}')

    if config.train:
        _train(config, args.name, logger, train_embeddings, valid_embeddings)
    elif config.test:
        # _test(config, args.name, logger)
        pass
    else:
        raise logger.error(f"Neither train nor test mode activated.")


def _train(config, name, logger, train, valid):
    """ Train classifier """

    # get the classifier
    model_type = config['model_type']
    logger.info(f' Classifier type: {model_type}')

    # classifiers
    probabilistic_classifiers = {'knn' : (KNeighborsClassifier(n_jobs=128),
                                         {'n_neighbors': [3, 5, 7, 11, 13, 15], 'weights': ['distance']}),

                                'lasso': (LogisticRegression(n_jobs=128, penalty='l1', solver = 'saga',
                                                                random_state=0, verbose=1, max_iter=5000, tol=0.001),
                                         {'C': [0.001, 0.01, 0.1, 1, 10, 100] }),

                                'ridge': (LogisticRegression(n_jobs=128, penalty='l2', solver = 'saga',
                                                                random_state=0, verbose=1, max_iter=5000, tol=0.001),
                                         {'C': [0.001, 0.01, 0.1, 1, 10, 100]}),

                                'rf': (RandomForestClassifier(n_jobs=128, verbose=1, random_state=0),
                                        {'n_estimators': [500, 750, 1000],
                                         'max_features': ["sqrt", 0.05, 0.1],
                                         'min_samples_split': [2, 4, 8]})}

    # neural network based classifiers
    dnn_classifiers = {'cnn' : Classifier(config, 'cnn', embedding_size=embed_shape[0]),
                       'fcn' : Classifier(config, 'fcn')}

    # training procedure based on classifier type
    if model_type in probabilistic_classifiers:
        classifier, param_grid = probabilistic_classifiers[model_type
        logger.info(classifier)
        logger.info(param_grid)

         # grid search arguments
        grids = GridSearchCV(classifier, \
                            param_grid, \
                            cv=10, \
                            scoring={'average_precision', 'roc_auc'}, \
                            refit='roc_auc')

        # perform the grid search
        start_time = time.time()
        grids.fit(valid[:0], train[:1])
        end_time = time.time()

        cv_time = end_time - start_time
        logging.info(' CV Parameter search took {} minutes'.format(cv_time/60)) # seconds

        # take cv results into a dataframe and slice row with best parameters
        cv_res = pd.DataFrame.from_dict(grids.cv_results_)
        cv_best_res = cv_res[cv_res.rank_test_roc_auc == 1]

        logger.info((f"Model: {config['model_type']}\n\tBest CV Params: {cv_best_res.params}\n\tTraining:\n\tAUROC: {cv_best_res[['mean_test_roc_auc']].iloc[0]['mean_test_roc_auc']}\
                                    \n\tAP: {cv_best_res[['mean_test_average_precision']].iloc[0]['mean_test_average_precision']}"))

        # get predictions (posterior probability)
        all_predictions = [grids.predict_proba(valid[:0]), grids.predict(valid[:0]),  valid[:1]]
        train_fitted = [grids.predict_proba(train[:0]), grids.predict(train[:0]), train[:1]]

        train_df = pd.DataFrame({
            'model_probs': train_fitted[0][:, 1],
            'ground_truth': augmented_train['labels'],
            'set': 'training_fitted',
            'data': name,
            'pca': config['pca'],
            'model_type': config['model_type'],
            'bert_model': config['bert_model'],
            'ids': augmented_train['ids']
        })

        val_df = pd.DataFrame({
            'model_probs': all_predictions[0][:, 1],
            'ground_truth': augmented_valid['labels'],
            'set': 'validation',
            'data': name,
            'pca': config['pca'],
            'model_type': config['model_type'],
            'bert_model': config['bert_model'],
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

        run_res = pd.DataFrame([roc_auc, ap,
                        roc_auc2, ap2,
                        cv_best_res[['mean_test_roc_auc']].iloc[0]['mean_test_roc_auc'],  cv_best_res[['mean_test_average_precision']].iloc[0]['mean_test_average_precision']])
        run_res['metrics'] = ['auroc', 'auprc', 'auroc', 'auprc', 'auroc', 'auprc']
        run_res['set'] = ['validation', 'validation', 'training_fitted', 'training_fitted', 'training', 'training']
        run_res['data'] = name
        run_res['PCA'] = config["PCA"]
        run_res['model_type'] = config["model_type"]
        run_res['bert_model'] =  config["bert_model"]
        run_res['cv_time'] = cv_time
        run_res['X_shape'] = train_reduc.shape[1] if PCA == 'use_pca' else train_embed_ss.shape[1]
        run_res['params'] = str(param_grid) # the parameter grid input
        run_res['best_params'] = str(cv_best_res[['params']].iloc[0, 0])# the params for the best CV model
        run_res['grids'] = str(grids) # sanity check that all parameters were actually passed into gridsearchcv

        run_res.to_csv(os.path.join(config['out_dir'], '{}_results.csv'.format(name)))

    elif config['model_type'] in dnn_classifiers: # if DNN, no need to do CV
        classifier = dnn_classifiers[model_type]
        criterion = CrossEntropyLoss()

        logger.info(classifier)
        logger.info(classifier.params())

        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(config['epochs']):

            losses = 0.0
            i = 0
            for local_data, local_labels in train:
                local_data, local_labels = local_data.to(device).float().squeeze(1), \
                                                        local_labels.to(device).float()
                print(local_labels)
                optimizer.zero_grad()
                outputs = classifier(local_data)
                loss = criterion(outputs, local_labels)
                loss.backward()
                optimizer.step()
                i += 1 

                losses += loss.item()

                if i % 2000 == 1999: 
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))

        logger.info("Classifier Training Complete")
                
        # save model
        torch.save(classifier.state_dict(), config["pretrained_classifier"_]+config['model_type']+'.p')

        # check prediction performance on the validation data 
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valid:
                embeddings, labels = data
                outputs = net(embeddings)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            logger.info(' Getting metrics...')
            roc_auc = metrics('roc_auc', preds = all_predictions[1], labels = all_predictions[2])
            ap = metrics('ap', preds = all_predictions[1], labels = all_predictions[2])
            
            logger.info(f'AUROC: {np.mean(roc_auc)} \nAP: {np.mean(ap)}')

            run_res = pd.DataFrame([roc_auc, ap])
            run_res['metrics'] = ['auroc', 'auprc']
            run_res['set'] = ['validation', 'validation', 'training_fitted', 'training_fitted', 'training', 'training']
            run_res['data'] = name
            run_res['PCA'] = config["PCA"]
            run_res['model_type'] = config["model_type"]
            run_res['bert_model'] =  config["bert_model"]
            run_res['cv_time'] = None
            run_res['X_shape'] = train.shape[1] 
            run_res['params'] = None
            run_res['best_params'] = None
            run_res['grids'] = None

            run_res.to_csv(os.path.join(config['out_dir'], '{}_results.csv'.format(name)))
    else:
        raise ValueError('Model not implemented yet')


if __name__ == '__main__':

    # check cuda
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', help='path to the json config file')
    parser.add_argument('--name', help='experiment name')

    args = parser.parse_args()

    main(args)




