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

# for dealing with multiprocessing/len(ancdata) error
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048*2, rlimit[1]))

def train(config, name, logger, train, valid, shape):
    """ Train classifier """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # get the classifier type
    model_type = config['model_type']
    logger.info(f' Classifier type: {model_type}')

    # general classifiers 
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

    # neural-network based classifiers
    nn_classifiers = {'cnn' : AbstractClassifier('cnn', embedding_size=shape)}
                      # 'fcn' : AbstractClassifier('fcn', embedding_size=shape)}

    # training procedure based on classifier type
    if model_type in probabilistic_classifiers:

        classifier, param_grid = probabilistic_classifiers[model_type]
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
        grids.fit(train['embeddings'], train['labels'])
        end_time = time.time()

        cv_time = end_time - start_time
        logging.info(' CV Parameter search took {} minutes'.format(cv_time/60)) # seconds

        # take cv results into a dataframe and slice row with best parameters
        cv_res = pd.DataFrame.from_dict(grids.cv_results_)
        cv_best_res = cv_res[cv_res.rank_test_roc_auc == 1]

        logger.info((f"Model: {config['model_type']}\n\tBest CV Params: {cv_best_res.params}\n\tTraining:\n\tAUROC: {cv_best_res[['mean_test_roc_auc']].iloc[0]['mean_test_roc_auc']}\
                                    \n\tAP: {cv_best_res[['mean_test_average_precision']].iloc[0]['mean_test_average_precision']}"))

        # get predictions (posterior probability)
        all_predictions = [grids.predict_proba(valid['embeddings']), grids.predict(valid['embeddings']),  valid['labels']]
        train_fitted = [grids.predict_proba(train['embeddings']), grids.predict(train['embeddings']), train['labels']]

        train_df = pd.DataFrame({
            'model_probs': train_fitted[0][:, 1],
            'ground_truth': train['labels'],
            'set': 'training_fitted',
            'data': name,
            'pca': config['do_pca'],
            'model_type': config['model_type'],
            'bert_model': config['bert_model'],
            'ids': train['ids']
        })

        val_df = pd.DataFrame({
            'model_probs': all_predictions[0][:, 1],
            'ground_truth': valid['labels'],
            'set': 'validation',
            'data': name,
            'pca': config['do_pca'],
            'model_type': config['model_type'],
            'bert_model': config['bert_model'],
            'ids': valid['ids']
        })
        
        combined_df = pd.concat([train_df, val_df], axis = 0)

        combined_df.to_csv(os.path.join(config["out_dir"], '{}_preds.csv'.format(name)))
        # save model
        torch.save(classifier, name+"_pretrained_"+config['model_type']+'.p')
            
        # check prediction performance on the validation data
        logger.info(' Getting metrics...')
        
        # AUROC and AUPRC are calculated with P(y == 1) so we want to use index 0 (probs) instead of 1 (thresholded classes) of all_predictions
        roc_auc = metrics('roc_auc', preds = all_predictions[0][:, 1], labels = all_predictions[2])
        ap = metrics('ap', preds = all_predictions[0][:, 1], labels = all_predictions[2])

        roc_auc2 = metrics('roc_auc', preds = train_fitted[0][:, 1], labels = train_fitted[2])
        ap2 = metrics('ap', preds = train_fitted[0][:, 1], labels = train_fitted[2])

        logger.info(f' Validation:\n\tAUROC: {roc_auc} \n\tAP: {ap}')
        embed_shape = np.array(train['embeddings']).shape

        run_res = pd.DataFrame([roc_auc, ap,
                        roc_auc2, ap2,
                        cv_best_res[['mean_test_roc_auc']].iloc[0]['mean_test_roc_auc'],  cv_best_res[['mean_test_average_precision']].iloc[0]['mean_test_average_precision']])
        run_res['metrics'] = ['auroc', 'auprc', 'auroc', 'auprc', 'auroc', 'auprc']
        run_res['set'] = ['validation', 'validation', 'training_fitted', 'training_fitted', 'training', 'training']
        run_res['data'] = name
        run_res['PCA'] = config["do_pca"]
        run_res['model_type'] = config["model_type"]
        run_res['bert_model'] =  config["bert_model"]
        run_res['cv_time'] = cv_time
        run_res['num_vars_train'] = embed_shape[1]
        run_res['num_obs_train'] = embed_shape[0]
        run_res['params'] = str(param_grid) # the parameter grid input
        run_res['best_params'] = str(cv_best_res[['params']].iloc[0, 0])# the params for the best CV model
        run_res['grids'] = str(grids) # sanity check that all parameters were actually passed into gridsearchcv

        run_res.to_csv(os.path.join(config['out_dir'], '{}_results.csv'.format(name)))

    elif config['model_type'] in nn_classifiers:
        
        # convert ids to proper format 
        train_id = torch.tensor([x[0].item() for x in train['ids']])
        valid_id = torch.tensor([x[0].item() for x in valid['ids']])

        # add in IDs 
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(train['embeddings']), \
                                                       torch.tensor(train['labels']), \
                                                       train_id)

        valid_dataset = torch.utils.data.TensorDataset(torch.tensor(valid['embeddings']), \
                                                      torch.tensor(valid['labels']), \
                                                      valid_id)

        # create torch dataloader 
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=config['nn_train_batch_size'])
        valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=True, batch_size=config['nn_valid_batch_size'])

        classifier = nn_classifiers[model_type]
        criterion = CrossEntropyLoss()

        logger.info(classifier)

        optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

        for epoch in range(config['epochs']):

            losses = 0.0

            for i, (data, labels, ids)  in enumerate(train_loader):

                local_data, local_labels = data.to(device).float().unsqueeze(1), \
                                           labels.to(device).type(torch.LongTensor)


                # 0-grad otherwise it sums up 
                optimizer.zero_grad()
                outputs = classifier(local_data)
                loss = criterion(outputs, local_labels)
                loss.backward()
                optimizer.step()

                losses += loss.item()

            if epoch % 25 == 0: 
                print(f'Epoch {epoch} loss: {loss}')

        logger.info("Classifier Training Complete")
                
        # save model
        torch.save(classifier.state_dict(), name+"_pretrained_"+config['model_type']+'.p')

        # check prediction performance on the validation data 
        correct = 0
        total = 0

        with torch.no_grad():
            for i, (data, labels)  in enumerate(valid_loader):
                data, labels = data.to(device).float().unsqueeze(1), \
                               labels.to(device).type(torch.LongTensor)
                outputs = classifier(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            logger.info(' Getting metrics...')
            roc_auc = metrics('roc_auc', preds = predicted, labels = labels)
            ap = metrics('ap', preds = predicted, labels = labels)
            
            logger.info(f'AUROC: {np.mean(roc_auc)} \nAP: {np.mean(ap)}')
            print(f'AUROC: {np.mean(roc_auc)} \nAP: {np.mean(ap)}')
        
            run_res = pd.DataFrame([roc_auc, ap])
            run_res['metrics'] = ['auroc', 'auprc']
            run_res['set'] = ['validation', 'training']
            run_res['data'] = name
            run_res['PCA'] = config["do_pca"]
            run_res['model_type'] = config["model_type"]
            run_res['bert_model'] =  config["bert_model"]
            run_res['cv_time'] = None
            run_res['X_shape'] = train['embeddings'].shape[1]
            run_res['params'] = None
            run_res['best_params'] = None
            run_res['grids'] = None

            run_res.to_csv(os.path.join(config['out_dir'], '{}_results.csv'.format(name)))
    else:
        raise ValueError('Model not implemented yet')

    return