import io
import torch
import logging
import json
import argparse
import os
import pickle
from pprint import pprint

# 3rd party libraries
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
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

''' TODO:
    - add other models for comparison, set up different structure on top of BERT (e.g., use AutoModel+torch.nn instead of BERTForSequenceClassification)
    - plot loss with tensorboard
    - testing function
    - OTHER TODO: line 55, 64, 106, 145, 150
    - sample and pos_frac
'''

def load_data(csv_file, tokenizer, max_len: int=512, partition: dict=None, labels: dict=None, sample =  None, pos_frac = None):

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

    # columns: [0] unique ID, [1] text, [2] label
    dataset = pd.read_csv(csv_file, header=None, sep='\t')
    # convert labels to integer
    dataset.iloc[:, 2] = pd.to_numeric(dataset.iloc[:, 2], errors = 'coerce' )
    # dataset = dataset.sample(frac = 0.1, axis = 0)
    # print(dataset.head())
    # print(dataset.shape)

    # create list of train/valid IDs if not provided

    if not partition and not labels:
        labels = {}
        ids = list(dataset.iloc[:,0])

        if (sample and pos_frac) : # TODO: check for both sample and pos frac
            # print(sample)
            pos_ids = list(dataset.iloc[:, 0][dataset.iloc[:, 2] == 1] )
            # print((pos_ids))
            neg_ids = list(dataset.iloc[:, 0][dataset.iloc[:, 2] == 0] )
            # print((neg_ids))
            num_pos_samples = int(sample * pos_frac)
            num_neg_samples = int(sample - num_pos_samples)
            print('Sample Size: {}, Positive Fraction: {}, Number Positive Samples: {}, Number Negative Samples: {}'.format(sample, pos_frac,
                                                                                                                            num_pos_samples, num_neg_samples))
            pos_ids_sample = resample(pos_ids, n_samples = num_pos_samples, replace = False)
            neg_ids_sample = resample(neg_ids, n_samples = num_neg_samples, replace = False)


            # print(len(pos_ids_sample))
            # print(len(neg_ids_sample))

            np.random.shuffle(pos_ids_sample)
            np.random.shuffle(neg_ids_sample)
            # check if  less than 1.0
            # sample by positive and negative ids so when n is small, there is still an even split by class for training and eval.
            partition = {'train': pos_ids_sample[ :int(len(pos_ids_sample) *0.7)] + neg_ids_sample[:int(len(neg_ids_sample) *0.7)],
                         'valid':  pos_ids_sample[ int(len(pos_ids_sample) *0.7):] + neg_ids_sample[int(len(neg_ids_sample) *0.7):]
                         }
            ids = neg_ids_sample + pos_ids_sample
        else:
            total_len = len(ids)
            # print(total_len)
            np.random.shuffle(ids)
            partition = {'train': ids[ :int(total_len*0.7)],
                         'valid': ids[int(total_len*0.7): ]
                         }


        dataset = dataset[dataset.iloc[:, 0].isin(ids)]
        for i in dataset.iloc[:,0]:
            labels[i] = dataset.loc[i][2] # we use loc to refer to the name now instead of idx

        # print(pd.Series(partition['train']))
        # print(pd.Series(partition['valid']))


    # set parameters for DataLoader -- num_workers = cores
    params = {'batch_size': 18,
              'shuffle': True,
              'num_workers': 16
              }

    ### NOTE: the tokenizer.encocde_plus function does the token/special/map/padding/attention all in one go

    try:
        dataset[1] = dataset[1].apply(lambda x: tokenizer.encode(x,
                                                                 add_special_tokens=True,
                                                                 max_length = max_len,
                                                                 pad_to_max_length=True)) # there's a line in NCDS, 5003,
        # 'spectrum burn case basaveshwara' that's throwing a ValueError
    except ValueError:
        print(dataset[1])
        # dataset[1] = dataset[1].apply(lambda x: tokenizer.encode(torch, add_special_tokens=True, max_length = max_len))

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
    # TODO: add logic to account for  when there is only one class label.
    """ Provides various metrics between predictions and labels.

    Arguments:vi co
        metric_type {str} -- type of metric to use ['flat_accuracy', 'f1', 'roc_auc', 'precision', 'recall']
        preds {list} -- predictions.
        labels {list} -- labels.

    Returns:
        int -- prediction accuracy
    """
    assert metric_type in ['flat_accuracy', 'f1', 'roc_auc', 'ap'], 'Metrics must be one of the following: [\'flat_accuracy\', \'f1\', \'roc_auc\'] \
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
        try:
            auc = roc_auc_score(labels, preds)
        except ValueError:
            auc = np.NaN
        return auc
    elif metric_type == 'precision':
        return precision_score(labels, preds)
    elif metric_type == 'recall':
        return recall_score(labels, preds)
    elif metric_type == 'ap':
        try:
            ap = average_precision_score(labels, preds)
        except ValueError:
            ap = np.Nan
        return ap



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

    with open(args.config_dir) as f:
        config = json.load(f)

    FILE = config['file']
    TYPE = FILE.split('/')[-1][:-4] # retrieve file name without the extension
    MAX_EPOCHS = config['epochs']
    MAX_LEN = config['max_len']
    SEED = config['seed']
    RUNS = config['runs']

    #TODO - check for both if one exists.
    try:
        SAMPLE = config['sample']
    except KeyError:
        SAMPLE = None

    try:
        POS_FRAC = config['pos_frac']
    except KeyError:
        POS_FRAC = None

    # assert(POS_FRAC < 1.0), 'Positive Fraction should be 1 or less'

    # setting up seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)

    # BERT-specific variables, load model + tokenizer, using pretrained SciBERT
    pretrained_weights = 'allenai/scibert_scivocab_uncased'
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

    epoch_count = []
    epoch_train_ap_list = []
    epoch_train_auroc_list = []
    epoch_val_ap_list = []
    epoch_val_auroc_list = []
    epoch_sample = []
    epoch_train_loss_list = []
    epoch_val_loss_list = []
    run_list = []
    epoch_val_ids = []
    epoch_val_preds = []
    epoch_val_labels = []
    epoch_train_ids = []
    epoch_train_preds = []
    epoch_train_labels = []
    pos_frac_list = []

    #  TODO: wrap runs above the training/evaluation, above model and optimizer instancing so each run is unique?
    runs = RUNS

    for run in range(runs):
        logger.info('############################### Run: {} ############################### '.format(run))

        model = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states=True)

        if torch.cuda.device_count() > 1:
            print("GPUs Available: ", torch.cuda.device_count())
            model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])

        model.to(device)


        # load data
        training_generator, validation_generator = load_data(FILE, tokenizer, MAX_LEN, sample = SAMPLE, pos_frac = POS_FRAC)

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
            ids_list = []

            model.train().to(device)
            i = 0 # enumerate?
            for local_batch, local_labels, ids in (training_generator):
                local_batch, local_labels = local_batch.to(device).long().squeeze(1), \
                                            local_labels.to(device).long()
                ids = ids.detach().cpu()

                # print(ids)

                model.zero_grad()
                # TODO: attention mask None for now

                output, embeddings = model(local_batch)
                # # print(embeddings)
                # print(len(embeddings))#.shape)
                # # print(embeddings[0].shape)
                # [print(embeddings[x].shape) for x in range(len(embeddings))] # b x len tokens x # of hidden units, length of # of hidden layers
                # embeddings_stacked = torch.stack(embeddings, dim=1) # now 13 x 18 x 128 x 768, or # of hidden layers x b x len_tokens x # of hidden units

                logits = output

                loss_fct = CrossEntropyLoss()
                # print(logits.view(-1, 2))
                loss = loss_fct(logits.view(-1, 2), local_labels.view(-1))

                total_train_loss += loss.item()
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                probs = torch.softmax(logits, dim = 1)
                probs = probs.detach().cpu().numpy()
                # print(probs)
                for j in range(probs.shape[0]):
                    preds = probs[j][1] # get p(y == 1) from each sample
                    preds_list.append(preds)
                # print(preds_list)

                # print(local_labels.detach().cpu().numpy())
                label_ids = local_labels.detach().cpu().numpy().flatten()#[0]
                labels_list.append(label_ids)
                ids_list.append(ids.detach().cpu().numpy())
                # total_eval_f1 += metrics('f1', logits, label_ids)

                #TODO: check nunique labels
                if (i % 10 == 0) & (i > 0):# & all(np.unique(labels_list).shape[0] == 2):
                    running_roc_auc = metrics('roc_auc', preds = preds_list, labels = labels_list)
                    running_ap = metrics('ap', preds = preds_list, labels = labels_list)
                    logging.info('Training Batch: {}, AUC: {:3f}, AP {:3f}'.format(i, running_roc_auc, running_ap))
                i = i + 1

            avg_train_loss = total_train_loss / len(training_generator)
            total_train_roc_auc = metrics('roc_auc', preds = preds_list, labels = labels_list)
            total_train_ap = metrics('ap', preds = preds_list, labels = labels_list)
            logger.info(f"  Average training loss: {avg_train_loss:.3f}, Training AUC: {total_train_roc_auc:.3f}, AP: {total_train_ap:.3f}")



            ########################## Validation ##########################
            model.eval().to(device)

            total_eval_loss = 0
            nb_eval_steps = 0
            val_labels_list = []
            val_preds_list = []
            best_eval_loss = float('Inf')
            embeddings_list = []
            val_ids_list = []

            with torch.set_grad_enabled(False):
                logger.info(' Validating')
                i = 0 # enumerate instead?
                for local_batch, local_labels, ids in (validation_generator):
                    local_batch, local_labels = local_batch.to(device).long().squeeze(1), \
                                                local_labels.to(device).long()

                    output, embeddings = model(local_batch)
                    logits = output

                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, 2), local_labels.view(-1))

                    total_eval_loss += loss.item()

                    probs = torch.softmax(logits, dim = 1)
                    probs = probs.detach().cpu().numpy()
                    # print(probs)

                    label_ids = local_labels.detach().cpu().numpy().flatten()#[0]
                    # total_eval_accuracy += metrics('flat_accuracy', logits, label_ids)

                    for j in range(probs.shape[0]):
                        preds = probs[j][1] # get p(y == 1) from each sample
                        val_preds_list.append(preds)

                    val_labels_list.append(label_ids)
                    val_ids_list.append(ids.detach().cpu().numpy())
                    # embeddings_list.append(embeddings[-2][:, 0, :].detach().cpu().numpy()) # we take only the first token [CLS] and its hidden units
                    embeddings_list.append(embeddings[-2].detach().cpu().numpy()) # second last layer of embeddings

                    # total_eval_f1 += metrics('f1', logits, label_ids)
                    if (i % 10 == 0) & (i > 0):# & all(np.unique(labels_list).shape[0] == 2):
                        running_roc_auc = metrics('roc_auc', preds = val_preds_list, labels = val_labels_list)
                        running_ap = metrics('ap', preds = val_preds_list, labels = val_labels_list)
                        logging.info('Validation Batch: {}, AUC: {:3f}, AP {:3f}'.format(i, running_roc_auc, running_ap))
                    i = i + 1

                    # total_eval_precision +=  metrics('precision', logits, label_ids)
                    # total_eval_recall += metrics('recall', logits, label_ids)
            total_eval_roc_auc = metrics('roc_auc', preds = val_preds_list, labels = val_labels_list)



            # TODO: clean this up
            # avg_val_accuracy = total_eval_accuracy / len(validation_generator) # no good w/ logits
            # avg_f1 = total_eval_f1 / len(validation_generator)
            avg_roc_auc = total_eval_roc_auc / len(validation_generator)
            # avg_precision = total_eval_precision / len(validation_generator)
            # avg_recall = total_eval_recall / len(validation_generator)
            # logger.info(f"  Accuracy: {avg_val_accuracy:.2f}\n \
            #                 F1 Score: {avg_f1:.2f}\n \
            #                 ROC_AUC Score: {avg_roc_auc:.2f}\n \
            #                 Precision: {avg_precision:.2f}\n \
            #                 Recall: {avg_recall:.2f}\n")

            # logger.info(f"ROC_AUC: {avg_roc_auc:.2f}")

            avg_val_loss = total_eval_loss / len(validation_generator)
            total_roc_auc = metrics('roc_auc', preds = val_preds_list, labels = val_labels_list)
            total_ap = metrics('ap', preds = val_preds_list, labels = val_labels_list)
            logger.info(f"  Average validation loss: {avg_val_loss:.3f}, Validation AUC: {total_roc_auc:.3f}, AP: {total_ap:.3f}")
            # save the model
            # TODO: add loss check, if < than previous loss save, else do nothing


            if avg_val_loss < best_eval_loss:
                best_eval_loss = avg_val_loss
                logging.info('Best Validation Loss: {}'.format(str(best_eval_loss)))

                # dd = {}
                # dd['embeddings'] = embeddings_list
                # dd['ids'] = val_ids_list
                # dd['labels'] = val_labels_list
                # # f = f'{TYPE}_sample_{str(SAMPLE)}_run_{str(run)}_epoch_{str(epoch)}_embeddings.p'
                # f = f'{TYPE}_sample_{str(SAMPLE)}_run_{str(run)}_epoch_{str(epoch)}_embeddings.p'
                # logging.info('Best Validation Loss: {}, Embeddings saved to {}'.format(str(best_eval_loss), f))
                # pickle.dump(dd, open(os.path.join('results', f), "wb" ))



            epoch_count.append(epoch)
            epoch_train_ap_list.append(total_train_ap)
            epoch_train_auroc_list.append(total_train_roc_auc)
            epoch_val_ap_list.append(total_ap)
            epoch_val_auroc_list.append(total_roc_auc)
            epoch_sample.append(SAMPLE) # because I sample size = SAMPLE from each class.
            epoch_train_loss_list.append(avg_train_loss)
            epoch_val_loss_list.append(avg_val_loss)
            run_list.append(run)
            epoch_val_ids.append(np.concatenate(val_ids_list))
            epoch_val_preds.append(val_preds_list)
            epoch_train_ids.append(np.concatenate(ids_list))
            epoch_train_preds.append(preds_list)
            pos_frac_list.append(POS_FRAC)
            epoch_val_labels.append(np.concatenate(val_labels_list))
            epoch_train_labels.append(np.concatenate(labels_list))

        # torch.save(model.state_dict(), ('scibert_'+TYPE+ '_' + str(epoch) +'.pt'))

        if runs == 1: break


        # model.module.save_state_dict('scibert_'+TYPE+'.pt')


    df = pd.DataFrame({
        'run': run_list,
        'epoch': epoch_count,
        'train_ap':epoch_train_ap_list,
        'train_auroc':epoch_train_auroc_list,
        'val_ap':epoch_val_ap_list,
        'val_auroc':epoch_val_auroc_list,
        'train_loss':epoch_train_loss_list,
        'val_loss':epoch_val_loss_list,
        'train_preds': None,
        'train_ids': None,
        'train_labels': None,
        'val_preds': None,
        'val_ids': None,
        'val_labels': None,
        'sample_size': epoch_sample,
        'pos_frac': pos_frac_list

    })


    df['val_preds'] = df['val_preds'].astype(object)
    df['val_ids'] = df['val_ids'].astype(object)
    df['val_labels'] = df['val_labels'].astype(object)
    df['train_ids'] = df['train_ids'].astype(object)
    df['train_preds'] = df['train_preds'].astype(object)
    df['train_labels'] = df['train_labels'].astype(object)

    df['val_preds'] = pd.Series(epoch_val_preds, index = df.index)
    df['val_ids'] = pd.Series(epoch_val_ids, index = df.index)
    df['val_labels'] = pd.Series(epoch_val_labels, index = df.index)
    df['train_preds'] = pd.Series(epoch_train_preds, index = df.index)
    df['train_ids'] = pd.Series(epoch_train_ids, index = df.index)
    df['train_labels'] = pd.Series(epoch_train_labels, index = df.index)

    out_res = os.path.join('results', TYPE)
    if not os.path.exists(out_res): os.makedirs(out_res)

    df.to_csv(os.path.join(out_res, f'{TYPE}_sample_{str(SAMPLE)}_{str(POS_FRAC)}_results.csv'))


# TODO - positive fraction sampling
# TODO - sampling by label

if __name__ == '__main__':
    main()
