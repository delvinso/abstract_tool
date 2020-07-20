import torch
from torch import nn
import numpy as np

from transformers import *
from torch.nn import CrossEntropyLoss

class BertForAbstractScreening(nn.Module):
  
    def __init__(self, vocab: str='bert-base-uncased', num_labels: int=2):
        """ BERT model with customizable layers for classification. 

        Keyword Arguments:
            pretrained_weights {str} -- pretrained weights to load BERT with (default: {'bert-base-uncased'})
            num_labels {int} -- number of labels for the data (default: {2})
        """          
        super(BertForAbstractScreening, self).__init__()
    
        self.num_labels = num_labels
        self.vocab = vocab
        # output: last_hidden_state, pooler_output, hidden_states


        config = AutoConfig.from_pretrained(vocab, output_hidden_states=True)
        self.bert = AutoModel.from_pretrained(vocab, config = config)
        # self.dropout = nn.Dropout(0.3)

        ########### NOTE, optional: add or change classifier on top of BERT here ###########
        # self.classifier = nn.Linear(config.hidden_size, num_labels)
        # nn.init.xavier_normal_(self.classifier.weight)
        ########### NOTE, optional: add or change classifier on top of BERT here ########### 

    def forward(self, input_ids, augment_ids = None, token_type_ids=None, attention_mask=None, labels=None, augmented=None):
        """ Forward method of the BERT model. 

        Arguments:
            input_ids {torch.tensor} -- unique identifier of the input.
            augment_ids {torch.tensor} -- unique identifier of the augmented input.

        Keyword Arguments:
            token_type_ids {torch.tensor} -- sentence type. (default: {None})
            attention_mask {torch.tensor} -- mask for padding. (default: {None})
            labels {torch.tensor} --  labels for the data. (default: {None})
            augmented {torch.tensor} -- metadata embeddings to augment the pooled output (default: {None})

        Returns:
            torch.tensor -- [description]
        """        
        _, pooled_output, hidden_states = self.bert(input_ids, token_type_ids, attention_mask)
        # pooled_output = self.dropout(pooled_output)
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        # returns last_hidden_state, pooler_output, hidden_states, attention
        # so we actually want hidden_states, not pooler_output
        # print('Last Hidden State: {}'.format(_.shape))
        # print('Pooler Output (CLS): {}'.format(pooled_output.shape))
        # print('Embeddings: {} Shape of Last State? \t{}'.format(str(len(hidden_states)), (hidden_states[-1].shape)))
        # print('\n')



    # TODO: better metric than average? get an average of all the items in the augmented embeddings
        # augment_ids = [x[0].clone().detach().requires_grad_(True) for x in augment_ids]
        #
        # avg_emb = torch.mean(torch.stack(augment_ids), dim=0)
        #
        # # concat the augementation to the pooled output
        # pooled_output_augmented = torch.cat((pooled_output, avg_emb), dim=1)

        # WITHOUT EMBEDDINGS


        ########### NOTE, OPTIONAL: send in the augmented embeddings through classifier ###########
        # logits = self.classifier(pooled_output_augmented)
        # return logits
        ########### NOTE, OPTIONAL: send in the augmented embeddings through classifier ###########

        # return pooled_output_augmented
        return pooled_output