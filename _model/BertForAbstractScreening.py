import torch 
from transformers import *
from torch.nn import CrossEntropyLoss
import numpy as np

class BertForAbstractScreening(nn.Module):
  
    def __init__(self, pretrained_weights: str='bert-base-uncased', num_labels: int=2):
        """ BERT model with customizable layers for classification. 

        Keyword Arguments:
            pretrained_weights {str} -- pretrained weights to load BERT with (default: {'bert-base-uncased'})
            num_labels {int} -- number of labels for the data (default: {2})
        """          
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        # output: last_hidden_state, pooler_output, hidden_states
        self.bert = BertModel.from_pretrained(pretrained_weights, output_hidden_states=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # can change classifier here 
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        nn.init.xavier_normal_(self.classifier.weight)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, augmented=None):
        """ Forward method of the BERT model. 

        Arguments:
            input_ids {torch.tensor} -- unique identifier of the input.

        Keyword Arguments:
            token_type_ids {torch.tensor} -- sentence type. (default: {None})
            attention_mask {torch.tensor} -- mask for padding. (default: {None})
            labels {torch.tensor} --  labels for the data. (default: {None})
            augmented {torch.tensor} -- metadata embeddings to augment the pooled output (default: {None})

        Returns:
            torch.tensor -- [description]
        """        
        _, pooled_output, hidden_states = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        # send in the augmented embeddings here
        logits = self.classifier(pooled_output)

        return logits