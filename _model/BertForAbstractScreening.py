import torch
import numpy as np
import torchtext 
from transformers import *
from torch.nn import CrossEntropyLoss

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
        self.glove = torchtext.vocab.GloVe(name="6B", dim=50)  

        ########### NOTE: add or change classifier on top of BERT here ###########
        # self.classifier = nn.Linear(config.hidden_size, num_labels)

        nn.init.xavier_normal_(self.classifier.weight)


    def forward(self, input_ids, augment_ids, token_type_ids=None, attention_mask=None, labels=None, augmented=None):
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
        pooled_output = self.dropout(pooled_output)

        augmented_emb = []

        for element in augment_list: 
            augmented_emb.append(self.glove(element))
        
        # get an average of all the items in the augmented embeddings
        avg_emb = torch.mean(torch.stack(augmented_emb), dim=0)

        # concat the augementation to the pooled output  
        pooled_output_augmented = torch.cat((pooled_output, avg_emb), dim=0)

        ########### NOTE, OPTIONAL: send in the augmented embeddings through classifier ###########
        # logits = self.classifier(pooled_output_augmented)
        # return logits

        return pooled_output_augmented