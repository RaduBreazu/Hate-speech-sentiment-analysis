import torch
import torch.nn as nn
import torch.optim as optim

from transformers import BertTokenizer, BertModel

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocess_data import *

class MultitaskLearner(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, train_dataloader = None, test_dataloader = None, num_sentiment_labels = 2, num_hate_speech_labels = 2):
        super(MultitaskLearner, self).__init__()
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(MultitaskLearner.device)
        self.sentiment_head = nn.Sequential(
            nn.Dropout(0.1), # in order to have regularization
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_sentiment_labels)
        ).to(MultitaskLearner.device)
        self.hate_speech_head = nn.Sequential(
            nn.Dropout(0.3), # in order to have regularization
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_hate_speech_labels)
        ).to(MultitaskLearner.device)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids = input_ids.to(self.device), attention_mask = attention_mask.to(self.device))
        pooled_output = outputs.pooler_output  # we use [CLS] token representation
        sentiment_output = self.sentiment_head(pooled_output)
        hate_speech_output = self.hate_speech_head(pooled_output)
        return sentiment_output, hate_speech_output
    
    def train(self, num_epochs = 10):
        pass

    def predict(self):
        pass
    
def main():
    sa_train_dataloader, sa_test_dataloader = load_sentiment_analysis_dataset()
    hs_train_dataloader, hs_test_dataloader = load_hate_speech_dataset()
    print('All data loaded')

if __name__ == '__main__':
    main()