import torch
import torch.nn as nn
import torch.optim as optim

from transformers import BertTokenizer, BertModel

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocess_data import *

class MultitaskLearner(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def __init__(self, train_sentiment_analysis, test_sentiment_analysis, train_hate_speech, test_hate_speech, num_sentiment_labels, num_hate_speech_labels):
        super(MultitaskLearner, self).__init__()
        self.train_sentiment_analysis = train_sentiment_analysis
        self.test_sentiment_analysis = test_sentiment_analysis
        self.train_hate_speech = train_hate_speech
        self.test_hate_speech = test_hate_speech
        
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
    
    def encode(data : List[str]) -> Tuple[torch.tensor, torch.tensor]:
        input_ids = []
        attention_mask = []

        for text in data:
            tokenized_text = MultitaskLearner.tokenizer.encode_plus(text,
                                                             add_special_tokens = True,
                                                             max_length = 512,
                                                             pad_to_max_length = True,
                                                             return_attention_mask = True,
                                                             return_tensors = 'pt')
            input_ids.append(tokenized_text['input_ids'])
            attention_mask.append(tokenized_text['attention_mask'])

        return torch.tensor(input_ids, dtype = torch.long), torch.tensor(attention_mask, dtype = torch.long)
    
    def train(self, num_epochs = 10):
        optimizer = optim.Adam(self.parameters(), lr = 1e-5)
        criterion = nn.CrossEntropyLoss()

        self.train()
        for _ in range(num_epochs):
            # we cannot iterate simultaneously over the two datasets, since they do not have the same number of batches
            for _, (X_sentiment, y_sentiment) in self.train_sentiment_analysis:
                input_ids, attention_mask = MultitaskLearner.encode(X_sentiment)
                y_sentiment = y_sentiment.to(MultitaskLearner.device)

                optimizer.zero_grad()
                sentiment_output, _ = self(input_ids, attention_mask)
                sentiment_loss = criterion(sentiment_output, y_sentiment)
                sentiment_loss.backward()
                optimizer.step()

            for _, (X_hate_speech, y_hate_speech) in self.train_hate_speech:
                input_ids, attention_mask = MultitaskLearner.encode(X_hate_speech)
                y_hate_speech = y_hate_speech.to(MultitaskLearner.device)

                optimizer.zero_grad()
                _, hate_speech_output = self(input_ids, attention_mask)
                hate_speech_loss = criterion(hate_speech_output, y_hate_speech)
                hate_speech_loss.backward()
                optimizer.step()

    def predict_sentiment_analysis(self, test_data : DataLoader):
        self.eval()
        sentiment_predictions = []
        true_values = []
        with torch.no_grad():
            for _, (X_sentiment, y_sentiment) in test_data:
                input_ids, attention_mask = MultitaskLearner.encode(X_sentiment)
                sentiment_output, _ = self(input_ids, attention_mask)
                sentiment_predictions.append(sentiment_output.argmax(dim = 1).cpu().numpy())
                true_values.append(y_sentiment.numpy())
        
        predictions = np.concatenate(sentiment_predictions)
        print(classification_report(true_values, predictions))

    def predict_hate_speech(self, test_data : DataLoader):
        predictions = []
        true_values = []
        self.eval()

        with torch.no_grad():
            for _, (X_hate_speech, y_hate_speech) in test_data:
                input_ids, attention_mask = MultitaskLearner.encode(X_hate_speech)
                _, hate_speech_output = self(input_ids, attention_mask)
                predictions.append(hate_speech_output.argmax(dim = 1).cpu().numpy())
                true_values.append(y_hate_speech.numpy())
                
        precision = precision_score(true_values, predictions)
        recall = recall_score(true_values, predictions)
        f1 = f1_score(true_values, predictions)
        print(f'Precision: {(precision * 100):.2f}, Recall: {(recall * 100):.2f}, F1 score: {(f1 * 100):.2f}')
        print(classification_report(true_values, predictions))
    
def main():
    sa_train_dataloader, sa_test_dataloader = load_sentiment_analysis_dataset()
    hs_train_dataloader, hs_test_dataloader = load_hate_speech_dataset()
    print('All data loaded')

if __name__ == '__main__':
    main()