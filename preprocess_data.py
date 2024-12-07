import os
import numpy as np
import pandas as pd

from typing import Tuple, List, Dict

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

from emoji import demojize

class SentimentAnalysisDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row['Content'], row['Label']

class HateSpeechDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row['Content'], row['Label']

def get_data_location():
    if os.getcwd() == '/kaggle/working':
        return '/kaggle/input'
    else:
        return './datasets'

def load_sentiment_analysis_dataset() -> Tuple[DataLoader, DataLoader]:
    DATA_LOCATION = get_data_location() + '/sentiment_analysis_dataset/'
    train_df = pd.DataFrame(columns = ['Content', 'Label'])
    with open(DATA_LOCATION + 'train.ft.txt', 'r') as f:
        for line in f:
            label = 0 if line.split()[0] == '__label__1' else 1
            content = ' '.join(line.split()[1:])
            train_df = train_df.append({'Content': content, 'Label': label}, ignore_index = True)

    train_df['Content'] = train_df['Content'].apply(demojize)

    test_df = pd.DataFrame(columns = ['Content', 'Label'])
    with open(DATA_LOCATION + 'test.ft.txt', 'r') as f:
        for line in f:
            label = 0 if line.split()[0] == '__label__1' else 1
            content = ' '.join(line.split()[1:])
            test_df = test_df.append({'Content': content, 'Label': label}, ignore_index = True)

    test_df['Content'] = test_df['Content'].apply(demojize)
    
    train_dataloader = DataLoader(SentimentAnalysisDataset(train_df), batch_size = 128, shuffle = True)
    test_dataloader = DataLoader(SentimentAnalysisDataset(test_df), batch_size = 128, shuffle = False)
    return train_dataloader, test_dataloader

def load_hate_speech_dataset() -> Tuple[DataLoader, DataLoader]:
    FILE = get_data_location() + '/hate_speech_dataset/HateSpeechDataset.csv'
    df = pd.read_csv(FILE)
    df = df.dropna().drop_duplicates()
    df['text'] = df['text'].apply(demojize) # convert emojis to text
    df = shuffle(df, random_state = 42) # randomly shuffle the dataset
    train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42) # split the dataset into training and testing sets
    train_dataloader = DataLoader(HateSpeechDataset(train_df), batch_size = 128, shuffle = True)
    test_dataloader = DataLoader(HateSpeechDataset(test_df), batch_size = 128, shuffle = False)
    return train_dataloader, test_dataloader