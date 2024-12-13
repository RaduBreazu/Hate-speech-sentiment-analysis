import os
import numpy as np
import pandas as pd

from typing import Tuple, List, Dict

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

from langdetect import detect
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
    
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return 'unknown'

def load_sentiment_analysis_dataset() -> Tuple[DataLoader, DataLoader]:
    DATA_LOCATION = get_data_location() + '/sentiment_analysis_dataset/'
    train_df = pd.DataFrame(columns = ['Content', 'Label'])
    for file in os.listdir(DATA_LOCATION + 'train/pos'):
        with open(DATA_LOCATION + 'train/pos/' + file, 'r') as f:
            label = file.split('_')[1].split('.')[0]
            content = f.read()
            
            index = content.find('Rating') # filter out the rating from the content
            if index != -1:
                content = content[:index]
            
            train_df = pd.concat([train_df, pd.DataFrame([{'Content': content, 'Label': label}])], ignore_index = True)

    for file in os.listdir(DATA_LOCATION + 'train/neg'):
        with open(DATA_LOCATION + 'train/neg/' + file, 'r') as f:
            label = file.split('_')[1].split('.')[0]
            content = f.read()

            index = content.find('Rating') # filter out the rating from the content
            if index != -1:
                content = content[:index]

            train_df = pd.concat([train_df, pd.DataFrame([{'Content': content, 'Label': label}])], ignore_index = True)

    train_df['Content'] = train_df['Content'].apply(demojize)

    test_df = pd.DataFrame(columns = ['Content', 'Label'])
    for file in os.listdir(DATA_LOCATION + 'test/pos'):
        with open(DATA_LOCATION + 'test/pos/' + file, 'r') as f:
            label = file.split('_')[1].split('.')[0]
            content = f.read()

            index = content.find('Rating') # filter out the rating from the content
            if index != -1:
                content = content[:index]
            
            train_df = pd.concat([train_df, pd.DataFrame([{'Content': content, 'Label': label}])], ignore_index = True)

    for file in os.listdir(DATA_LOCATION + 'test/neg'):
        with open(DATA_LOCATION + 'test/neg/' + file, 'r') as f:
            label = file.split('_')[1].split('.')[0]
            content = f.read()

            index = content.find('Rating') # filter out the rating from the content
            if index != -1:
                content = content[:index]
            
            train_df = pd.concat([train_df, pd.DataFrame([{'Content': content, 'Label': label}])], ignore_index = True)

    test_df['Content'] = test_df['Content'].apply(demojize)
    
    train_dataloader = DataLoader(SentimentAnalysisDataset(train_df), batch_size = 32, shuffle = True)
    test_dataloader = DataLoader(SentimentAnalysisDataset(test_df), batch_size = 32, shuffle = False)
    return train_dataloader, test_dataloader

def load_hate_speech_dataset() -> Tuple[DataLoader, DataLoader]:
    FILE = get_data_location() + '/hate_speech_dataset/HateSpeechDataset.csv'
    df = pd.read_csv(FILE)
    df = df.dropna().drop_duplicates()
    df['Content'] = df['Content'].apply(demojize) # convert emojis to text
    df = shuffle(df, random_state = 42) # randomly shuffle the dataset
    train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42) # split the dataset into training and testing sets
    train_dataloader = DataLoader(HateSpeechDataset(train_df), batch_size = 128, shuffle = True)
    test_dataloader = DataLoader(HateSpeechDataset(test_df), batch_size = 128, shuffle = False)
    return train_dataloader, test_dataloader