import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Tuple, List, Dict

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

from fast_langdetect import detect
from emoji import demojize
import re
from tqdm import tqdm


def text_cleanup(text:str):
    text = demojize(text)
    text = re.sub(r"<[A-Za-z0-9 /;:\"]*>",r" ",text)
    text = re.sub(r"([a-zA-Z0-9])\(",r"\1 (",text)
    text = re.sub(r"[*#@]","",text)
    text = re.sub(r"(\. ){2,}",r". ",text)
    text = re.sub(r"(\- ){2,}",r"",text)
    text = re.sub(r"[^a-zA-Z0-9.,;:\"' ()\-,!?/]",r" ",text)
    text = re.sub(r"([a-zA-Z0-9]) {1,}([.,;:\"\-])",r"\1\2",text)
    text = re.sub(r"([a-zA-Z0-9])([\"])([a-zA-Z0-9])",r"\1 \2\3",text)
    text = re.sub(r"([a-zA-Z])- ",r"\1 - ",text)
    text = re.sub(r" {2,}",r" ",text)
    return text

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

def load_sentiment_analysis_dataset() -> Tuple[pd.DataFrame,pd.DataFrame]:
    DATA_LOCATION = get_data_location() + '/sentiment_analysis_dataset/'
    train_df = pd.DataFrame(columns = ['Content', 'Label'])
    for file in tqdm(os.listdir(DATA_LOCATION + 'train/pos'),desc="Train_pos_load"):
        with open(DATA_LOCATION + 'train/pos/' + file, 'r') as f:
            label = file.split('_')[1].split('.')[0]
            content = f.read()
            
            index = content.find('Rating') # filter out the rating from the content
            if index != -1:
                content = content[:index]
            if(detect(content)["lang"]=="en"):
                train_df = pd.concat([train_df, pd.DataFrame([{'Content': content, 'Label': int(label)-1}])], ignore_index = True)

    for file in tqdm(os.listdir(DATA_LOCATION + 'train/neg'),desc="Train_neg_load"):
        with open(DATA_LOCATION + 'train/neg/' + file, 'r') as f:
            label = file.split('_')[1].split('.')[0]
            content = f.read()

            index = content.find('Rating') # filter out the rating from the content
            if index != -1:
                content = content[:index]
            if(detect(content)["lang"]=="en"):
                train_df = pd.concat([train_df, pd.DataFrame([{'Content': content, 'Label': int(label)-1}])], ignore_index = True)

    train_df['Content'] = train_df['Content'].apply(text_cleanup)

    test_df = pd.DataFrame(columns = ['Content', 'Label'])
    for file in tqdm(os.listdir(DATA_LOCATION + 'test/pos'),desc="Test_pos_load"):
        with open(DATA_LOCATION + 'test/pos/' + file, 'r') as f:
            label = file.split('_')[1].split('.')[0]
            content = f.read()

            index = content.find('Rating') # filter out the rating from the content
            if index != -1:
                content = content[:index]
            if(detect(content)["lang"]=="en"):
                test_df = pd.concat([test_df, pd.DataFrame([{'Content': content, 'Label': int(label)-1}])], ignore_index = True)

    for file in tqdm(os.listdir(DATA_LOCATION + 'test/neg'),desc="Test_neg_load"):
        with open(DATA_LOCATION + 'test/neg/' + file, 'r') as f:
            label = file.split('_')[1].split('.')[0]
            content = f.read()

            index = content.find('Rating') # filter out the rating from the content
            if index != -1:
                content = content[:index]
            if(detect(content)["lang"]=="en"):
                test_df = pd.concat([test_df, pd.DataFrame([{'Content': content, 'Label': int(label)-1}])], ignore_index = True)

    test_df['Content'] = test_df['Content'].apply(text_cleanup)
    
    return train_df,test_df

def load_hate_speech_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    FILE = get_data_location() + '/hate_speech_dataset/HateSpeechDataset.csv'
    df = pd.read_csv(FILE)
    df = df.dropna().drop_duplicates()
    df['Content'] = df['Content'].apply(text_cleanup) # convert emojis to text
    df = shuffle(df, random_state = 42) # randomly shuffle the dataset
    train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42) # split the dataset into training and testing sets
    #train_dataloader = DataLoader(HateSpeechDataset(train_df), batch_size = 128, shuffle = True)
    #test_dataloader = DataLoader(HateSpeechDataset(test_df), batch_size = 128, shuffle = False)
    return train_df,test_df
