import os
import string
import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Tuple, List, Dict

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

from fast_langdetect import detect
from emoji import demojize, replace_emoji
import wordninja # source: https://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words
import re
from tqdm import tqdm


def text_cleanup(text:str):
    text = text.lower()
    text = demojize(text)
    text = re.sub(r"<[a-z0-9 /;:\"]*>",r" ",text)
    text = re.sub(r"([a-z0-9])\(",r"\1 (",text)
    text = re.sub(r"(\. ){2,}",r". ",text)
    text = re.sub(r"(\- ){2,}",r"",text)
    text = re.sub(r"[^a-z0-9.,;:\"' ()\-,!?/]",r" ",text)
    text = re.sub(r" {2,}",r" ",text)
    return text

def text_cleanup_hate_speech(text : str) -> str:
    text = replace_emoji(text, replace = "<emoticon>")
    text = re.sub(r"[+-]?[1-9][0-9]*\.[0-9]*", r"<number>", text) # replace numbers with <number>
    text = re.sub(r"@\w+", r"<user>", text) # usernames begin with @
    text = re.sub(r"#(\w+)", r"<hashtag> \1", text) # hashtags begin with #
    text = re.sub(r"(\w)\1{2,}", r"\1", text) # remove repeated characters (naive implementation)
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    text = text.lower() # convert text to lowercase
    
    lst = text.split()
    for i in range(len(lst)):
        if any(lst[i].startswith(x) for x in ['http', 'https', 'www']):
            lst[i] = "<url>"

    text = " ".join(lst)
    text = " ".join(wordninja.split(text)) # tokenize words that are not separated by spaces
    text = text.strip() # remove leading and trailing whitespaces
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
        return row['tweet'], row['class'] # on the dataset that we had before: row['Content'], row['Label']
    

class CombinedHSSent(Dataset):
    def __init__(self,df_hate,df_sent):
        self.hate = df_hate
        self.sent = df_sent
        self.true_len = min(len(self.hate),len(self.sent))
    def __getitem__(self,idx):
        return self.hate[idx],self.sent[idx]
    def __len__(self):
        return self.true_len

def get_data_location():
    if os.getcwd() == '/kaggle/working':
        return '/kaggle/input'
    else:
        return './datasets'

def load_sentiment_analysis_dataset(batch_size:int) -> Tuple[pd.DataFrame,pd.DataFrame]:
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
                train_df = pd.concat([train_df, pd.DataFrame([{'Content': content, 'Label': 1}])], ignore_index = True)

    for file in tqdm(os.listdir(DATA_LOCATION + 'train/neg'),desc="Train_neg_load"):
        with open(DATA_LOCATION + 'train/neg/' + file, 'r') as f:
            label = file.split('_')[1].split('.')[0]
            content = f.read()

            index = content.find('Rating') # filter out the rating from the content
            if index != -1:
                content = content[:index]
            if(detect(content)["lang"]=="en"):
                train_df = pd.concat([train_df, pd.DataFrame([{'Content': content, 'Label': 0}])], ignore_index = True)

    train_df['Content'] = train_df['Content'].apply(text_cleanup)

    test_df = pd.DataFrame(columns = ['Content', 'Label'])
    for i,file in tqdm(enumerate(os.listdir(DATA_LOCATION + 'test/pos')),desc="Test_pos_load"):
        with open(DATA_LOCATION + 'test/pos/' + file, 'r') as f:
            label = file.split('_')[1].split('.')[0]
            content = f.read()

            index = content.find('Rating') # filter out the rating from the content
            if index != -1:
                content = content[:index]
            if(detect(content)["lang"]=="en"):
                test_df = pd.concat([test_df, pd.DataFrame([{'Content': content, 'Label': 1}])], ignore_index = True)

    for i,file in tqdm(enumerate(os.listdir(DATA_LOCATION + 'test/neg')),desc="Test_neg_load"):
        with open(DATA_LOCATION + 'test/neg/' + file, 'r') as f:
            label = file.split('_')[1].split('.')[0]
            content = f.read()

            index = content.find('Rating') # filter out the rating from the content
            if index != -1:
                content = content[:index]
            if(detect(content)["lang"]=="en"):
                test_df = pd.concat([test_df, pd.DataFrame([{'Content': content, 'Label': 0}])], ignore_index = True)

    test_df['Content'] = test_df['Content'].apply(text_cleanup)
    train_df = train_df.sample(frac=1.0,random_state=42)
    test_df = test_df.sample(frac=1.0,random_state=42)
    train_loader = DataLoader(SentimentAnalysisDataset(train_df),batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(SentimentAnalysisDataset(test_df),batch_size=batch_size,shuffle=False)
    return SentimentAnalysisDataset(train_df),SentimentAnalysisDataset(test_df)

"""
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
"""

def load_hate_speech_dataset(batch_size:int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    FILE = get_data_location() + '/hate_speech_dataset/data/labeled_data.csv'
    df = pd.read_csv(FILE)
    df = df.drop(columns = ['count', 'hate_speech', 'offensive_language', 'neither']).dropna().drop_duplicates()
    df['tweet'] = df['tweet'].apply(text_cleanup_hate_speech) # preprocess the text in every tweet
    df = shuffle(df, random_state = 42) # randomly shuffle the dataset
    train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42) # split the dataset into training and testing sets
    train_dataloader = DataLoader(HateSpeechDataset(train_df), batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(HateSpeechDataset(test_df), batch_size = batch_size, shuffle = False)
    return HateSpeechDataset(train_df),HateSpeechDataset(test_df)

if __name__=='__main__':
    train_sent,test_sent = load_sentiment_analysis_dataset()
    train_hate,test_hate = load_hate_speech_dataset()
    train_sent.dropna(inplace=True)
    test_sent.dropna(inplace=True)
    train_hate.dropna(inplace=True)
    test_hate.dropna(inplace=True)
    reduced_HS_train = train_hate.sample(n=len(train_sent),random_state=42)
    reduced_HS_test = test_hate.sample(n=len(test_sent),random_state=42)
    reduced_HS_train.to_csv("HS_train.csv")
    reduced_HS_test.to_csv("HS_test.csv")
    train_sent.to_csv("sent_train.csv")
    test_sent.to_csv("sent_test.csv")