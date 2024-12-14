#!/usr/bin/env python
# coding: utf-8

# In[49]:


from torch.utils.data import Dataset,DataLoader
import pandas as pd
from preprocess_data import load_sentiment_analysis_dataset,load_hate_speech_dataset
import random
from transformers import BertTokenizer, BertModel, DistilBertModel
import lightning as pl
import torch.nn as nn
import torch
from aim.pytorch_lightning import AimLogger


# In[3]:


class HateSpeechSentAnalysis(Dataset):
    def __init__(self,df_hate,df_sent):
        self.content_hate = df_hate["Content"]
        self.content_sent = df_sent["Content"]
        self.label_hate = df_hate["Label"]
        self.label_sent = df_sent["Label"]
        self.size = len(self.content_hate)
    def __getitem__(self,idx):
        return {
            "hate_content":self.content_hate.iloc[idx],
            "hate_label":self.label_hate.iloc[idx],
            "sent_content":self.content_sent.iloc[idx],
            "sent_label":self.label_sent.iloc[idx]
        }
    def __len__(self):
        return self.size


# In[5]:


train_sent,test_sent = load_sentiment_analysis_dataset()
train_hate,test_hate = load_hate_speech_dataset()


# In[37]:


reduced_HS_train = train_hate.sample(n=train_sent.shape[0],random_state=42)
reduced_HS_test = test_hate.sample(n=test_sent.shape[0],random_state=42)
trainParallelDataset = HateSpeechSentAnalysis(reduced_HS_train,train_sent)
testParallelDataset = HateSpeechSentAnalysis(reduced_HS_test,test_sent)


# In[38]:


# In[39]:


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def collator(dict_list):
    result_dict={
        "tokenized_hate_content":None,
        "hate_labels":[],
        "tokenized_sent_content":None,
        "sent_labels":[]
    }
    hate_list,sent_list=[],[]
    for D in dict_list:
        hate_list.append(D["hate_content"])
        result_dict["hate_labels"].append(D["hate_label"])
        result_dict["sent_labels"].append(D["sent_label"])
        sent_list.append(D["sent_content"])
    result_dict["tokenized_hate_content"]=tokenizer(hate_list,padding='longest',truncation=True,return_tensors='pt')
    result_dict["tokenized_sent_content"]=tokenizer(sent_list,padding='longest',truncation=True,return_tensors='pt')
    return result_dict


# In[54]:


seed_value=42
pl.seed_everything(seed_value)
batch_size = 16
trainLoader = DataLoader(trainParallelDataset,batch_size=batch_size,shuffle=True,collate_fn=collator)
testLoader = DataLoader(testParallelDataset,batch_size=batch_size,shuffle=False,collate_fn=collator)


# In[ ]:





# In[55]:


class TeacherBERTMultiTask(nn.Module):
    def __init__(self,num_sentiment_labels,num_hate_speech_labels):
        super().__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased",torch_dtype=torch.bfloat16)
        self.sentiment_head = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.bert_model.config.hidden_size, num_sentiment_labels)
        )
        self.hate_speech_head = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.bert_model.config.hidden_size, num_hate_speech_labels)
        )
    def forward(self,batch_dict):
        tokenized_hate = batch_dict["tokenized_hate_content"]
        tokenized_sent = batch_dict["tokenized_sent_content"]
        shared_encoder_sent = self.bert_model(input_ids=tokenized_sent["input_ids"],attention_mask=tokenized_sent["attention_mask"])
        shared_encoder_hate = self.bert_model(input_ids=tokenized_hate["input_ids"],attention_mask=tokenized_hate["attention_mask"])
        y_hat_sent = self.sentiment_head(shared_encoder_sent.pooler_output)
        y_hat_hate = self.hate_speech_head(shared_encoder_hate.pooler_output)
        return y_hat_sent,y_hat_hate


# In[56]:

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision("medium")
class MTLModule(pl.LightningModule):
    def __init__(self,num_sentiment_labels,num_hate_speech_labels):
        super().__init__()
        self.teacher_bert = TeacherBERTMultiTask(num_sentiment_labels,num_hate_speech_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    def compute_total_loss(self,batch_dict):
        sent_labels = torch.tensor(batch_dict["sent_labels"]).to(device)
        hate_labels = torch.tensor(batch_dict["hate_labels"]).to(device)
        y_hat_sent,y_hat_hate = self.teacher_bert(batch_dict)
        loss_sent = self.loss_fn(y_hat_sent,sent_labels)
        loss_hate = self.loss_fn(y_hat_hate,hate_labels)
        loss_total = loss_sent+loss_hate
        return loss_total
    def training_step(self,batch_dict,batch_idx):
        loss_total = self.compute_total_loss(batch_dict)
        self.log("Train_loss",loss_total,batch_size=batch_size,on_step=False,on_epoch=True)
        return loss_total
    def validation_step(self,batch_dict,batch_idx):
        loss_total = self.compute_total_loss(batch_dict)
        self.log("Test_loss",loss_total,batch_size=batch_size,on_step=False,on_epoch=True)
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=1e-5)
        return optimizer

MTLlog = AimLogger(experiment="Teacher_model_train",train_metric_prefix="Train_",val_metric_prefix="Test_")
checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(dirpath="Teacher_Model_CKPT")
pl_model = MTLModule(10,2)
trainer = pl.Trainer(max_epochs=10,logger=MTLlog,callbacks=[checkpoint_callback],precision="bf16")
trainer.fit(pl_model,trainLoader,testLoader)