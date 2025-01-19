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
from torchmetrics.classification import MulticlassAccuracy,MulticlassF1Score
import transformers
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
acc_sent = MulticlassAccuracy(num_classes=10).to(device)
acc_hate = MulticlassAccuracy(num_classes=2).to(device)
f1_sent = MulticlassF1Score(num_classes=10).to(device)
f1_hate = MulticlassF1Score(num_classes=2).to(device)


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


class TeacherBERTMultiTask(nn.Module):
    def __init__(self,num_sentiment_labels,num_hate_speech_labels):
        super().__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased",torch_dtype=torch.bfloat16)
        self.sentiment_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.bert_model.config.hidden_size, num_sentiment_labels,dtype=torch.bfloat16)
        )
        self.hate_speech_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.bert_model.config.hidden_size, num_hate_speech_labels,dtype=torch.bfloat16)
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

batch_size = 16
num_epochs=10
class MTLModule(pl.LightningModule):
    def __init__(self,num_sentiment_labels,num_hate_speech_labels,dataloader_len):
        super().__init__()
        self.save_hyperparameters()
        self.teacher_bert = TeacherBERTMultiTask(num_sentiment_labels,num_hate_speech_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        #self.automatic_optimization = False
        self.dataloader_len = dataloader_len
    def compute_metrics(self,y_hat_sent,sent_labels,y_hat_hate,hate_labels,mode):
        self.log(f"{mode}_accuracy_sent",acc_sent(y_hat_sent,sent_labels),batch_size=batch_size,on_step=False,on_epoch=True)
        self.log(f"{mode}_accuracy_hate",acc_hate(y_hat_hate,hate_labels),batch_size=batch_size,on_step=False,on_epoch=True)
        self.log(f"{mode}_f1_sent",f1_sent(y_hat_sent,sent_labels),batch_size=batch_size,on_step=False,on_epoch=True)
        self.log(f"{mode}_f1_hate",f1_hate(y_hat_hate,hate_labels),batch_size=batch_size,on_step=False,on_epoch=True)
    def compute_total_loss(self,batch_dict,mode):
        sent_labels = torch.tensor(batch_dict["sent_labels"]).to(device)
        hate_labels = torch.tensor(batch_dict["hate_labels"]).to(device)
        y_hat_sent,y_hat_hate = self.teacher_bert(batch_dict)
        self.compute_metrics(y_hat_sent,sent_labels,y_hat_hate,hate_labels,mode)
        loss_sent = self.loss_fn(y_hat_sent,sent_labels)
        loss_hate = self.loss_fn(y_hat_hate,hate_labels)
        loss_total = loss_sent+loss_hate
        return loss_total
    def training_step(self,batch_dict,batch_idx):
        #opt = self.optimizers()
        #sch = self.lr_schedulers()
        #opt.zero_grad()
        loss_total = self.compute_total_loss(batch_dict,"Train")
        #self.manual_backward(loss_total)
        #opt.step()
        self.log("Train_loss",loss_total,batch_size=batch_size,on_step=False,on_epoch=True)
        return loss_total
    def validation_step(self,batch_dict,batch_idx):
        loss_total = self.compute_total_loss(batch_dict,"Test")
        self.log("Test_loss",loss_total,batch_size=batch_size,on_step=False,on_epoch=True)
    def configure_optimizers(self):
        #optimizer = torch.optim.SGD(self.parameters(),lr=1e-2,momentum=0.9,weight_decay=0.01)
        optimizer = torch.optim.AdamW(self.parameters(),lr=1e-4,weight_decay=1e-2)
        #scheduler = transformers.get_linear_schedule_with_warmup(optimizer,500,self.dataloader_len*num_epochs)
        return optimizer
        #return [optimizer],[scheduler]

if __name__=='__main__':
    
    train_sent,test_sent = pd.read_csv("sent_train.csv"),pd.read_csv("sent_test.csv")
    train_hate,test_hate = pd.read_csv("HS_train.csv"),pd.read_csv("HS_test.csv")
    train_sent.fillna(value="",inplace=True)
    test_sent.fillna(value="",inplace=True)
    trainParallelDataset = HateSpeechSentAnalysis(train_hate,train_sent)
    testParallelDataset = HateSpeechSentAnalysis(test_hate,test_sent)

    seed_value=42
    pl.seed_everything(seed_value)
    
    trainLoader = DataLoader(trainParallelDataset,batch_size=batch_size,shuffle=True,collate_fn=collator)
    testLoader = DataLoader(testParallelDataset,batch_size=batch_size,shuffle=False,collate_fn=collator)

    MTLlog = AimLogger(experiment="Teacher_model_train",train_metric_prefix="Train_",val_metric_prefix="Test_")
    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(dirpath="Teacher_Model_CKPT")
    pl_model = MTLModule(10,2,math.ceil(len(trainParallelDataset)/batch_size))
    trainer = pl.Trainer(max_epochs=num_epochs,logger=MTLlog,callbacks=[checkpoint_callback])
    trainer.fit(pl_model,trainLoader,testLoader)